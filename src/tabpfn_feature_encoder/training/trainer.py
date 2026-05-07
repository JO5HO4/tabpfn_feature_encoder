from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from tabpfn_feature_encoder.config import EncoderConfig
from tabpfn_feature_encoder.data.base import to_label_vector, to_numpy_matrix
from tabpfn_feature_encoder.data.graphs import EventGraphDataset, GraphStandardizer
from tabpfn_feature_encoder.data.preprocessing import Standardizer
from tabpfn_feature_encoder.evaluation.metrics import accuracy, log_loss, roc_auc
from tabpfn_feature_encoder.models.encoders import build_encoder, require_torch
from tabpfn_feature_encoder.models.tabpfn_adapter import TabPFNPromptAdapter
from tabpfn_feature_encoder.training.episodes import RatioEpisodeSampler


@dataclass
class TrainingRecord:
    epoch: int
    batch: int
    query_loss: float
    query_accuracy: float
    batch_size: int
    support_size: int
    query_size: int


@dataclass
class EvaluationRecord:
    epoch: int
    context_size: int
    query_size: int
    metrics: dict[str, float]


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    train_roc_auc: float
    val_loss: float | None
    val_accuracy: float | None
    val_roc_auc: float | None
    val_p1_mean: float | None
    val_p1_std: float | None
    val_context_size: int | None
    val_query_size: int | None
    batches: int


@dataclass
class EncoderTabPFNClassifier:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    device: str = "cpu"
    random_state: int = 42

    encoder_model_: Any = None
    tabpfn_adapter_: TabPFNPromptAdapter | None = None
    standardizer_: Standardizer = field(default_factory=Standardizer)
    graph_standardizer_: GraphStandardizer = field(default_factory=GraphStandardizer)
    X_train_: Any | None = None
    y_train_: np.ndarray | None = None
    classes_: np.ndarray | None = None
    history_: list[TrainingRecord] = field(default_factory=list)
    evaluation_history_: list[EvaluationRecord] = field(default_factory=list)
    epoch_metrics_: list[EpochMetrics] = field(default_factory=list)
    best_epoch_: int | None = None
    best_query_loss_: float | None = None
    latest_evaluation_: dict[str, float] | None = None

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_eval: Any | None = None,
        y_eval: Any | None = None,
        eval_metrics: list[str] | None = None,
    ) -> "EncoderTabPFNClassifier":
        torch_mod, _ = require_torch()
        self._seed_everything()

        y_np = to_label_vector(y_train)
        is_graph_input = isinstance(X_train, EventGraphDataset)
        if self.encoder.type.lower() in {"gnn", "graph", "graph_gnn"} and not is_graph_input:
            raise TypeError("encoder.type `gnn` requires an EventGraphDataset input.")
        if is_graph_input and self.encoder.type.lower() not in {"gnn", "graph", "graph_gnn"}:
            raise TypeError("EventGraphDataset input requires encoder.type `gnn`.")

        if len(X_train) != len(y_np):
            raise ValueError("X_train and y_train length mismatch.")

        device = self._effective_device()
        global_dim = 0
        if is_graph_input:
            X_model = self.graph_standardizer_.fit_transform(X_train)
            input_dim = X_model.node_dim
            global_dim = X_model.global_dim
        else:
            X_np = to_numpy_matrix(X_train)
            X_model = self.standardizer_.fit_transform(X_np)
            input_dim = X_model.shape[1]
        self.X_train_ = X_model
        self.y_train_ = y_np
        self.classes_ = np.unique(y_np)
        self.history_ = []
        self.evaluation_history_ = []
        self.epoch_metrics_ = []
        self.best_epoch_ = None
        self.best_query_loss_ = None
        self.latest_evaluation_ = None
        X_eval_std = None
        y_eval_np = None
        if X_eval is not None and y_eval is not None:
            y_eval_np = to_label_vector(y_eval)
            if is_graph_input:
                if not isinstance(X_eval, EventGraphDataset):
                    raise TypeError("Graph training requires graph validation input.")
                X_eval_std = self.graph_standardizer_.transform(X_eval)
            else:
                X_eval_std = self.standardizer_.transform(to_numpy_matrix(X_eval))
            if len(X_eval_std) != len(y_eval_np):
                raise ValueError("X_eval and y_eval length mismatch.")

        self.encoder_model_ = build_encoder(
            encoder_type=self.encoder.type,
            input_dim=input_dim,
            global_dim=global_dim,
            hidden_dim=self.encoder.hidden_dim,
            output_dim=self.encoder.output_dim,
            layers=self.encoder.layers,
            residual_scale=self.encoder.residual_scale,
        ).to(device)
        self.tabpfn_adapter_ = TabPFNPromptAdapter(device=device).build()

        optimizer = torch_mod.optim.Adam(
            self.encoder_model_.parameters(),
            lr=self.encoder.learning_rate,
        )
        sampler = RatioEpisodeSampler(
            support_query_ratio=self.encoder.support_query_ratio,
            random_state=self.random_state,
        )
        rng = np.random.default_rng(self.random_state)
        batch_size = max(1, int(self.encoder.batch_size))
        validation_random_state = self.random_state + 10_000

        eval_metric_names = self._epoch_eval_metrics(eval_metrics)
        best_state = None
        best_loss = float("inf")
        best_val_roc_auc = float("-inf")
        best_eval_metrics: dict[str, float] | None = None
        epochs_without_improvement = 0
        use_validation_for_best = X_eval_std is not None and y_eval_np is not None
        settings = [
            f"type={self.encoder_model_.encoder_type}",
            f"device={device}",
            f"output_dim={self.encoder.output_dim}",
            f"batch_size={batch_size}",
            f"support_query_ratio={self.encoder.support_query_ratio}",
            f"identity_residual={self.encoder_model_.uses_identity_residual}",
            f"residual_scale={self.encoder_model_.residual_scale}",
            f"identity_weight={self.encoder.identity_weight}",
            f"grad_clip_norm={self.encoder.grad_clip_norm}",
            f"early_stopping_patience={self.encoder.early_stopping_patience}",
        ]
        if self.encoder_model_.encoder_type in {"residual_mlp", "gnn"}:
            settings.insert(2, f"layers={self.encoder.layers}")
            settings.insert(3, f"hidden_dim={self.encoder.hidden_dim}")
        if getattr(self.encoder_model_, "summary_dim", 0):
            settings.append(f"summary_dim={self.encoder_model_.summary_dim}")
            settings.append(f"learned_dim={self.encoder_model_.learned_dim}")
        print(f"EncoderTabPFN settings: {', '.join(settings)}")
        if X_eval_std is not None and y_eval_np is not None:
            initial_eval = self._evaluate_standardized_context_query(
                X_eval_std,
                y_eval_np,
                metrics=eval_metric_names,
                random_state=validation_random_state,
                max_samples=batch_size,
                use_full_query=True,
            )
            initial_metrics_text = ", ".join(
                f"val_{key}={value:.3f}"
                for key, value in initial_eval["metrics"].items()
            )
            print(
                "initial val: "
                f"context={initial_eval['context_size']}, "
                f"query={initial_eval['query_size']}, "
                f"{initial_metrics_text}"
            )
            best_state = self._encoder_state_cpu()
            best_eval_metrics = dict(initial_eval["metrics"])
            best_val_roc_auc = float(initial_eval["metrics"].get("roc_auc", float("-inf")))
            self.best_epoch_ = 0
            self.best_query_loss_ = float(initial_eval["metrics"].get("log_loss", float("nan")))

        for epoch_idx in range(max(1, self.encoder.epochs)):
            epoch_losses: list[float] = []
            epoch_y_parts: list[np.ndarray] = []
            epoch_proba_parts: list[np.ndarray] = []
            shuffled_idx = rng.permutation(len(y_np))
            n_batches = int(np.ceil(len(shuffled_idx) / batch_size))

            for batch_idx, start in enumerate(range(0, len(shuffled_idx), batch_size), start=1):
                batch_indices = shuffled_idx[start : start + batch_size]
                _, batch_counts = np.unique(y_np[batch_indices], return_counts=True)
                if len(batch_counts) < 2 or np.min(batch_counts) < 2:
                    continue

                episode = sampler.sample(y_np[batch_indices])
                support_idx = batch_indices[episode.support_idx]
                query_idx = batch_indices[episode.query_idx]

                x_support = self._model_input(X_model, support_idx, device)
                y_support = torch_mod.tensor(
                    y_np[support_idx],
                    dtype=torch_mod.long,
                    device=device,
                )
                x_query = self._model_input(X_model, query_idx, device)
                y_query = torch_mod.tensor(
                    y_np[query_idx],
                    dtype=torch_mod.long,
                    device=device,
                )

                self.encoder_model_.train()
                optimizer.zero_grad(set_to_none=True)
                loss, query_proba = self._episode_step(
                    x_support,
                    y_support,
                    x_query,
                    y_query,
                )
                loss.backward()
                if self.encoder.grad_clip_norm > 0.0:
                    torch_mod.nn.utils.clip_grad_norm_(
                        self.encoder_model_.parameters(),
                        max_norm=self.encoder.grad_clip_norm,
                    )
                optimizer.step()
                self.tabpfn_adapter_.clear_prompt()
                self._clear_cuda_cache(device)

                y_query_np = y_np[query_idx]
                loss_value = log_loss(y_query_np, query_proba)
                query_pred = np.argmax(query_proba, axis=1)
                query_accuracy = accuracy(y_query_np, query_pred)
                epoch_losses.append(loss_value)
                epoch_y_parts.append(y_query_np)
                epoch_proba_parts.append(query_proba)
                self.history_.append(
                    TrainingRecord(
                        epoch=epoch_idx + 1,
                        batch=batch_idx,
                        query_loss=loss_value,
                        query_accuracy=query_accuracy,
                        batch_size=int(len(batch_indices)),
                        support_size=int(len(support_idx)),
                        query_size=int(len(query_idx)),
                    )
                )

            mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            train_metrics = self._aggregate_epoch_metrics(
                epoch_y_parts,
                epoch_proba_parts,
                mean_loss=mean_epoch_loss,
            )
            if (
                not use_validation_for_best
                and np.isfinite(mean_epoch_loss)
                and mean_epoch_loss < best_loss
            ):
                best_loss = mean_epoch_loss
                self.best_epoch_ = epoch_idx + 1
                self.best_query_loss_ = mean_epoch_loss
                best_state = self._encoder_state_cpu()

            print(
                f"epoch {epoch_idx + 1}/{self.encoder.epochs}: "
                f"train_loss={train_metrics['log_loss']:.3f}, "
                f"train_accuracy={train_metrics['accuracy']:.3f}, "
                f"train_roc_auc={train_metrics['roc_auc']:.3f}, "
                f"batches={len(epoch_losses)}/{n_batches}"
            )
            val_metrics = None
            val_context_size = None
            val_query_size = None
            if X_eval_std is not None and y_eval_np is not None:
                eval_result = self._evaluate_standardized_context_query(
                    X_eval_std,
                    y_eval_np,
                    metrics=eval_metric_names,
                    random_state=validation_random_state,
                    max_samples=batch_size,
                    use_full_query=True,
                )
                self.latest_evaluation_ = eval_result["metrics"]
                val_metrics = dict(eval_result["metrics"])
                val_context_size = int(eval_result["context_size"])
                val_query_size = int(eval_result["query_size"])
                self.evaluation_history_.append(
                    EvaluationRecord(
                        epoch=epoch_idx + 1,
                        context_size=val_context_size,
                        query_size=val_query_size,
                        metrics=val_metrics,
                    )
                )
                metrics_text = ", ".join(
                    f"val_{key}={value:.3f}" for key, value in val_metrics.items()
                )
                print(
                    f"epoch {epoch_idx + 1}/{self.encoder.epochs} val: "
                    f"context={val_context_size}, query={val_query_size}, {metrics_text}"
                )
                val_roc_auc = val_metrics.get("roc_auc")
                improved = (
                    val_roc_auc is not None
                    and float(val_roc_auc) >= best_val_roc_auc + self.encoder.min_delta
                )
                if improved:
                    best_val_roc_auc = float(val_roc_auc)
                    best_eval_metrics = dict(val_metrics)
                    self.best_epoch_ = epoch_idx + 1
                    self.best_query_loss_ = val_metrics.get("log_loss")
                    best_state = self._encoder_state_cpu()
                    epochs_without_improvement = 0
                elif best_state is not None:
                    epochs_without_improvement += 1
                    self.encoder_model_.load_state_dict(best_state)
                    self.encoder_model_.to(device)
                    print(
                        f"restored best encoder after epoch {epoch_idx + 1} "
                        f"(best_val_roc_auc={best_val_roc_auc:.3f})"
                    )
            self.epoch_metrics_.append(
                EpochMetrics(
                    epoch=epoch_idx + 1,
                    train_loss=train_metrics["log_loss"],
                    train_accuracy=train_metrics["accuracy"],
                    train_roc_auc=train_metrics["roc_auc"],
                    val_loss=None if val_metrics is None else val_metrics.get("log_loss"),
                    val_accuracy=None if val_metrics is None else val_metrics.get("accuracy"),
                    val_roc_auc=None if val_metrics is None else val_metrics.get("roc_auc"),
                    val_p1_mean=None if val_metrics is None else val_metrics.get("p1_mean"),
                    val_p1_std=None if val_metrics is None else val_metrics.get("p1_std"),
                    val_context_size=val_context_size,
                    val_query_size=val_query_size,
                    batches=len(epoch_losses),
                )
            )
            if (
                use_validation_for_best
                and self.encoder.early_stopping_patience > 0
                and epochs_without_improvement >= self.encoder.early_stopping_patience
            ):
                print(
                    "early stopping: "
                    f"no validation AUC improvement for {epochs_without_improvement} epochs"
                )
                break

        if best_state is not None:
            self.encoder_model_.load_state_dict(best_state)
            self.encoder_model_.to(device)
            if use_validation_for_best:
                if best_eval_metrics is not None:
                    self.latest_evaluation_ = best_eval_metrics
                print(
                    f"restored encoder from epoch {self.best_epoch_} "
                    f"with val_roc_auc={best_val_roc_auc:.3f}"
                )
            else:
                print(
                    f"restored encoder from epoch {self.best_epoch_} "
                    f"with query_loss={self.best_query_loss_:.3f}"
                )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Model has not been fitted.")
        if isinstance(self.X_train_, EventGraphDataset):
            if not isinstance(X, EventGraphDataset):
                raise TypeError("This model was trained on graph inputs.")
            X_std = self.graph_standardizer_.transform(X)
        else:
            X_np = to_numpy_matrix(X)
            X_std = self.standardizer_.transform(X_np)
        return self._predict_proba_standardized(X_std)

    def predict(self, X: Any) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def get_training_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "encoder": asdict(self.encoder),
            "history": [asdict(record) for record in self.history_],
            "evaluation_history": [
                asdict(record) for record in self.evaluation_history_
            ],
            "epoch_metrics": [asdict(record) for record in self.epoch_metrics_],
        }
        if self.best_epoch_ is not None:
            summary["best_epoch"] = int(self.best_epoch_)
        if self.best_query_loss_ is not None:
            summary["best_query_loss"] = float(self.best_query_loss_)
        if self.latest_evaluation_ is not None:
            summary["latest_evaluation"] = self.latest_evaluation_
        return summary

    def evaluate_context_query(
        self,
        X: Any,
        y: Any,
        metrics: list[str],
        *,
        random_state: int | None = None,
    ) -> dict[str, Any]:
        X_std = self.standardizer_.transform(to_numpy_matrix(X))
        y_np = to_label_vector(y)
        return self._evaluate_standardized_context_query(
            X_std,
            y_np,
            metrics=metrics,
            random_state=self.random_state if random_state is None else random_state,
        )

    def _episode_step(
        self,
        x_support: Any,
        y_support: Any,
        x_query: Any,
        y_query: Any,
    ) -> tuple[Any, np.ndarray]:
        torch_mod, _ = require_torch()
        import torch.nn.functional as F

        if self.encoder_model_ is None or self.tabpfn_adapter_ is None:
            raise RuntimeError("Model has not been initialized.")

        z_support = self.encoder_model_(x_support).contiguous()
        z_query = self.encoder_model_(x_query).contiguous()
        self.tabpfn_adapter_.fit_prompt(z_support.detach(), y_support)
        logits = self.tabpfn_adapter_.forward_logits(z_query)
        if logits.ndim != 2:
            raise RuntimeError(
                f"Expected query logits with shape (n, c), got {tuple(logits.shape)}."
            )
        task_loss = F.cross_entropy(logits.float(), y_query)
        identity_loss = self._identity_regularization(
            z_support=z_support,
            x_support=x_support,
            z_query=z_query,
            x_query=x_query,
            weight=self.encoder.identity_weight,
        )
        loss = task_loss + identity_loss
        proba = torch_mod.softmax(logits.detach().float(), dim=1).cpu().numpy()
        return loss, proba

    @staticmethod
    def _model_input(X: Any, indices: np.ndarray, device: str) -> Any:
        torch_mod, _ = require_torch()
        idx = np.asarray(indices, dtype=np.int64)
        if isinstance(X, EventGraphDataset):
            return X.to_batch(idx, device=device)
        return torch_mod.tensor(X[idx], dtype=torch_mod.float32, device=device)

    @staticmethod
    def _subset_model_input(X: Any, indices: np.ndarray) -> Any:
        idx = np.asarray(indices, dtype=np.int64)
        if isinstance(X, EventGraphDataset):
            return X.subset(idx)
        return X[idx]

    @staticmethod
    def _identity_regularization(
        *,
        z_support: Any,
        x_support: Any,
        z_query: Any,
        x_query: Any,
        weight: float,
    ) -> Any:
        if weight <= 0.0:
            return z_query.new_tensor(0.0)
        x_support_shape = getattr(x_support, "shape", None)
        x_query_shape = getattr(x_query, "shape", None)
        if z_support.shape != x_support_shape or z_query.shape != x_query_shape:
            return z_query.new_tensor(0.0)
        support_penalty = (z_support - x_support).pow(2).mean()
        query_penalty = (z_query - x_query).pow(2).mean()
        return weight * (support_penalty + query_penalty)

    def _predict_proba_standardized(self, X_std: np.ndarray) -> np.ndarray:
        torch_mod, _ = require_torch()
        if self.encoder_model_ is None:
            raise RuntimeError("Model has not been initialized.")
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Training data is missing.")

        device = self._effective_device()
        if self.tabpfn_adapter_ is None:
            self.tabpfn_adapter_ = TabPFNPromptAdapter(device=device).build()
        support_idx = self._inference_support_indices()
        support_x = self._model_input(self.X_train_, support_idx, device)
        support_y = torch_mod.tensor(
            self.y_train_[support_idx],
            dtype=torch_mod.long,
            device=device,
        )
        query_x = self._model_input(X_std, np.arange(len(X_std)), device)

        self.encoder_model_.eval()
        with torch_mod.no_grad():
            z_support = self.encoder_model_(support_x).contiguous()
            z_query = self.encoder_model_(query_x).contiguous()
            self.tabpfn_adapter_.fit_prompt(z_support, support_y)
            proba = np.asarray(self.tabpfn_adapter_.predict_proba(z_query))
        self.tabpfn_adapter_.clear_prompt()
        self._clear_cuda_cache(device)
        return proba

    def _evaluate_standardized_context_query(
        self,
        X_std: np.ndarray,
        y: np.ndarray,
        metrics: list[str],
        *,
        random_state: int,
        max_samples: int | None = None,
        use_full_query: bool = False,
    ) -> dict[str, Any]:
        if use_full_query and max_samples is not None and len(y) > max_samples:
            context_size = int(round(max_samples * 0.5))
            context_size = min(max(2, context_size), len(y) - 2)
            context_idx = self._stratified_eval_indices(
                y,
                n_samples=context_size,
                random_state=random_state,
            )
            query_mask = np.ones(len(y), dtype=bool)
            query_mask[context_idx] = False
            query_idx = np.flatnonzero(query_mask).astype(np.int64)
            rng = np.random.default_rng(random_state)
            rng.shuffle(query_idx)
            X_context = self._subset_model_input(X_std, context_idx)
            y_context = y[context_idx]
            X_query = self._subset_model_input(X_std, query_idx)
            y_query = y[query_idx]
            query_chunk_size = max(1, int(max_samples) - context_size)
        elif max_samples is not None and len(y) > max_samples:
            eval_idx = self._stratified_eval_indices(
                y,
                n_samples=max_samples,
                random_state=random_state,
            )
            X_eval = self._subset_model_input(X_std, eval_idx)
            y_eval = y[eval_idx]
            episode = RatioEpisodeSampler(
                support_query_ratio=0.5,
                random_state=random_state,
            ).sample(y_eval)
            X_context = self._subset_model_input(X_eval, episode.support_idx)
            y_context = y_eval[episode.support_idx]
            X_query = self._subset_model_input(X_eval, episode.query_idx)
            y_query = y_eval[episode.query_idx]
            query_chunk_size = None
        else:
            episode = RatioEpisodeSampler(
                support_query_ratio=0.5,
                random_state=random_state,
            ).sample(y)
            X_context = self._subset_model_input(X_std, episode.support_idx)
            y_context = y[episode.support_idx]
            X_query = self._subset_model_input(X_std, episode.query_idx)
            y_query = y[episode.query_idx]
            query_chunk_size = None
        proba = self._predict_proba_with_context(
            X_context_std=X_context,
            y_context=y_context,
            X_query_std=X_query,
            query_chunk_size=query_chunk_size,
        )
        pred = np.argmax(proba, axis=1)
        computed_metrics = self._compute_metrics(y_query, pred, proba, metrics)
        if proba.shape[1] > 1:
            computed_metrics["p1_mean"] = float(np.mean(proba[:, 1]))
            computed_metrics["p1_std"] = float(np.std(proba[:, 1]))
        return {
            "context_size": int(len(y_context)),
            "query_size": int(len(y_query)),
            "metrics": computed_metrics,
        }

    def _predict_proba_with_context(
        self,
        X_context_std: np.ndarray,
        y_context: np.ndarray,
        X_query_std: np.ndarray,
        query_chunk_size: int | None = None,
    ) -> np.ndarray:
        torch_mod, _ = require_torch()
        if self.encoder_model_ is None:
            raise RuntimeError("Model has not been initialized.")

        device = self._effective_device()
        if self.tabpfn_adapter_ is None:
            self.tabpfn_adapter_ = TabPFNPromptAdapter(device=device).build()
        context_x = self._model_input(
            X_context_std,
            np.arange(len(X_context_std)),
            device,
        )
        context_y = torch_mod.tensor(y_context, dtype=torch_mod.long, device=device)
        chunk_size = len(X_query_std) if query_chunk_size is None else int(query_chunk_size)
        chunk_size = max(1, chunk_size)

        self.encoder_model_.eval()
        with torch_mod.no_grad():
            z_context = self.encoder_model_(context_x).contiguous()
            self.tabpfn_adapter_.fit_prompt(z_context, context_y)
            proba_parts: list[np.ndarray] = []
            for start in range(0, len(X_query_std), chunk_size):
                query_idx = np.arange(
                    start,
                    min(start + chunk_size, len(X_query_std)),
                    dtype=np.int64,
                )
                query_x = self._model_input(X_query_std, query_idx, device)
                z_query = self.encoder_model_(query_x).contiguous()
                proba_parts.append(np.asarray(self.tabpfn_adapter_.predict_proba(z_query)))
            proba = np.concatenate(proba_parts, axis=0)
        self.tabpfn_adapter_.clear_prompt()
        self._clear_cuda_cache(device)
        return proba

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        proba: np.ndarray,
        metrics: list[str],
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for metric in metrics:
            if metric == "accuracy":
                out[metric] = accuracy(y_true, y_pred)
            elif metric == "roc_auc":
                out[metric] = roc_auc(y_true, proba)
            elif metric == "log_loss":
                out[metric] = log_loss(y_true, proba)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        return out

    @classmethod
    def _aggregate_epoch_metrics(
        cls,
        y_parts: list[np.ndarray],
        proba_parts: list[np.ndarray],
        *,
        mean_loss: float,
    ) -> dict[str, float]:
        if not y_parts or not proba_parts:
            return {
                "log_loss": float(mean_loss),
                "accuracy": float("nan"),
                "roc_auc": float("nan"),
            }
        y_true = np.concatenate(y_parts, axis=0)
        proba = np.concatenate(proba_parts, axis=0)
        pred = np.argmax(proba, axis=1)
        metrics = cls._compute_metrics(
            y_true,
            pred,
            proba,
            ["accuracy", "roc_auc"],
        )
        metrics["log_loss"] = float(mean_loss)
        return metrics

    @staticmethod
    def _epoch_eval_metrics(metrics: list[str] | None) -> list[str]:
        requested = list(metrics or [])
        out = ["log_loss", "accuracy", "roc_auc"]
        for metric in requested:
            if metric not in out:
                out.append(metric)
        return out

    def _inference_support_indices(self) -> np.ndarray:
        if self.y_train_ is None:
            raise RuntimeError("Training labels are missing.")
        return RatioEpisodeSampler(
            support_query_ratio=0.5,
            random_state=self.random_state + 999,
        ).sample(self.y_train_).support_idx

    @staticmethod
    def _stratified_eval_indices(
        y: np.ndarray,
        *,
        n_samples: int,
        random_state: int,
    ) -> np.ndarray:
        from tabpfn_feature_encoder.data.preprocessing import stratified_sample_indices

        return stratified_sample_indices(
            y,
            n_samples=min(int(n_samples), len(y)),
            random_state=random_state,
        )

    def _effective_device(self) -> str:
        torch_mod, _ = require_torch()
        if str(self.device).startswith("cuda") and not torch_mod.cuda.is_available():
            return "cpu"
        return str(self.device)

    @staticmethod
    def _clear_cuda_cache(device: str) -> None:
        torch_mod, _ = require_torch()
        if str(device).startswith("cuda") and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()

    def _seed_everything(self) -> None:
        torch_mod, _ = require_torch()
        np.random.seed(self.random_state)
        torch_mod.manual_seed(self.random_state)
        if torch_mod.cuda.is_available():
            torch_mod.cuda.manual_seed_all(self.random_state)

    def prepare_for_serialization(self) -> "EncoderTabPFNClassifier":
        if self.encoder_model_ is not None:
            self.encoder_model_.to("cpu")
        if self.tabpfn_adapter_ is not None:
            self.tabpfn_adapter_.clear_prompt()
        self.tabpfn_adapter_ = None
        self._clear_cuda_cache("cuda")
        return self

    def _encoder_state_cpu(self) -> dict[str, Any]:
        if self.encoder_model_ is None:
            raise RuntimeError("Encoder has not been built.")
        return {
            name: param.detach().cpu().clone()
            for name, param in self.encoder_model_.state_dict().items()
        }
