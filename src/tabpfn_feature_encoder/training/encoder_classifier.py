from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np

from tabpfn_feature_encoder.config import EncoderConfig
from tabpfn_feature_encoder.data.base import to_label_vector, to_numpy_matrix
from tabpfn_feature_encoder.data.graphs import EventGraphDataset, GraphStandardizer
from tabpfn_feature_encoder.data.preprocessing import Standardizer
from tabpfn_feature_encoder.evaluation.metrics import accuracy, log_loss, roc_auc
from tabpfn_feature_encoder.models.factory import build_encoder
from tabpfn_feature_encoder.models.tabpfn_adapter import TabPFNPromptAdapter
from tabpfn_feature_encoder.models.torch_utils import require_torch


@dataclass
class EncoderClassifierEpoch:
    epoch: int
    train_loss: float
    train_accuracy: float
    train_roc_auc: float
    val_loss: float | None
    val_accuracy: float | None
    val_roc_auc: float | None
    train_grad_norm_mean: float | None
    train_grad_norm_max: float | None
    skipped_nonfinite_updates: int
    batches: int


@dataclass
class EncoderOnlyClassifier:
    """Train an encoder by backpropagating a frozen TabPFN support/query loss.

    The historical class name is kept for checkpoint compatibility. There is no
    learned classifier head in the training objective: source episodes are
    encoded, passed to frozen TabPFN as support/query tensors, and the loss is
    backpropagated only into the encoder.
    """

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    device: str = "cpu"
    random_state: int = 42

    encoder_model_: Any = None
    classifier_head_: Any = None
    standardizer_: Standardizer = field(default_factory=Standardizer)
    graph_standardizer_: GraphStandardizer = field(default_factory=GraphStandardizer)
    classes_: np.ndarray | None = None
    history_: list[EncoderClassifierEpoch] = field(default_factory=list)
    best_epoch_: int | None = None
    latest_evaluation_: dict[str, float] | None = None
    is_graph_input_: bool = False
    tabpfn_max_classes_: int = 10
    ecoc_codebook_: np.ndarray | None = None
    prompt_X_model_: Any = None
    prompt_y_encoded_: np.ndarray | None = None

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
    ) -> EncoderOnlyClassifier:
        torch_mod, _ = require_torch()
        self._seed_everything()

        y_np = to_label_vector(y_train)
        self.classes_ = np.unique(y_np)
        y_encoded = self._encode_labels(y_np)
        self.is_graph_input_ = isinstance(X_train, EventGraphDataset)
        if self.is_graph_input_:
            X_model = self.graph_standardizer_.fit_transform(X_train)
            input_dim = X_model.node_dim
            global_dim = X_model.global_dim
        else:
            X_model = self.standardizer_.fit_transform(to_numpy_matrix(X_train))
            input_dim = X_model.shape[1]
            global_dim = 0

        X_val_model = None
        y_val_np = None
        if X_val is not None and y_val is not None:
            y_val_np = to_label_vector(y_val)
            if self.is_graph_input_:
                if not isinstance(X_val, EventGraphDataset):
                    raise TypeError("Graph classifier training requires graph validation input.")
                X_val_model = self.graph_standardizer_.transform(X_val)
            else:
                X_val_model = self.standardizer_.transform(to_numpy_matrix(X_val))

        device = self._effective_device()
        self.encoder_model_ = build_encoder(
            encoder_type=self.encoder.type,
            input_dim=input_dim,
            global_dim=global_dim,
            hidden_dim=self.encoder.hidden_dim,
            output_dim=self.encoder.output_dim,
            layers=self.encoder.layers,
            residual_scale=self.encoder.residual_scale,
            attention_heads=self.encoder.attention_heads,
        ).to(device)
        self.classifier_head_ = None
        trainable_params = self._count_trainable_parameters(self.encoder_model_)

        tabpfn = TabPFNPromptAdapter(device=device, random_state=self.random_state).build()
        self.tabpfn_max_classes_ = self._infer_tabpfn_max_classes(tabpfn)
        self.ecoc_codebook_ = self._make_ecoc_codebook(
            n_classes=len(self.classes_),
            alphabet_size=self.tabpfn_max_classes_,
            redundancy=self.encoder.many_class_redundancy,
            random_state=self.random_state,
        )

        rng = np.random.default_rng(self.random_state)
        batch_size = max(1, int(self.encoder.batch_size))
        prompt_idx = self._sample_prompt_indices(y_encoded, rng, batch_size)
        self.prompt_X_model_ = self._subset_model_input(X_model, prompt_idx)
        self.prompt_y_encoded_ = y_encoded[prompt_idx].copy()

        optimizer = torch_mod.optim.Adam(self.encoder_model_.parameters(), lr=self.encoder.learning_rate)
        best_state = None
        best_val_roc_auc = float("-inf")
        best_train_loss = float("inf")
        epochs_without_improvement = 0
        self.history_ = []
        self.best_epoch_ = None
        self.latest_evaluation_ = None

        task_count = self._task_count()
        task_text = (
            f"ecoc_tasks={task_count}, alphabet_size={self.tabpfn_max_classes_}"
            if self.ecoc_codebook_ is not None
            else f"classes={len(self.classes_)}"
        )
        print(
            "Encoder+TabPFN settings: "
            f"type={self.encoder.type}, device={device}, layers={self.encoder.layers}, "
            f"hidden_dim={self.encoder.hidden_dim}, output_dim={self.encoder.output_dim}, "
            f"trainable_encoder_params={trainable_params}, "
            f"batch_size={batch_size}, support_query_ratio={self.encoder.support_query_ratio}, "
            f"learning_rate={self.encoder.learning_rate}, grad_clip_norm={self.encoder.grad_clip_norm}, "
            f"validation_episodes={self.encoder.validation_episodes}, "
            f"detach_support_gradients={self.encoder.detach_support_gradients}, "
            f"{task_text}"
        )

        n_batches = max(1, int(np.ceil(len(y_np) / batch_size)))
        for epoch_idx in range(max(1, self.encoder.epochs)):
            losses: list[float] = []
            grad_norms: list[float] = []
            y_parts: list[np.ndarray] = []
            proba_parts: list[np.ndarray] = []
            skipped_nonfinite_updates = 0

            self.encoder_model_.train()
            for batch_idx in range(n_batches):
                task_idx = (epoch_idx * n_batches + batch_idx) % task_count
                task_y = self._task_labels(y_encoded, task_idx)
                support_idx, query_idx = self._sample_episode_indices(
                    labels=y_encoded,
                    rng=rng,
                    batch_size=batch_size,
                    support_query_ratio=self.encoder.support_query_ratio,
                )
                y_support_np = task_y[support_idx]
                y_query_np = task_y[query_idx]
                y_support = torch_mod.tensor(y_support_np, dtype=torch_mod.long, device=device)
                y_query = torch_mod.tensor(y_query_np, dtype=torch_mod.long, device=device)

                optimizer.zero_grad(set_to_none=True)
                tabpfn.clear_prompt()
                z_support = self.encoder_model_(self._model_input(X_model, support_idx, device))
                z_query = self.encoder_model_(self._model_input(X_model, query_idx, device))
                z_support_prompt = (
                    z_support.detach() if self.encoder.detach_support_gradients else z_support
                )
                tabpfn.fit_prompt(z_support_prompt, y_support)
                logits = tabpfn.forward_logits(z_query).float()
                loss = torch_mod.nn.functional.cross_entropy(logits, y_query)
                loss.backward()
                grad_norm = self._grad_norm(self.encoder_model_)
                if np.isfinite(grad_norm):
                    grad_norms.append(grad_norm)
                    if self.encoder.grad_clip_norm > 0.0:
                        torch_mod.nn.utils.clip_grad_norm_(
                            self.encoder_model_.parameters(),
                            max_norm=self.encoder.grad_clip_norm,
                            error_if_nonfinite=True,
                        )
                    optimizer.step()
                else:
                    skipped_nonfinite_updates += 1
                    optimizer.zero_grad(set_to_none=True)
                    tabpfn.clear_prompt()
                    if skipped_nonfinite_updates <= 3:
                        print(
                            "encoder_tabpfn warning: skipped optimizer step due to "
                            f"non-finite gradient norm at epoch={epoch_idx + 1}, "
                            f"batch={batch_idx + 1}/{n_batches}",
                            flush=True,
                        )
                    elif skipped_nonfinite_updates == 4:
                        print(
                            "encoder_tabpfn warning: suppressing further non-finite "
                            "gradient skip messages for this epoch.",
                            flush=True,
                        )

                proba = torch_mod.softmax(logits.detach(), dim=1).cpu().numpy()
                losses.append(float(loss.detach().cpu().item()))
                y_parts.append(y_query_np)
                proba_parts.append(proba)

            train_metrics = self._aggregate_metrics(
                y_parts,
                proba_parts,
                mean_loss=float(np.mean(losses)) if losses else float("nan"),
            )
            val_metrics = None
            if X_val_model is not None and y_val_np is not None:
                val_rng = np.random.default_rng(
                    self.random_state + 50_000 + epoch_idx * 10_000
                )
                val_metrics = self.evaluate_standardized_episodic(
                    X_val_model,
                    y_val_np,
                    rng=val_rng,
                    episodes=self.encoder.validation_episodes,
                    batch_size=batch_size,
                )
                self.latest_evaluation_ = val_metrics
                val_roc_auc = val_metrics["roc_auc"]
                improved = val_roc_auc >= best_val_roc_auc + self.encoder.min_delta
                if improved:
                    best_val_roc_auc = float(val_roc_auc)
                    best_state = self._state_cpu()
                    self.best_epoch_ = epoch_idx + 1
                    epochs_without_improvement = 0
                elif best_state is not None:
                    epochs_without_improvement += 1
            elif (
                np.isfinite(train_metrics["log_loss"])
                and train_metrics["log_loss"] < best_train_loss
            ):
                best_train_loss = train_metrics["log_loss"]
                best_state = self._state_cpu()
                self.best_epoch_ = epoch_idx + 1

            self.history_.append(
                EncoderClassifierEpoch(
                    epoch=epoch_idx + 1,
                    train_loss=train_metrics["log_loss"],
                    train_accuracy=train_metrics["accuracy"],
                    train_roc_auc=train_metrics["roc_auc"],
                    val_loss=None if val_metrics is None else val_metrics["log_loss"],
                    val_accuracy=None if val_metrics is None else val_metrics["accuracy"],
                    val_roc_auc=None if val_metrics is None else val_metrics["roc_auc"],
                    train_grad_norm_mean=(
                        float(np.mean(grad_norms)) if grad_norms else None
                    ),
                    train_grad_norm_max=(
                        float(np.max(grad_norms)) if grad_norms else None
                    ),
                    skipped_nonfinite_updates=skipped_nonfinite_updates,
                    batches=len(losses),
                )
            )
            val_text = ""
            if val_metrics is not None:
                val_text = (
                    f", val_loss={val_metrics['log_loss']:.3f}, "
                    f"val_accuracy={val_metrics['accuracy']:.3f}, "
                    f"val_roc_auc={val_metrics['roc_auc']:.3f}"
                )
            print(
                f"encoder_tabpfn epoch {epoch_idx + 1}/{self.encoder.epochs}: "
                f"train_loss={train_metrics['log_loss']:.3f}, "
                f"train_accuracy={train_metrics['accuracy']:.3f}, "
                f"train_roc_auc={train_metrics['roc_auc']:.3f}, "
                f"grad_norm_mean={np.mean(grad_norms) if grad_norms else float('nan'):.3g}, "
                f"grad_norm_max={np.max(grad_norms) if grad_norms else float('nan'):.3g}, "
                f"skipped_nonfinite_updates={skipped_nonfinite_updates}, "
                f"batches={len(losses)}/{n_batches}{val_text}"
            )

            if (
                X_val_model is not None
                and self.encoder.early_stopping_patience > 0
                and epochs_without_improvement >= self.encoder.early_stopping_patience
            ):
                print(
                    "encoder_tabpfn early stopping: "
                    f"no validation AUC improvement for {epochs_without_improvement} epochs"
                )
                break

        if best_state is not None:
            self._load_state(best_state, device)
        tabpfn.clear_prompt()
        self._clear_cuda_cache(device)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.is_graph_input_:
            if not isinstance(X, EventGraphDataset):
                raise TypeError("This classifier was trained on graph inputs.")
            X_model = self.graph_standardizer_.transform(X)
        else:
            X_model = self.standardizer_.transform(to_numpy_matrix(X))
        return self.predict_proba_standardized(X_model)

    def encode(self, X: Any, batch_size: int | None = None) -> np.ndarray:
        if self.is_graph_input_:
            if not isinstance(X, EventGraphDataset):
                raise TypeError("This classifier was trained on graph inputs.")
            X_model = self.graph_standardizer_.transform(X)
        else:
            X_model = self.standardizer_.transform(to_numpy_matrix(X))
        return self.encode_standardized(X_model, batch_size=batch_size)

    def encode_standardized(self, X_model: Any, batch_size: int | None = None) -> np.ndarray:
        torch_mod, _ = require_torch()
        if self.encoder_model_ is None:
            raise RuntimeError("Classifier has not been fitted.")

        device = self._effective_device()
        self.encoder_model_.eval()
        parts: list[np.ndarray] = []
        effective_batch_size = max(1, int(batch_size or self.encoder.batch_size))
        with torch_mod.no_grad():
            for start in range(0, len(X_model), effective_batch_size):
                idx = np.arange(
                    start,
                    min(start + effective_batch_size, len(X_model)),
                    dtype=np.int64,
                )
                x_batch = self._model_input(X_model, idx, device)
                parts.append(self.encoder_model_(x_batch).detach().cpu().numpy())
        self._clear_cuda_cache(device)
        if not parts:
            return np.zeros((0, int(self.encoder.output_dim)), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32)

    def evaluate(self, X: Any, y: Any) -> dict[str, float]:
        proba = self.predict_proba(X)
        return self.metrics_from_proba(y, proba)

    def metrics_from_proba(self, y: Any, proba: np.ndarray) -> dict[str, float]:
        y_np = self._encode_labels(to_label_vector(y))
        return self._classification_metrics(y_np, proba)

    def get_training_summary(self) -> dict[str, Any]:
        return {
            "encoder": asdict(self.encoder),
            "classes": None if self.classes_ is None else self.classes_.tolist(),
            "best_epoch": self.best_epoch_,
            "latest_evaluation": self.latest_evaluation_,
            "history": [asdict(record) for record in self.history_],
            "training_target": "frozen_tabpfn_support_query_loss",
            "tabpfn_used_for_training": True,
            "tabpfn_trainable": False,
            "tabpfn_max_classes": self.tabpfn_max_classes_,
            "trainable_encoder_params": self._count_trainable_parameters(self.encoder_model_),
            "ecoc_codebook": None if self.ecoc_codebook_ is None else self.ecoc_codebook_.tolist(),
        }

    def prepare_for_serialization(self) -> EncoderOnlyClassifier:
        if self.encoder_model_ is not None:
            self.encoder_model_.to("cpu")
        if self.classifier_head_ is not None:
            self.classifier_head_.to("cpu")
        self.prompt_X_model_ = None
        self.prompt_y_encoded_ = None
        self._clear_cuda_cache("cuda")
        return self

    def evaluate_standardized(self, X_model: Any, y: Any) -> dict[str, float]:
        y_np = self._encode_labels(to_label_vector(y))
        proba = self.predict_proba_standardized(X_model)
        return self._classification_metrics(y_np, proba)

    def evaluate_standardized_episodic(
        self,
        X_model: Any,
        y: Any,
        *,
        rng: np.random.Generator | None = None,
        episodes: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        if self.encoder_model_ is None:
            raise RuntimeError("Classifier has not been fitted.")
        y_encoded = self._encode_labels(to_label_vector(y))
        rng = rng or np.random.default_rng(self.random_state + 50_000)
        episode_count = max(1, int(episodes or self.encoder.validation_episodes))
        effective_batch_size = max(1, int(batch_size or self.encoder.batch_size))
        class_orders, cursors = self._make_rotation_state(y_encoded, rng)

        y_parts: list[np.ndarray] = []
        proba_parts: list[np.ndarray] = []
        for _ in range(episode_count):
            support_idx, query_idx = self._rotating_episode_indices(
                labels=y_encoded,
                rng=rng,
                batch_size=effective_batch_size,
                support_query_ratio=self.encoder.support_query_ratio,
                class_orders=class_orders,
                cursors=cursors,
            )
            encoded_support = self.encode_standardized(
                self._subset_model_input(X_model, support_idx),
                batch_size=effective_batch_size,
            )
            encoded_query = self.encode_standardized(
                self._subset_model_input(X_model, query_idx),
                batch_size=effective_batch_size,
            )
            proba = self._tabpfn_predict_encoded(
                encoded_context=encoded_support,
                y_context_encoded=y_encoded[support_idx],
                encoded_query=encoded_query,
            )
            y_parts.append(y_encoded[query_idx])
            proba_parts.append(proba)

        return self._classification_metrics(
            np.concatenate(y_parts, axis=0),
            np.concatenate(proba_parts, axis=0),
        )

    def predict_proba_standardized(self, X_model: Any) -> np.ndarray:
        if self.encoder_model_ is None:
            raise RuntimeError("Classifier has not been fitted.")
        if self.prompt_X_model_ is None or self.prompt_y_encoded_ is None:
            raise RuntimeError("TabPFN prompt context is not available for predict_proba.")

        encoded_context = self.encode_standardized(
            self.prompt_X_model_,
            batch_size=self.encoder.batch_size,
        )
        encoded_query = self.encode_standardized(X_model, batch_size=self.encoder.batch_size)
        return self._tabpfn_predict_encoded(
            encoded_context=encoded_context,
            y_context_encoded=self.prompt_y_encoded_,
            encoded_query=encoded_query,
        )

    def _tabpfn_predict_encoded(
        self,
        *,
        encoded_context: np.ndarray,
        y_context_encoded: np.ndarray,
        encoded_query: np.ndarray,
    ) -> np.ndarray:
        torch_mod, _ = require_torch()
        device = self._effective_device()
        tabpfn = TabPFNPromptAdapter(device=device, random_state=self.random_state).build()
        if self.ecoc_codebook_ is None:
            return self._tabpfn_predict_task(
                tabpfn=tabpfn,
                encoded_context=encoded_context,
                y_context_task=y_context_encoded,
                encoded_query=encoded_query,
                n_task_classes=0 if self.classes_ is None else len(self.classes_),
            )

        class_scores = np.zeros((len(encoded_query), len(self.classes_)), dtype=np.float64)
        for task_idx in range(self.ecoc_codebook_.shape[1]):
            y_context_task = self._task_labels(y_context_encoded, task_idx)
            task_proba = self._tabpfn_predict_task(
                tabpfn=tabpfn,
                encoded_context=encoded_context,
                y_context_task=y_context_task,
                encoded_query=encoded_query,
                n_task_classes=self.tabpfn_max_classes_,
            )
            task_log_proba = np.log(np.clip(task_proba, 1e-15, 1.0))
            class_scores += task_log_proba[:, self.ecoc_codebook_[:, task_idx]]
        return self._softmax_np(class_scores)

    def _tabpfn_predict_task(
        self,
        *,
        tabpfn: TabPFNPromptAdapter,
        encoded_context: np.ndarray,
        y_context_task: np.ndarray,
        encoded_query: np.ndarray,
        n_task_classes: int,
    ) -> np.ndarray:
        torch_mod, _ = require_torch()
        device = self._effective_device()
        context_x = torch_mod.tensor(encoded_context, dtype=torch_mod.float32, device=device)
        context_y = torch_mod.tensor(y_context_task, dtype=torch_mod.long, device=device)
        tabpfn.clear_prompt()
        tabpfn.fit_prompt(context_x, context_y)
        parts: list[np.ndarray] = []
        batch_size = max(1, int(self.encoder.batch_size))
        with torch_mod.no_grad():
            for start in range(0, len(encoded_query), batch_size):
                query_x = torch_mod.tensor(
                    encoded_query[start : start + batch_size],
                    dtype=torch_mod.float32,
                    device=device,
                )
                logits = tabpfn.forward_logits(query_x).float()
                proba = torch_mod.softmax(logits, dim=1).detach().cpu().numpy()
                parts.append(self._pad_task_proba(proba, n_task_classes))
        tabpfn.clear_prompt()
        self._clear_cuda_cache(device)
        return np.concatenate(parts, axis=0)

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Classifier classes are not initialized.")
        labels = np.asarray(y, dtype=np.int64)
        encoded = np.searchsorted(self.classes_, labels)
        in_bounds = encoded < len(self.classes_)
        valid = np.zeros(len(labels), dtype=bool)
        valid[in_bounds] = self.classes_[encoded[in_bounds]] == labels[in_bounds]
        if not np.all(valid):
            raise ValueError("Labels contain values not seen during classifier training.")
        return encoded.astype(np.int64)

    def _task_count(self) -> int:
        return 1 if self.ecoc_codebook_ is None else int(self.ecoc_codebook_.shape[1])

    def _task_labels(self, y_encoded: np.ndarray, task_idx: int) -> np.ndarray:
        if self.ecoc_codebook_ is None:
            return np.asarray(y_encoded, dtype=np.int64)
        return self.ecoc_codebook_[np.asarray(y_encoded, dtype=np.int64), int(task_idx)]

    def _sample_prompt_indices(
        self,
        y_encoded: np.ndarray,
        rng: np.random.Generator,
        batch_size: int,
    ) -> np.ndarray:
        n_classes = len(np.unique(y_encoded))
        support_size = max(n_classes, int(round(batch_size * self.encoder.support_query_ratio)))
        per_class = max(1, support_size // n_classes)
        return self._balanced_indices(y_encoded, rng, per_class=per_class)

    @classmethod
    def _sample_episode_indices(
        cls,
        *,
        labels: np.ndarray,
        rng: np.random.Generator,
        batch_size: int,
        support_query_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        support_total = max(n_classes, int(round(batch_size * support_query_ratio)))
        query_total = max(n_classes, batch_size - support_total)
        support_per_class = max(1, support_total // n_classes)
        query_per_class = max(1, query_total // n_classes)
        support_parts: list[np.ndarray] = []
        query_parts: list[np.ndarray] = []
        for label in unique_labels:
            class_idx = np.flatnonzero(labels == label)
            if len(class_idx) >= support_per_class + query_per_class:
                selected = rng.choice(
                    class_idx,
                    size=support_per_class + query_per_class,
                    replace=False,
                )
                support_parts.append(selected[:support_per_class])
                query_parts.append(selected[support_per_class:])
            else:
                support_parts.append(
                    rng.choice(class_idx, size=support_per_class, replace=len(class_idx) < support_per_class)
                )
                query_parts.append(
                    rng.choice(class_idx, size=query_per_class, replace=len(class_idx) < query_per_class)
                )
        support_idx = np.concatenate(support_parts).astype(np.int64)
        query_idx = np.concatenate(query_parts).astype(np.int64)
        rng.shuffle(support_idx)
        rng.shuffle(query_idx)
        return support_idx, query_idx

    @staticmethod
    def _make_rotation_state(
        labels: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[dict[int, np.ndarray], dict[int, int]]:
        class_orders = {
            int(label): rng.permutation(np.flatnonzero(labels == label)).astype(np.int64)
            for label in np.unique(labels)
        }
        cursors = {int(label): 0 for label in class_orders}
        return class_orders, cursors

    @classmethod
    def _rotating_episode_indices(
        cls,
        *,
        labels: np.ndarray,
        rng: np.random.Generator,
        batch_size: int,
        support_query_ratio: float,
        class_orders: dict[int, np.ndarray],
        cursors: dict[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        support_total = max(n_classes, int(round(batch_size * support_query_ratio)))
        query_total = max(n_classes, batch_size - support_total)
        support_per_class = max(1, support_total // n_classes)
        query_per_class = max(1, query_total // n_classes)
        support_parts: list[np.ndarray] = []
        query_parts: list[np.ndarray] = []
        for label_value in unique_labels:
            label = int(label_value)
            selected = cls._take_rotating(
                label=label,
                count=support_per_class + query_per_class,
                rng=rng,
                class_orders=class_orders,
                cursors=cursors,
            )
            support_parts.append(selected[:support_per_class])
            query_parts.append(selected[support_per_class:])
        support_idx = np.concatenate(support_parts).astype(np.int64)
        query_idx = np.concatenate(query_parts).astype(np.int64)
        rng.shuffle(support_idx)
        rng.shuffle(query_idx)
        return support_idx, query_idx

    @staticmethod
    def _take_rotating(
        *,
        label: int,
        count: int,
        rng: np.random.Generator,
        class_orders: dict[int, np.ndarray],
        cursors: dict[int, int],
    ) -> np.ndarray:
        order = class_orders[label]
        if len(order) == 0:
            raise ValueError("Cannot sample from an empty validation class.")
        parts: list[np.ndarray] = []
        remaining = int(count)
        while remaining > 0:
            cursor = int(cursors[label])
            available = len(order) - cursor
            if available <= 0:
                order = rng.permutation(order).astype(np.int64)
                class_orders[label] = order
                cursors[label] = 0
                cursor = 0
                available = len(order)
            take = min(remaining, available)
            parts.append(order[cursor : cursor + take])
            cursors[label] = cursor + take
            remaining -= take
        return np.concatenate(parts).astype(np.int64)

    @staticmethod
    def _balanced_indices(
        labels: np.ndarray,
        rng: np.random.Generator,
        *,
        per_class: int,
    ) -> np.ndarray:
        parts = []
        for label in np.unique(labels):
            class_idx = np.flatnonzero(labels == label)
            parts.append(
                rng.choice(class_idx, size=per_class, replace=len(class_idx) < per_class)
            )
        out = np.concatenate(parts).astype(np.int64)
        rng.shuffle(out)
        return out

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
        return np.asarray(X[idx], dtype=np.float32).copy()

    @classmethod
    def _aggregate_metrics(
        cls,
        y_parts: list[np.ndarray],
        proba_parts: list[np.ndarray],
        *,
        mean_loss: float,
    ) -> dict[str, float]:
        if not y_parts or not proba_parts:
            return {"log_loss": float(mean_loss), "accuracy": float("nan"), "roc_auc": float("nan")}
        y_true = np.concatenate(y_parts, axis=0)
        proba = np.concatenate(proba_parts, axis=0)
        out = cls._classification_metrics(y_true, proba)
        out["log_loss"] = float(mean_loss)
        return out

    @staticmethod
    def _classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
        pred = np.argmax(proba, axis=1)
        return {
            "accuracy": accuracy(y_true, pred),
            "log_loss": log_loss(y_true, proba),
            "roc_auc": roc_auc(y_true, proba),
        }

    def _state_cpu(self) -> dict[str, Any]:
        if self.encoder_model_ is None:
            raise RuntimeError("Classifier has not been fitted.")
        return {
            "encoder": {
                name: param.detach().cpu().clone()
                for name, param in self.encoder_model_.state_dict().items()
            },
        }

    def _load_state(self, state: dict[str, Any], device: str) -> None:
        if self.encoder_model_ is None:
            raise RuntimeError("Classifier has not been fitted.")
        self.encoder_model_.load_state_dict(state["encoder"])
        self.encoder_model_.to(device)

    def _infer_tabpfn_max_classes(self, tabpfn: TabPFNPromptAdapter) -> int:
        model_limit = getattr(tabpfn.model, "max_num_classes_", None)
        configured_limit = int(self.encoder.tabpfn_max_classes)
        if model_limit is None:
            return configured_limit
        return min(configured_limit, int(model_limit))

    @staticmethod
    def _count_trainable_parameters(model: Any) -> int:
        if model is None:
            return 0
        return int(sum(param.numel() for param in model.parameters() if param.requires_grad))

    @staticmethod
    def _grad_norm(model: Any) -> float:
        torch_mod, _ = require_torch()
        if model is None:
            return float("nan")
        total_sq = 0.0
        seen_grad = False
        for param in model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            seen_grad = True
            if not bool(torch_mod.isfinite(grad).all().detach().cpu().item()):
                return float("nan")
            total_sq += float(grad.norm(2).cpu().item()) ** 2
        if not seen_grad:
            return float("nan")
        return float(math.sqrt(total_sq))

    @classmethod
    def _make_ecoc_codebook(
        cls,
        *,
        n_classes: int,
        alphabet_size: int,
        redundancy: int,
        random_state: int,
    ) -> np.ndarray | None:
        if n_classes <= alphabet_size:
            return None
        rng = np.random.default_rng(random_state + 19_991)
        min_estimators = int(np.ceil(np.log(n_classes) / np.log(alphabet_size)))
        n_estimators = max(1, min_estimators * int(redundancy))
        for _ in range(1_000):
            columns = [
                cls._ecoc_column(n_classes, alphabet_size, rng)
                for _ in range(n_estimators)
            ]
            codebook = np.stack(columns, axis=1).astype(np.int64)
            if len(np.unique(codebook, axis=0)) == n_classes:
                return codebook
        raise RuntimeError("Could not build a unique ECOC codebook for the source classes.")

    @staticmethod
    def _ecoc_column(
        n_classes: int,
        alphabet_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if n_classes < alphabet_size:
            return rng.permutation(alphabet_size)[:n_classes].astype(np.int64)
        repeats = int(np.ceil(n_classes / alphabet_size))
        column = np.tile(np.arange(alphabet_size, dtype=np.int64), repeats)[:n_classes]
        rng.shuffle(column)
        return column

    @staticmethod
    def _pad_task_proba(proba: np.ndarray, n_classes: int) -> np.ndarray:
        if proba.shape[1] == n_classes:
            return proba
        if proba.shape[1] > n_classes:
            return proba[:, :n_classes]
        out = np.full((proba.shape[0], n_classes), 1e-15, dtype=np.float64)
        out[:, : proba.shape[1]] = proba
        out /= out.sum(axis=1, keepdims=True)
        return out

    @staticmethod
    def _softmax_np(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

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
