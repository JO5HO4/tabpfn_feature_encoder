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


@dataclass
class EncoderClassifierEpoch:
    epoch: int
    train_loss: float
    train_accuracy: float
    train_roc_auc: float
    val_loss: float | None
    val_accuracy: float | None
    val_roc_auc: float | None
    batches: int


@dataclass
class EncoderOnlyClassifier:
    """A supervised classifier using the same encoder architecture as Encoder+TabPFN."""

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

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
    ) -> EncoderOnlyClassifier:
        torch_mod, nn_mod = require_torch()
        self._seed_everything()

        y_np = to_label_vector(y_train)
        self.classes_ = np.unique(y_np)
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
        self.classifier_head_ = nn_mod.Linear(
            self.encoder.output_dim,
            int(len(self.classes_)),
        ).to(device)

        optimizer = torch_mod.optim.Adam(
            list(self.encoder_model_.parameters()) + list(self.classifier_head_.parameters()),
            lr=self.encoder.learning_rate,
        )
        rng = np.random.default_rng(self.random_state)
        batch_size = max(1, int(self.encoder.batch_size))
        best_state = None
        best_val_roc_auc = float("-inf")
        best_train_loss = float("inf")
        epochs_without_improvement = 0
        self.history_ = []
        self.best_epoch_ = None
        self.latest_evaluation_ = None

        print(
            "Encoder-only classifier settings: "
            f"type={self.encoder.type}, device={device}, layers={self.encoder.layers}, "
            f"hidden_dim={self.encoder.hidden_dim}, output_dim={self.encoder.output_dim}, "
            f"batch_size={batch_size}"
        )

        for epoch_idx in range(max(1, self.encoder.epochs)):
            shuffled_idx = rng.permutation(len(y_np))
            losses: list[float] = []
            y_parts: list[np.ndarray] = []
            proba_parts: list[np.ndarray] = []
            n_batches = int(np.ceil(len(shuffled_idx) / batch_size))

            self.encoder_model_.train()
            self.classifier_head_.train()
            for batch_indices in np.array_split(shuffled_idx, n_batches):
                if len(batch_indices) == 0:
                    continue
                x_batch = self._model_input(X_model, batch_indices, device)
                y_batch = torch_mod.tensor(
                    self._encode_labels(y_np[batch_indices]),
                    dtype=torch_mod.long,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                logits = self.classifier_head_(self.encoder_model_(x_batch))
                loss = torch_mod.nn.functional.cross_entropy(logits.float(), y_batch)
                loss.backward()
                if self.encoder.grad_clip_norm > 0.0:
                    torch_mod.nn.utils.clip_grad_norm_(
                        list(self.encoder_model_.parameters())
                        + list(self.classifier_head_.parameters()),
                        max_norm=self.encoder.grad_clip_norm,
                    )
                optimizer.step()

                proba = torch_mod.softmax(logits.detach().float(), dim=1).cpu().numpy()
                losses.append(float(loss.detach().cpu().item()))
                y_parts.append(self._encode_labels(y_np[batch_indices]))
                proba_parts.append(proba)

            train_metrics = self._aggregate_metrics(
                y_parts,
                proba_parts,
                mean_loss=float(np.mean(losses)) if losses else float("nan"),
            )
            val_metrics = None
            if X_val_model is not None and y_val_np is not None:
                val_metrics = self.evaluate_standardized(X_val_model, y_val_np)
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
                f"encoder_only epoch {epoch_idx + 1}/{self.encoder.epochs}: "
                f"train_loss={train_metrics['log_loss']:.3f}, "
                f"train_accuracy={train_metrics['accuracy']:.3f}, "
                f"train_roc_auc={train_metrics['roc_auc']:.3f}, "
                f"batches={len(losses)}/{n_batches}{val_text}"
            )

            if (
                X_val_model is not None
                and self.encoder.early_stopping_patience > 0
                and epochs_without_improvement >= self.encoder.early_stopping_patience
            ):
                print(
                    "encoder_only early stopping: "
                    f"no validation AUC improvement for {epochs_without_improvement} epochs"
                )
                break

        if best_state is not None:
            self._load_state(best_state, device)
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
            "training_target": "supervised_multiclass_encoder_classifier",
            "tabpfn_used_for_training": False,
        }

    def prepare_for_serialization(self) -> EncoderOnlyClassifier:
        if self.encoder_model_ is not None:
            self.encoder_model_.to("cpu")
        if self.classifier_head_ is not None:
            self.classifier_head_.to("cpu")
        self._clear_cuda_cache("cuda")
        return self

    def evaluate_standardized(self, X_model: Any, y: Any) -> dict[str, float]:
        y_np = self._encode_labels(to_label_vector(y))
        proba = self.predict_proba_standardized(X_model)
        return self._classification_metrics(y_np, proba)

    def predict_proba_standardized(self, X_model: Any) -> np.ndarray:
        torch_mod, _ = require_torch()
        if self.encoder_model_ is None or self.classifier_head_ is None:
            raise RuntimeError("Classifier has not been fitted.")

        device = self._effective_device()
        self.encoder_model_.eval()
        self.classifier_head_.eval()
        parts: list[np.ndarray] = []
        batch_size = max(1, int(self.encoder.batch_size))
        with torch_mod.no_grad():
            for start in range(0, len(X_model), batch_size):
                idx = np.arange(start, min(start + batch_size, len(X_model)), dtype=np.int64)
                x_batch = self._model_input(X_model, idx, device)
                logits = self.classifier_head_(self.encoder_model_(x_batch))
                parts.append(torch_mod.softmax(logits.float(), dim=1).cpu().numpy())
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

    @staticmethod
    def _model_input(X: Any, indices: np.ndarray, device: str) -> Any:
        torch_mod, _ = require_torch()
        idx = np.asarray(indices, dtype=np.int64)
        if isinstance(X, EventGraphDataset):
            return X.to_batch(idx, device=device)
        return torch_mod.tensor(X[idx], dtype=torch_mod.float32, device=device)

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
        if self.encoder_model_ is None or self.classifier_head_ is None:
            raise RuntimeError("Classifier has not been fitted.")
        return {
            "encoder": {
                name: param.detach().cpu().clone()
                for name, param in self.encoder_model_.state_dict().items()
            },
            "classifier_head": {
                name: param.detach().cpu().clone()
                for name, param in self.classifier_head_.state_dict().items()
            },
        }

    def _load_state(self, state: dict[str, Any], device: str) -> None:
        if self.encoder_model_ is None or self.classifier_head_ is None:
            raise RuntimeError("Classifier has not been fitted.")
        self.encoder_model_.load_state_dict(state["encoder"])
        self.classifier_head_.load_state_dict(state["classifier_head"])
        self.encoder_model_.to(device)
        self.classifier_head_.to(device)

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
