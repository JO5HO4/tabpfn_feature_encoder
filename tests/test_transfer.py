import numpy as np
import pytest

from tabpfn_feature_encoder.evaluation.plots import save_encoder_comparison_plots
from tabpfn_feature_encoder.evaluation.transfer import _context_scan_sizes


def test_context_scan_sizes_start_at_per_class_minimum_and_end_at_full_val() -> None:
    y_context = np.repeat([0, 1], 12_500)

    sizes = _context_scan_sizes(
        y_context,
        min_per_class=100,
        n_points=12,
        max_context_size=None,
    )

    assert sizes[0] == 200
    assert sizes[-1] == 25_000
    assert 10 <= len(sizes) <= 12
    assert sizes == sorted(set(sizes))


def test_context_scan_sizes_honor_optional_cap() -> None:
    y_context = np.repeat([0, 1, 2], 1_000)

    sizes = _context_scan_sizes(
        y_context,
        min_per_class=100,
        n_points=16,
        max_context_size=900,
    )

    assert sizes[0] == 300
    assert sizes[-1] == 900


def test_encoder_comparison_plots_save_pdf_outputs(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    csv_dir = tmp_path / "source_residual_mlp" / "source_generalization"
    csv_dir.mkdir(parents=True)
    (csv_dir / "source_12_class_generalization_context_scan_metrics.csv").write_text(
        "\n".join(
            [
                "context_size,status,baseline_accuracy,baseline_roc_auc,"
                "frozen_encoder_accuracy,frozen_encoder_roc_auc",
                "100,ok,0.50,0.55,0.60,0.65",
                "100,ok,0.52,0.57,0.62,0.67",
                "200,ok,0.54,0.59,0.64,0.69",
                "200,ok,0.56,0.61,0.66,0.71",
            ]
        ),
        encoding="utf-8",
    )

    saved = save_encoder_comparison_plots(
        [("MLP encoder", tmp_path / "source_residual_mlp")],
        tmp_path / "plots",
    )

    paths = saved["source_12_class_generalization"]
    assert {path.suffix for path in paths} == {".pdf"}
    assert {path.name for path in paths} == {
        "source_12_class_generalization_roc_auc_comparison.pdf",
        "source_12_class_generalization_accuracy_comparison.pdf",
    }
    assert all(path.exists() for path in paths)
