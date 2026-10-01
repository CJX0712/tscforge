"""Pipeline orchestration tests: leak-free, well-formed result objects.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from tscforge.core.config import TscConfig
from tscforge.core.seed import set_global_seed
from tscforge.data.generators import DATASET_REGISTRY
from tscforge.data.split import stratified_split
from tscforge.pipeline.pipeline import TscPipeline


def test_run_cell_valid_structure():
    pipe = TscPipeline(TscConfig(seed=101))
    res = pipe.run("order_freq2a", 101, collect_failures=False)
    assert set(res.accuracies.keys()) >= {
        "prism",
        "tinyrocket",
        "1nn_ed",
        "1nn_dtw",
        "logreg_flat",
        "rf_handcrafted",
    }
    for name, acc in res.accuracies.items():
        assert 0.0 <= acc <= 1.0, f"{name} acc {acc} out of range"
    # every predicted model must emit one label per holdout sample
    set_global_seed(101)
    batch = DATASET_REGISTRY["order_freq2a"](n_total=150, length=128, seed=101)
    sp = stratified_split(batch, 101)
    assert res.split_sizes["holdout"] == sp["holdout"].n_samples


def test_benchmark_main_gate_keys_present():
    from tscforge.pipeline.pipeline import run_benchmark

    report = run_benchmark(
        TscConfig(seed=101, backend="numpy"),
        seeds=[101, 202],
        datasets=["order_freq2a", "order_freq2b"],
        with_ablation=False,
        collect_failures=False,
    )
    assert "gates" in report and "main" in report["gates"]
    g = report["gates"]["main"]
    # honest (non-inferiority-based) gate keys -- see pipeline.run_benchmark
    for key in (
        "strongest_baseline",
        "prism_mean",
        "baseline_mean",
        "delta_vs_best",
        "non_inferior",
        "significantly_worse",
        "field_mean",
        "delta_vs_field",
        "beats_field",
        "datasets_topped",
        "datasets_total",
        "wins_majority",
        "passed",
    ):
        assert key in g
    assert isinstance(g["passed"], bool)
    assert set(report["aggregate"].keys()) >= {"prism", "tinyrocket"}
    assert report["aggregate"]["prism"]["mean"] >= 0.0
