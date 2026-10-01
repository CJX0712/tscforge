"""Full benchmark runner for TscForge gate validation.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import json
import sys

from tscforge.core.config import TscConfig
from tscforge.pipeline.pipeline import run_benchmark

BACKEND = sys.argv[1] if len(sys.argv) > 1 else "numpy"
DATASETS = sys.argv[2].split(",") if len(sys.argv) > 2 else None
if DATASETS is not None:
    DATASETS = [d for d in DATASETS if d and d != "all"]
    if not DATASETS:
        DATASETS = None
ABLATION = sys.argv[3] != "no" if len(sys.argv) > 3 else True
SEEDS = [101, 202, 303, 404, 505]

cfg = TscConfig(seed=SEEDS[0], backend=BACKEND)
print(f"backend={BACKEND} datasets={DATASETS} ablation={ABLATION}", flush=True)
report = run_benchmark(
    cfg,
    seeds=SEEDS,
    datasets=DATASETS,
    with_ablation=ABLATION,
    collect_failures=True,
)
print("\n=== MAIN GATE ===", flush=True)
g = report["gates"]["main"]
for k, v in g.items():
    print(f"  {k}: {v}")
print("\n=== AGGREGATE (per model, mean over seeds/datasets) ===", flush=True)
for name, a in sorted(report["aggregate"].items(), key=lambda kv: -kv[1]["mean"]):
    print(f"  {name:16s} mean={a['mean']:.4f} std={a['std']:.4f}")
print("\n=== ABLATION GATE ===", flush=True)
for k, v in report["ablation"]["gate"].items():
    print(f"  {k}: {v}")
with open("benchmark.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("\nwrote benchmark.json", flush=True)
