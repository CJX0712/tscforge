"""tscforge CLI: demo / benchmark / ablation / backends.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tscforge import __version__
from tscforge.core.config import TscConfig
from tscforge.core.seed import set_global_seed
from tscforge.data.generators import DATASET_REGISTRY
from tscforge.pipeline.pipeline import TscPipeline, run_benchmark

# quick demo uses a reduced budget so end-to-end stays well under 60s on CPU
_DEMO_KERNELS = 400


def _print_table(header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(header)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths, strict=False))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=False)))


def cmd_demo(args: argparse.Namespace) -> int:
    """Quick end-to-end demo: all datasets, 1 seed, reduced kernels."""
    cfg = TscConfig(n_kernels=_DEMO_KERNELS, seed=args.seed)
    set_global_seed(args.seed)
    header = ["dataset", "prism", "tinyrocket", "1nn_dtw", "rf_hand"]
    rows = []
    pipe = TscPipeline(cfg)
    for ds in DATASET_REGISTRY:
        res = pipe.run(ds, args.seed, collect_failures=False)
        rows.append(
            [
                ds,
                f"{res.accuracies['prism']:.4f}",
                f"{res.accuracies['tinyrocket']:.4f}",
                f"{res.accuracies['1nn_dtw']:.4f}",
                f"{res.accuracies['rf_handcrafted']:.4f}",
            ]
        )
    print(f"TscForge v{__version__} demo (seed={args.seed}, n_kernels={_DEMO_KERNELS})")
    _print_table(header, rows)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Full benchmark: seeds x datasets, gates, ablation, failure cases."""
    cfg = TscConfig.from_env()
    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = args.datasets.split(",") if args.datasets else list(DATASET_REGISTRY)
    report = run_benchmark(
        cfg,
        seeds=seeds,
        datasets=datasets,
        with_ablation=not args.no_ablation,
        collect_failures=not args.no_failures,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    g = report["gates"]["main"]
    header = ["model", "agg_mean", "agg_std"]
    rows = [
        [name, f"{stats['mean']:.4f}", f"{stats['std']:.4f}"]
        for name, stats in sorted(report["aggregate"].items(), key=lambda kv: -kv[1]["mean"])
    ]
    _print_table(header, rows)
    print(
        f"\nmain gate: prism {g['prism_mean']:.4f} vs {g['strongest_baseline']} "
        f"{g['baseline_mean']:.4f} (delta={g['delta']:+.4f}, "
        f"significant={g['significant']}, passed={g['passed']})"
    )
    if report["ablation"].get("gate"):
        print(f"component gate: {report['ablation']['gate']}")
    print(f"benchmark.json written to {out}")
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    """Component ablation only (3 switches), prints variant table."""
    cfg = TscConfig.from_env()
    seeds = [int(s) for s in args.seeds.split(",")]
    report = run_benchmark(cfg, seeds=seeds, with_ablation=True, collect_failures=False)
    rows = [
        [vname, f"{info['aggregate_mean']:.4f}"]
        for vname, info in report["ablation"]["variants"].items()
    ]
    _print_table(["variant", "agg_mean"], rows)
    print(f"gate: {report['ablation']['gate']}")
    return 0


def cmd_backends(_args: argparse.Namespace) -> int:
    from tscforge.domain.backends.registry import available_backends

    rows = [
        [name, "available" if st.available else "missing", st.version or "-", st.reason or "-"]
        for name, st in available_backends().items()
    ]
    _print_table(["backend", "status", "version", "reason"], rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(prog="tscforge", description="TscForge: deterministic TSC")
    parser.add_argument("--version", action="version", version=f"tscforge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="quick end-to-end demo (<60s)")
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.set_defaults(func=cmd_demo)

    p_bm = sub.add_parser("benchmark", help="full benchmark with gates")
    p_bm.add_argument("--seeds", type=str, default="101,202,303,404,505")
    p_bm.add_argument("--datasets", type=str, default="")
    p_bm.add_argument("--out", type=str, default="results/benchmark.json")
    p_bm.add_argument("--no-ablation", action="store_true")
    p_bm.add_argument("--no-failures", action="store_true")
    p_bm.set_defaults(func=cmd_benchmark)

    p_ab = sub.add_parser("ablation", help="component ablation only")
    p_ab.add_argument("--seeds", type=str, default="101,202,303,404,505")
    p_ab.set_defaults(func=cmd_ablation)

    p_bk = sub.add_parser("backends", help="probe Tier-0 optional backends")
    p_bk.set_defaults(func=cmd_backends)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
