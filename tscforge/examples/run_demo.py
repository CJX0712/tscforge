"""End-to-end demo: runs the full benchmark and writes results/benchmark.json.

Full run (5 seeds x 5 datasets + ablation) takes a few minutes on CPU.
For the sub-60s smoke demo use: `tscforge demo` or `python run_demo.py --quick`.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tscforge.cli import main as cli_main  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="TscForge end-to-end demo")
    parser.add_argument("--quick", action="store_true", help="1 seed, no ablation, small kernels")
    parser.add_argument("--seeds", type=str, default="101,202,303,404,505")
    parser.add_argument("--out", type=str, default="results/benchmark.json")
    args = parser.parse_args()
    if args.quick:
        os.environ.setdefault("TSCFORGE_N_KERNELS", "400")
        argv = ["benchmark", "--seeds", "42", "--out", args.out, "--no-ablation"]
    else:
        argv = ["benchmark", "--seeds", args.seeds, "--out", args.out]
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
