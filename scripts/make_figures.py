"""Make every paper figure and table (F1-F6, Elo table, puzzle table) into paper/figures/.

Same code the notebook calls (src/eval/figures.py). Each figure is saved as PNG (300 dpi), PDF and a CSV of
its numbers; F3 also writes F3_tests.csv (H2 paired bootstrap tests, Holm-corrected, and H3 recovery).

Run from the repo root:
    python -m scripts.make_figures --config configs/default.yaml --bench-run bench_test_... \
        [--tour-run tour_1] [--puzzle-run puzzles_1]
"""

from __future__ import annotations

import argparse

from src.eval.figures import make_all
from src.utils import load_config, load_gate_config, resolve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--bench-run", required=True)
    p.add_argument("--tour-run", default=None)
    p.add_argument("--puzzle-run", default=None)
    p.add_argument("--out", default="paper/figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    base = resolve(cfg["results_dir"])
    res = make_all(cfg, load_gate_config(), base / args.bench_run / "benchmark_moves.csv", resolve(args.out),
                   base / args.tour_run / "ratings.csv" if args.tour_run else None,
                   base / args.puzzle_run / "puzzles_results.csv" if args.puzzle_run else None)
    pts, tests = res["F3"]
    print(pts.round(3).to_string(index=False))
    if len(tests):
        print(tests.round(4).to_string(index=False))
    print(f"figures written to {resolve(args.out)}")


if __name__ == "__main__":
    main()
