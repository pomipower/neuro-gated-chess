"""v2: build fresh, never-used evaluation sets from a month no v1 split touched (default 2025-03).

Writes into the v2 benchmark_dir (data/benchmark_v2 in configs/v2.yaml):
  positions_test.csv   fresh Study A positions (same sampling rules as v1; excludes every v1 training position)
  openings.csv         fresh tournament openings (ids f00, f01, ...; excludes v1 openings)
  positions_val.csv    copy of the v1 validation positions (development only, already used in v1)
  openings_dev.csv     copy of the v1 test openings (development / practice matches only)
The fresh sets must not be looked at before the frozen v2 runs (docs/V2_PROTOCOL.md).

Run from the repo root:  python -m scripts.build_fresh_sets --config configs/v2.yaml --month 2025-03
"""

from __future__ import annotations

import argparse
import json
import shutil

import numpy as np

from scripts.prepare_data import benchmark_positions, build_openings, discover, write_csv
from src.utils import load_config, resolve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/v2.yaml")
    p.add_argument("--month", default="2025-03")
    p.add_argument("--v1-benchmark", default="data/benchmark")
    p.add_argument("--v1-processed", default="data/processed")
    p.add_argument("--n-positions", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--id-prefix", default="f", help="prefix for position/opening ids (v2: f, v3: g)")
    p.add_argument("--exclude-months", nargs="*", default=[], help="months used by later versions (e.g. v2's 2025-03)")
    p.add_argument("--exclude-benchmarks", nargs="*", default=[], help="other benchmark dirs whose positions/openings to exclude")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    raw, out = resolve(cfg["data"]["raw_dir"]), resolve(cfg["data"]["benchmark_dir"])
    v1b, v1p = resolve(args.v1_benchmark), resolve(args.v1_processed)
    out.mkdir(parents=True, exist_ok=True)
    by_month = discover(raw)
    v1_summary = json.loads((v1b / "split_summary.json").read_text())
    used = set(v1_summary.get("train_months", [])) | {v1_summary.get("val_month"), v1_summary.get("test_month")}
    used |= set(args.exclude_months)
    if args.month in used:
        raise SystemExit(f"{args.month} was used by v1 ({sorted(m for m in used if m)}); pick an unused month")
    if args.month not in by_month:
        raise SystemExit(f"no data/raw file for {args.month}; available: {sorted(by_month)}")
    print(f"fresh month {args.month} (v1 used {sorted(m for m in used if m)})")

    exclude: set[str] = set()
    with np.load(v1p / "train.npz") as z:
        exclude |= {" ".join(str(f).split()[:4]) for f in z["fens"]}
    import pandas as pd

    for d in [v1b, *map(resolve, args.exclude_benchmarks)]:
        for f in ("positions_val.csv", "positions_test.csv"):
            if (d / f).exists():
                exclude |= {" ".join(s.split()[:4]) for s in pd.read_csv(d / f).fen}
    rows = benchmark_positions(by_month[args.month], args.n_positions, cfg, rng, exclude, prefix=args.id_prefix)
    write_csv(out / "positions_test.csv", rows, ["pos_id", "fen", "ply", "phase", "n_legal", "game_id"])
    counts = {ph: sum(r["phase"] == ph for r in rows) for ph in ("opening", "middlegame", "endgame")}
    print(f"  positions_test (fresh): {len(rows)} {counts}")

    seen_open = set(pd.read_csv(v1b / "openings.csv").uci_moves)
    for d in map(resolve, args.exclude_benchmarks):
        if (d / "openings.csv").exists():
            seen_open |= set(pd.read_csv(d / "openings.csv").uci_moves)
    ops = [o for o in build_openings(by_month[args.month], cfg, rng) if o["uci_moves"] not in seen_open]
    for i, o in enumerate(ops):
        o["opening_id"] = f"{args.id_prefix}{i:02d}"
    write_csv(out / "openings.csv", ops, ["opening_id", "uci_moves", "fen", "sf_cp_white"])
    print(f"  openings (fresh): {len(ops)}")

    shutil.copy(v1b / "positions_val.csv", out / "positions_val.csv")
    shutil.copy(v1b / "openings.csv", out / "openings_dev.csv")
    if (v1b / "puzzles.csv").exists():
        shutil.copy(v1b / "puzzles.csv", out / "puzzles.csv")
    (out / "split_summary.json").write_text(json.dumps({
        **{k: v1_summary.get(k) for k in ("train_months", "val_month", "test_month")},
        "fresh_month": args.month, "seed": args.seed, "positions_test": {"n": len(rows), **counts},
        "n_openings": len(ops), "note": "v2 sets; positions_val/openings_dev/puzzles are copies of v1 sets"},
        indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
