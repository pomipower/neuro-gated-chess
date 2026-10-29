"""v2 results: Study A on fresh positions (paired vs laya_only) and the v2 tournament summary.

Outputs paper/figures/V2_studyA.csv, V2_studyA_by_phase.csv, V2_tournament.csv and prints them.
Run from the repo root:
    python -m scripts.analyze_v2 --config configs/v2.yaml --bench-run v2_bench --tour-runs v2_tour_a,v2_tour_b
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.utils import load_config, resolve

N_BOOT = 5000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/v2.yaml")
    p.add_argument("--bench-run", default=None)
    p.add_argument("--tour-runs", default=None, help="comma-separated tournament run ids (ratings from the first)")
    p.add_argument("--out", default="paper/figures")
    return p.parse_args()


def paired(a: np.ndarray, b: np.ndarray, seed: int = 0) -> tuple[float, float, float, float]:
    """mean(b - a) with a 95% paired bootstrap CI and two-sided p."""
    d = b - a
    rng = np.random.default_rng(seed)
    stats = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    p = min(1.0, 2 * min((stats <= 0).mean(), (stats >= 0).mean()) + 2 / (N_BOOT + 1))
    return float(d.mean()), float(lo), float(hi), float(p)


def study_a(cfg: dict, run: str, out) -> None:
    bm = pd.read_csv(resolve(cfg["results_dir"]) / run / "benchmark_moves.csv")
    blunder = cfg["judge"]["blunder_cpl"]
    bm["blunder"] = (bm.cpl >= blunder).astype(float)
    bm["is_best"] = bm.is_best.astype(str).str.lower().eq("true").astype(float)
    wide = {a: g.set_index("pos_id") for a, g in bm.groupby("agent")}
    base = wide["laya_only"]
    rows = []
    for a, g in wide.items():
        common = base.index.intersection(g.index)
        r = {"agent": a, "n": len(common), "acpl": g.loc[common, "cpl"].mean(),
             "blunder_rate": g.loc[common, "blunder"].mean(), "top1": g.loc[common, "is_best"].mean(),
             "ms_per_move": g.loc[common, "wall_ms"].mean(),
             "search_rate": (g.loc[common, "system"] == 2).mean(),
             "safety_override_rate": pd.to_numeric(g.loc[common].get("safety_override"), errors="coerce").mean()}
        if a != "laya_only":
            for metric in ("cpl", "blunder"):
                d, lo, hi, p = paired(base.loc[common, metric].to_numpy(float), g.loc[common, metric].to_numpy(float))
                key = "acpl" if metric == "cpl" else "blunder"
                r.update({f"{key}_diff_vs_laya": d, f"{key}_lo": lo, f"{key}_hi": hi, f"{key}_p": p})
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("acpl")
    t.to_csv(out / "V2_studyA.csv", index=False)
    ph = bm.groupby(["agent", "phase"]).agg(acpl=("cpl", "mean"), blunder_rate=("blunder", "mean"), n=("cpl", "size"))
    ph.reset_index().to_csv(out / "V2_studyA_by_phase.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("== v2 Study A (fresh positions; diffs are agent minus laya_only, 95% paired bootstrap CI)\n",
          t.round(3).to_string(index=False))
    print("\n== by phase\n", ph.round(3).to_string())


def tournament(cfg: dict, runs: list[str], out) -> None:
    base = resolve(cfg["results_dir"])
    g = pd.concat([pd.read_csv(base / r / "games.csv") for r in runs], ignore_index=True)
    v = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
    rows = []
    for (w, b), d in g.groupby(["white", "black"]):
        rows += [{"agent": w, "opp": b, "score": v[r]} for r in d.result] + \
                [{"agent": b, "opp": w, "score": 1 - v[r]} for r in d.result]
    s = pd.DataFrame(rows).groupby(["agent", "opp"]).score.agg(["mean", "size", "sum"]).reset_index()
    s.to_csv(out / "V2_tournament.csv", index=False)
    print("\n== v2 tournament scores\n", s.round(3).to_string(index=False))
    print("\nterminations:\n", g.termination.value_counts().to_string())
    ratings = base / runs[0] / "ratings.csv"
    if ratings.exists():
        print("\n== Elo (fit_elo)\n", pd.read_csv(ratings).round(1).to_string(index=False))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.bench_run:
        study_a(cfg, args.bench_run, out)
    if args.tour_runs:
        tournament(cfg, [r.strip() for r in args.tour_runs.split(",")], out)


if __name__ == "__main__":
    main()
