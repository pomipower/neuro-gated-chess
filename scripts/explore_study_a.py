"""EXPLORATORY (not pre-registered) analysis of Study A: where does search actually help, and why the gate failed.

Everything here is computed offline from data/results/<bench_run>/benchmark_moves.csv (CPU only, seconds).
The paper must label these results "exploratory / post-hoc"; they explain the pre-registered results, they do not
replace them, and nothing here was used to change the frozen gate.

Outputs (paper/figures/EXP_*):
    EXP_h1_search_vs_laya.csv  paired bootstrap: always_s2 vs laya_only (ACPL, blunder rate, top-1)
    EXP_gain_by_group.csv      mean search gain (cpl_laya - cpl_search, >0 = search helped) by confidence decile,
                               phase, forcedness and #legal moves, with bootstrap CIs
    EXP_alt_gates.csv          ACPL at matched S2 rates for alternative gates: confidence (frozen), random
                               (expected), inverted confidence, fewest-legal-moves, forcedness (needs Stockfish:
                               not deployable) and the oracle (perfect knowledge of the gain: an upper bound)
    EXP_gain.png/.pdf          figure: gain by confidence decile and by forcedness

Run from the repo root:  python -m scripts.explore_study_a --bench-run bench_test
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.eval import compose
from src.eval.figures import BLUE, ORANGE, set_style
from src.utils import load_config, resolve

RATES = (10, 25, 50, 75)
N_BOOT = 2000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--bench-run", default="bench_test")
    p.add_argument("--out", default="paper/figures")
    return p.parse_args()


def boot_mean(x: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    stats = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def h1(d: pd.DataFrame, good_cpl: float, blunder_cpl: float) -> pd.DataFrame:
    rows = []
    for name, a, b in [("acpl", d.cpl1, d.cpl2),
                       ("blunder_rate", d.cpl1 >= blunder_cpl, d.cpl2 >= blunder_cpl),
                       ("good_rate", d.cpl1 <= good_cpl, d.cpl2 <= good_cpl),
                       ("top1", d.best1, d.best2)]:
        diff = b.to_numpy(float) - a.to_numpy(float)          # always_s2 minus laya_only
        m, lo, hi = boot_mean(diff)
        rng = np.random.default_rng(1)
        stats = diff[rng.integers(0, len(diff), (N_BOOT, len(diff)))].mean(axis=1)
        p_two = float(min(1.0, 2 * min((stats <= 0).mean(), (stats >= 0).mean()) + 2 / (N_BOOT + 1)))
        rows.append({"metric": name, "laya_only": float(a.astype(float).mean()), "always_s2": float(b.astype(float).mean()),
                     "diff_s2_minus_laya": m, "ci_lo": lo, "ci_hi": hi, "p_two_sided": p_two})
    changed = ~d.same_move
    g = d.cpl1 - d.cpl2
    rows.append({"metric": "moves_changed_by_search", "always_s2": float(changed.mean())})
    rows.append({"metric": "of_changed_improved(cpl_drop>10)", "always_s2": float((g[changed] > 10).mean())})
    rows.append({"metric": "of_changed_worsened(cpl_rise>10)", "always_s2": float((g[changed] < -10).mean())})
    return pd.DataFrame(rows)


def gain_by_group(d: pd.DataFrame) -> pd.DataFrame:
    gain = d.cpl1 - d.cpl2
    groups = {
        "confidence decile": pd.qcut(d.conf.rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)]),
        "phase": d.phase,
        "forcedness (SF best-2nd cp)": pd.cut(d.sf_gap_cp, [-np.inf, 25, 100, 300, np.inf],
                                              labels=["<=25", "25-100", "100-300", ">300"]),
        "#legal moves": pd.cut(d.n_legal, [0, 15, 30, 45, 1000], labels=["1-15", "16-30", "31-45", "46+"]),
    }
    rows = []
    for gname, key in groups.items():
        for level, idx in d.groupby(key, observed=True).groups.items():
            m, lo, hi = boot_mean(gain.loc[idx].to_numpy())
            rows.append({"group": gname, "level": str(level), "n": len(idx), "mean_gain_cp": m, "ci_lo": lo, "ci_hi": hi,
                         "mean_conf": float(d.conf.loc[idx].mean()), "laya_acpl": float(d.cpl1.loc[idx].mean())})
    rho, p = spearmanr(d.conf, gain, nan_policy="omit")
    rows.append({"group": "spearman(conf, gain)", "level": "all", "n": len(d), "mean_gain_cp": float(rho), "ci_lo": float(p)})
    return pd.DataFrame(rows)


def top_k_mask(score: np.ndarray, rate: float) -> np.ndarray:
    """Search the `rate` fraction of positions with the highest score (stable tie-break by position order)."""
    k = int(round(rate * len(score)))
    order = np.argsort(-score, kind="stable")
    m = np.zeros(len(score), bool)
    m[order[:k]] = True
    return m


def alt_gates(w: compose.Wide) -> pd.DataFrame:
    d = w.df
    conf = d.conf.fillna(-1).to_numpy()
    gain = (d.cpl1 - d.cpl2).to_numpy()
    scores = {
        "confidence (frozen gate: search when LOW)": -conf,
        "inverted confidence (search when HIGH)": conf,
        "fewest legal moves": -d.n_legal.to_numpy(float),
        "forcedness: SF gap (needs Stockfish; not deployable)": d.sf_gap_cp.to_numpy(float),
        "oracle (knows the gain; upper bound)": gain,
    }
    rows = []
    for pct in RATES:
        r = pct / 100
        exp_random = float(d.cpl1.mean() - r * gain.mean())
        rows.append({"gate": "random (expected)", "rate_pct": pct, "acpl": exp_random, "vs_random": 0.0,
                     "cost_ms": float(d.laya_ms.mean() + r * d.search_ms.mean())})
        for name, s in scores.items():
            rows_ = compose.apply_mask(w, top_k_mask(s, r))
            acpl = float(rows_.cpl.mean())
            rows.append({"gate": name, "rate_pct": pct, "acpl": acpl, "vs_random": acpl - exp_random,
                         "blunder_rate": float(rows_.blunder.mean()), "cost_ms": float(rows_.cost_ms.mean())})
    rows.append({"gate": "laya_only", "rate_pct": 0, "acpl": float(d.cpl1.mean())})
    rows.append({"gate": "always_s2", "rate_pct": 100, "acpl": float(d.cpl2.mean())})
    return pd.DataFrame(rows)


def figure(g: pd.DataFrame, out) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, gname, color, title in [(axes[0], "confidence decile", BLUE, "by Laya confidence (D1 = least confident)"),
                                    (axes[1], "forcedness (SF best-2nd cp)", ORANGE, "by forcedness (Stockfish best − 2nd)")]:
        s = g[g.group == gname]
        x = np.arange(len(s))
        ax.bar(x, s.mean_gain_cp, color=color, width=0.65)
        ax.errorbar(x, s.mean_gain_cp, yerr=[s.mean_gain_cp - s.ci_lo, s.ci_hi - s.mean_gain_cp], fmt="none",
                    ecolor="#444", elinewidth=1, capsize=2)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xticks(x, s.level, fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("search gain (cp, >0 = search helped)")
    fig.suptitle("Exploratory: where does search help? (Study A test set, 95% bootstrap CI)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"EXP_gain.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    bm = pd.read_csv(resolve(cfg["results_dir"]) / args.bench_run / "benchmark_moves.csv")
    w = compose.Wide.from_benchmark(bm)
    d = w.df
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    j = cfg.get("judge", {})
    good, blunder = j.get("good_cpl", 25), j.get("blunder_cpl", 200)

    t1 = h1(d, good, blunder)
    t2 = gain_by_group(d)
    t3 = alt_gates(w)
    t1.to_csv(out / "EXP_h1_search_vs_laya.csv", index=False)
    t2.to_csv(out / "EXP_gain_by_group.csv", index=False)
    t3.to_csv(out / "EXP_alt_gates.csv", index=False)
    figure(t2, out)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("EXPLORATORY ANALYSIS (post hoc, label it so in the paper)\n")
    print("== always_s2 vs laya_only (paired bootstrap)\n", t1.round(4).to_string(index=False))
    print("\n== search gain by group\n", t2.round(3).to_string(index=False))
    print("\n== alternative gates at matched rates (vs_random < 0 = better than random)\n",
          t3.round(2).to_string(index=False))
    print(f"\nwrote EXP_* files to {out}")


if __name__ == "__main__":
    main()
