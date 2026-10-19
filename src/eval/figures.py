"""Paper figures F1-F6 and tables (protocol section 2.5). Static matplotlib; PNG (300 dpi) + PDF + CSV table.

Every figure also writes its numbers as a CSV next to it (the "table view"; several palette slots are below 3:1
contrast on white, so values must also be readable without the color). Color roles (validated palette):
    BLUE   = the proposed method (confidence-gated hybrid) / the main series
    ORANGE = the random-gate control
    AQUA, YELLOW = 3rd/4th series in grouped bars only
    GRAY   = baselines and reference lines (always direct-labeled, distinct marker shapes)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval import compose  # noqa: E402
from src.eval.metrics import auroc, ece, holm, reliability_table  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, INK2, GRID, SURFACE = "#8a8984", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SLOTS = (BLUE, ORANGE, AQUA, YELLOW)
PCTS = (10, 25, 50, 75)


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.labelsize": 9, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.5, "lines.markersize": 5, "font.size": 9, "figure.dpi": 100,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, out: Path, name: str, table: pd.DataFrame | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    if table is not None:
        table.to_csv(out / f"{name}.csv", index=False)


def frozen_taus(gate_cfg: dict[str, Any] | None) -> dict[int, float]:
    if not gate_cfg:
        return {}
    return {int(round(float(k) * 100)): float(v) for k, v in gate_cfg["tau"].items()}


# ---------------------------------------------------------------- F1, F2 (H1)
def f1_reliability(w: compose.Wide, out: Path, good_cpl: float = 25) -> pd.DataFrame:
    good = w.df.cpl1 <= good_cpl
    tab = pd.DataFrame(reliability_table(w.df.conf, good, bins=10))
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot([0, 1], [0, 1], ls="--", color=GRAY, lw=1)
    ax.text(0.66, 0.58, "perfect calibration", color=INK2, fontsize=7, rotation=38, ha="center", va="top")
    ax.plot(tab.mean_conf, tab.frac_correct, marker="o", color=BLUE)
    ax.text(0.03, 0.97, f"ECE {ece(w.df.conf, good):.3f}\nAUROC {auroc(w.df.conf, good):.3f}\nn = {len(good)}",
            transform=ax.transAxes, fontsize=7, color=INK2, va="top")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Laya confidence (binned)", ylabel="fraction of good moves (CPL ≤ 25)")
    ax.set_title("F1  Confidence reliability", loc="left")
    _save(fig, out, "F1_reliability", tab)
    return tab


def f2_roc(w: compose.Wide, out: Path, good_cpl: float = 25) -> pd.DataFrame:
    from sklearn.metrics import roc_curve

    good = (w.df.cpl1 <= good_cpl).to_numpy()
    rows = []
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot([0, 1], [0, 1], ls="--", color=GRAY, lw=1)
    ax.text(0.6, 0.52, "random", color=INK2, fontsize=7, rotation=38)
    for label, score, color in (("Laya confidence", w.df.conf.to_numpy(), BLUE),
                                ("1 / #legal moves", 1.0 / w.df.n_legal.to_numpy(), ORANGE)):
        a = auroc(score, good)
        rows.append({"score": label, "auroc": a})
        if np.isnan(a):
            continue
        fpr, tpr, _ = roc_curve(good, score)
        ax.plot(fpr, tpr, color=color, label=f"{label} (AUROC {a:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="false positive rate", ylabel="true positive rate")
    ax.set_title("F2  Does confidence predict a good move?", loc="left")
    ax.legend(loc="lower right")
    tab = pd.DataFrame(rows)
    _save(fig, out, "F2_roc", tab)
    return tab


# ---------------------------------------------------------------- F3 (H2, H3)
def f3_tradeoff(w: compose.Wide, out: Path, gate_cfg: dict[str, Any] | None, mcts_uniform: dict[str, float] | None,
                seeds: int = 200, n_boot: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = w.df
    laya = compose.summarize(compose.apply_mask(w, np.zeros(len(w), bool)))
    s2 = compose.summarize(compose.apply_mask(w, np.ones(len(w), bool)))
    # descriptive sweeps (thresholds from TEST quantiles, only to draw the curve)
    qs = np.linspace(0, 1, 21)
    curve = [compose.summarize(compose.apply_mask(w, compose.hybrid_mask(w, np.nanquantile(d.conf, q) if q > 0 else -np.inf)))
             for q in qs]
    curve_df = pd.DataFrame(curve)
    rnd_curve = pd.DataFrame([compose.random_control(w, q, max(20, seeds // 5)).mean(numeric_only=True).to_dict()
                              for q in qs])
    # pre-registered points (frozen val thresholds)
    taus = frozen_taus(gate_cfg)
    rows = [{"agent": "laya_only", **laya}, {"agent": "always_s2", **s2}]
    if mcts_uniform:
        rows.append({"agent": "mcts_uniform", **mcts_uniform})
    tests = []
    for pct in PCTS:
        if pct not in taus:
            continue
        m = compose.hybrid_mask(w, taus[pct])
        hyb = compose.summarize(compose.apply_mask(w, m))
        rows.append({"agent": f"hybrid_s2_{pct}", **hyb, "recovery_R": compose.recovery(laya["acpl"], hyb["acpl"], s2["acpl"])})
        rate = hyb["s2_rate"]                                   # matched to the ACHIEVED rate
        rc = compose.random_control(w, rate, seeds)
        rows.append({"agent": f"random_matched_{pct}", "acpl": rc.acpl.mean(), "blunder_rate": rc.blunder_rate.mean(),
                     "top1": rc.top1.mean(), "s2_rate": rc.s2_rate.mean(), "cost_ms": rc.cost_ms.mean(), "n": len(w)})
        t = compose.paired_diff_ci(w, m, lambda rng, r=rate: rng.random(len(w)) < r, n=n_boot, seed=seed + pct)
        tests.append({"rate_pct": pct, "achieved_s2_rate": rate, **{f"random_minus_hybrid_acpl_{k}": v for k, v in t.items()}})
    pts = pd.DataFrame(rows)
    tests_df = pd.DataFrame(tests)
    if len(tests_df):
        prereg = tests_df.rate_pct.isin([25, 50])               # H2 is pre-registered at 25% and 50%
        tests_df["p_holm"] = np.nan
        tests_df.loc[prereg, "p_holm"] = holm(tests_df.loc[prereg, "random_minus_hybrid_acpl_p_one_sided"].tolist())

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(rnd_curve.cost_ms, rnd_curve.acpl, color=ORANGE, ls="--", lw=1.5, label="random gate (control)")
    ax.plot(curve_df.cost_ms, curve_df.acpl, color=BLUE, lw=2, label="confidence gate (ours)")
    for _, r in pts.iterrows():
        a = r.agent
        if a.startswith("hybrid_s2_"):
            ax.plot(r.cost_ms, r.acpl, "o", ms=8, color=BLUE, mec=SURFACE, mew=1.5, zorder=4)
            ax.annotate(f"{a.split('_')[-1]}%", (r.cost_ms, r.acpl), xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color=INK2)
        elif a.startswith("random_matched_"):
            ax.plot(r.cost_ms, r.acpl, "D", ms=6, color=ORANGE, mec=SURFACE, mew=1.5, zorder=4)
        else:
            marker = {"laya_only": "o", "always_s2": "s", "mcts_uniform": "^"}[a]
            ax.plot(r.cost_ms, r.acpl, marker, ms=8, color=GRAY, mec=SURFACE, mew=1.5, zorder=5)
            ax.annotate(a, (r.cost_ms, r.acpl), xytext=(6, -10), textcoords="offset points", fontsize=7, color=INK2)
    ax.set(xlabel="mean cost per move (ms, measured)", ylabel="ACPL (centipawns, lower is better)")
    ax.set_title("F3  Move quality vs compute", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2)
    _save(fig, out, "F3_tradeoff", pts)
    tests_df.to_csv(out / "F3_tests.csv", index=False)
    return pts, tests_df


# ---------------------------------------------------------------- F4 (H4)
def f4_blunders(w: compose.Wide, out: Path, gate_cfg: dict[str, Any] | None, blunder_cpl: float = 200,
                seeds: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = w.df
    q = pd.qcut(d.conf.rank(method="first"), 4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"])
    quart = d.assign(quartile=q, blunder=d.cpl1 >= blunder_cpl).groupby("quartile", observed=True) \
        .agg(blunder_rate=("blunder", "mean"), n=("blunder", "size"), mean_conf=("conf", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    bars = ax.bar(quart.quartile.astype(str), quart.blunder_rate, color=BLUE, width=0.7, edgecolor="white", lw=1)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in quart.blunder_rate], fontsize=7, color=INK2, padding=2)
    ax.set(xlabel="Laya confidence quartile", ylabel="laya_only blunder rate (CPL ≥ 200)")
    ax.set_title("F4a  Blunders concentrate where confidence is low", loc="left")
    ax.grid(axis="x", visible=False)
    _save(fig, out, "F4a_blunder_by_confidence", quart)

    taus = frozen_taus(gate_cfg)
    masks = {"laya_only": np.zeros(len(w), bool), "always_s2": np.ones(len(w), bool)}
    if 50 in taus:
        masks["hybrid_s2_50"] = compose.hybrid_mask(w, taus[50])
    phases = ["opening", "middlegame", "endgame"]
    rows = []
    for name, m in masks.items():
        r = compose.apply_mask(w, m)
        for ph in phases:
            sel = r.phase == ph
            rows.append({"agent": name, "phase": ph, "blunder_rate": float(r.blunder[sel].mean()) if sel.any() else np.nan,
                         "n": int(sel.sum())})
    if 50 in taus:
        rate = float(masks["hybrid_s2_50"].mean())
        acc = {ph: [] for ph in phases}
        for s in range(seeds):
            r = compose.apply_mask(w, compose.random_mask(w, rate, s))
            for ph in phases:
                acc[ph].append(float(r.blunder[r.phase == ph].mean()) if (r.phase == ph).any() else np.nan)
        rows += [{"agent": "random_matched_50", "phase": ph, "blunder_rate": float(np.nanmean(acc[ph])),
                  "n": int((d.phase == ph).sum())} for ph in phases]
    tab = pd.DataFrame(rows)
    order = [a for a in ("laya_only", "hybrid_s2_50", "random_matched_50", "always_s2") if a in set(tab.agent)]
    colors = {"laya_only": GRAY, "hybrid_s2_50": BLUE, "random_matched_50": ORANGE, "always_s2": AQUA}
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.8 / len(order)
    x = np.arange(len(phases))
    for i, a in enumerate(order):
        vals = [tab[(tab.agent == a) & (tab.phase == ph)].blunder_rate.iloc[0] for ph in phases]
        ax.bar(x + (i - (len(order) - 1) / 2) * width, vals, width, color=colors[a], label=a, edgecolor="white", lw=1)
    ax.set_xticks(x, phases)
    ax.set(ylabel="blunder rate (CPL ≥ 200)")
    ax.set_title("F4b  Blunder rate by game phase", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7)
    _save(fig, out, "F4b_blunder_by_phase", tab)
    return quart, tab


# ---------------------------------------------------------------- F5 (H5)
def f5_trigger(w: compose.Wide, out: Path, gate_cfg: dict[str, Any] | None,
               puzzles: pd.DataFrame | None = None) -> pd.DataFrame:
    taus = frozen_taus(gate_cfg)
    if 50 not in taus:
        return pd.DataFrame()
    d = w.df.assign(s2=compose.hybrid_mask(w, taus[50]))
    gap_bins = pd.cut(d.sf_gap_cp, [-np.inf, 25, 100, 300, np.inf], labels=["≤25", "25–100", "100–300", ">300"])
    rows = [{"group": "phase", "level": ph, "s2_rate": float(d.s2[d.phase == ph].mean()), "n": int((d.phase == ph).sum())}
            for ph in ("opening", "middlegame", "endgame")]
    rows += [{"group": "forcedness (SF best − 2nd, cp)", "level": str(b), "s2_rate": float(d.s2[gap_bins == b].mean()),
              "n": int((gap_bins == b).sum())} for b in gap_bins.cat.categories]
    rows.append({"group": "source", "level": "game positions (Study A)", "s2_rate": float(d.s2.mean()), "n": len(d)})
    if puzzles is not None and len(puzzles):
        p = puzzles[puzzles.agent == "hybrid_s2_50"]
        if len(p):
            rows.append({"group": "source", "level": "tactical puzzles (Study C)", "s2_rate": float((p.system == 2).mean()),
                         "n": int(len(p))})
    tab = pd.DataFrame(rows)
    groups = list(dict.fromkeys(tab.group))
    fig, axes = plt.subplots(1, len(groups), figsize=(2.6 * len(groups), 2.8), sharey=True)
    for ax, g in zip(np.atleast_1d(axes), groups):
        t = tab[tab.group == g]
        bars = ax.bar(t.level, t.s2_rate, color=BLUE, width=0.7, edgecolor="white", lw=1)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in t.s2_rate], fontsize=7, color=INK2, padding=2)
        ax.set_title(g, fontsize=8, loc="left", color=INK2, fontweight="normal")
        ax.tick_params(axis="x", labelsize=7, rotation=15)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("System 2 trigger rate (hybrid_s2_50)")
    fig.suptitle("F5  Where does the gate send positions to search?", x=0.01, y=1.06, ha="left", fontsize=10,
                 fontweight="bold")
    _save(fig, out, "F5_trigger_rate", tab)
    return tab


# ---------------------------------------------------------------- F6
def f6_depth(w: compose.Wide, out: Path, gate_cfg: dict[str, Any] | None) -> pd.DataFrame:
    taus = frozen_taus(gate_cfg)
    d = w.df
    sel = compose.hybrid_mask(w, taus[50]) if 50 in taus else np.ones(len(d), bool)
    depths = d.max_depth2[sel]
    tab = depths.describe().to_frame("max_depth").join(d.mean_leaf_depth2[sel].describe().to_frame("mean_leaf_depth"))
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    if len(depths):
        bins = np.arange(depths.min(), depths.max() + 2) - 0.5
        ax.hist(depths, bins=bins, color=BLUE, edgecolor="white", lw=1)
        med = float(np.median(depths))
        ax.axvline(med, color=INK2, lw=1, ls="--")
        ax.annotate(f"median {med:.0f}", (med, ax.get_ylim()[1] * 0.92), xytext=(4, 0), textcoords="offset points",
                    fontsize=7, color=INK2)
    ax.set(xlabel="max search depth (plies) when System 2 fires", ylabel="positions")
    ax.grid(axis="x", visible=False)
    ax.set_title("F6  Search depth used by System 2", loc="left")
    _save(fig, out, "F6_depth", tab.reset_index().rename(columns={"index": "stat"}))
    return tab


# ---------------------------------------------------------------- tables for Studies B, C
def table_elo(ratings: pd.DataFrame, out: Path) -> str:
    t = ratings.copy()
    t["Elo (95% CI)"] = [f"{e:.0f}" if a else f"{e:.0f} [{lo:.0f}, {hi:.0f}]" + (" *" if off else "")
                         for e, lo, hi, a, off in zip(t.elo, t.ci_low, t.ci_high, t.anchor, t.off_scale)]
    tex = t[["agent", "Elo (95% CI)", "games", "score"]].to_latex(index=False, escape=True)
    (out / "table_elo.tex").write_text(tex + "% * off-scale: scored >90% or <10% against every anchor\n")
    return tex


def table_puzzles(pz: pd.DataFrame, out: Path, buckets: list[list[int]]) -> pd.DataFrame:
    tab = pz.pivot_table(index="bucket", columns="agent", values="solved", aggfunc="mean")
    tab.index = [f"{buckets[i][0]}–{buckets[i][1]}" for i in tab.index]
    tab.index.name = "puzzle rating"
    tab.to_csv(out / "table_puzzles.csv")
    (out / "table_puzzles.tex").write_text(tab.style.format("{:.1%}").to_latex())
    return tab


def make_all(cfg: dict[str, Any], gate_cfg: dict[str, Any] | None, bench_csv: Path, out: Path,
             ratings_csv: Path | None = None, puzzles_csv: Path | None = None) -> dict[str, Any]:
    set_style()
    bm = pd.read_csv(bench_csv)
    w = compose.Wide.from_benchmark(bm)
    j = cfg["judge"]
    mu = None
    if (bm.agent == "mcts_uniform").any():
        u = bm[bm.agent == "mcts_uniform"]
        mu = {"acpl": float(u.cpl.mean()), "blunder_rate": float((u.cpl >= j["blunder_cpl"]).mean()),
              "top1": float(u.is_best.mean()), "s2_rate": 1.0, "cost_ms": float(u.wall_ms.mean()), "n": len(u)}
    pz = pd.read_csv(puzzles_csv) if puzzles_csv and Path(puzzles_csv).exists() else None
    res: dict[str, Any] = {
        "F1": f1_reliability(w, out, j["good_move_cpl"]),
        "F2": f2_roc(w, out, j["good_move_cpl"]),
        "F3": f3_tradeoff(w, out, gate_cfg, mu, cfg["gate"]["random_control_seeds"], cfg["benchmark"]["bootstrap_resamples"],
                          cfg["seed"]),
        "F4": f4_blunders(w, out, gate_cfg, j["blunder_cpl"], cfg["gate"]["random_control_seeds"]),
        "F5": f5_trigger(w, out, gate_cfg, pz),
        "F6": f6_depth(w, out, gate_cfg),
    }
    if ratings_csv and Path(ratings_csv).exists():
        res["elo"] = table_elo(pd.read_csv(ratings_csv), out)
    if pz is not None:
        res["puzzles"] = table_puzzles(pz, out, cfg["puzzles"]["buckets"])
    return res
