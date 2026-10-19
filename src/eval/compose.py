"""Offline composition of hybrids and random-gate controls from Study A rows (protocol section 2.3).

The gate depends only on System 1's output and System 2 does not depend on the gate, so for every position
    hybrid(tau)  : move = laya_only move if conf > tau else always_s2 move
                   cost = laya cost + (search cost if System 2 fired)
    random(rate) : same, with System 2 chosen by a seeded coin flip at `rate`
Costs use the measured component times from benchmark_moves.csv (`always_s2` rows log laya_ms and search_ms
separately). This is an approximation of running the real agent; Study B runs the real agents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

QUALITY_COLS = ("cpl", "is_best")


@dataclass
class Wide:
    """One row per position with System 1 and System 2 outcomes side by side."""

    df: pd.DataFrame

    @classmethod
    def from_benchmark(cls, bm: pd.DataFrame, cfg_judge: dict | None = None) -> "Wide":
        s1 = bm[bm.agent == "laya_only"].set_index("pos_id")
        s2 = bm[bm.agent == "always_s2"].set_index("pos_id")
        common = s1.index.intersection(s2.index)
        s1, s2 = s1.loc[common], s2.loc[common]
        df = pd.DataFrame({
            "phase": s1.phase, "ply": s1.ply, "n_legal": s1.n_legal, "sf_gap_cp": s1.sf_gap_cp,
            "conf": s1.confidence.astype(float),
            "cpl1": s1.cpl.astype(float), "cpl2": s2.cpl.astype(float),
            "best1": s1.is_best.astype(bool), "best2": s2.is_best.astype(bool),
            "same_move": (s1.move_uci == s2.move_uci),
            "laya_ms": s1.wall_ms.astype(float),
            "search_ms": s2.search_ms.astype(float),
            "laya_scored": s1.laya_moves_scored.astype(float),
            "cnn_evals2": s2.cnn_evals.astype(float),
            "sims2": s2.mcts_sims.astype(float),
            "max_depth2": s2.max_depth.astype(float),
            "mean_leaf_depth2": s2.mean_leaf_depth.astype(float),
        })
        return cls(df)

    def __len__(self) -> int:
        return len(self.df)


def apply_mask(w: Wide, s2: np.ndarray, good_cpl: float = 25, blunder_cpl: float = 200) -> pd.DataFrame:
    """Per-position outcome of an agent that uses System 2 exactly where `s2` is True."""
    d = w.df
    cpl = np.where(s2, d.cpl2, d.cpl1)
    return pd.DataFrame({
        "s2": s2, "cpl": cpl, "is_best": np.where(s2, d.best2, d.best1),
        "good": cpl <= good_cpl, "blunder": cpl >= blunder_cpl,
        "cost_ms": d.laya_ms.to_numpy() + np.where(s2, d.search_ms, 0.0),
        "cnn_evals": np.where(s2, d.cnn_evals2, 0.0),
        "phase": d.phase.to_numpy(),
    }, index=d.index)


def hybrid_mask(w: Wide, tau: float) -> np.ndarray:
    c = w.df.conf.to_numpy()
    return np.isnan(c) | (c <= tau)


def random_mask(w: Wide, rate: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(len(w)) < rate


def summarize(rows: pd.DataFrame) -> dict[str, float]:
    return {"acpl": float(rows.cpl.mean()), "blunder_rate": float(rows.blunder.mean()),
            "top1": float(rows.is_best.mean()), "s2_rate": float(rows.s2.mean()),
            "cost_ms": float(rows.cost_ms.mean()), "n": int(len(rows))}


def random_control(w: Wide, rate: float, seeds: int) -> pd.DataFrame:
    """Summary of `seeds` independent random gates at `rate`."""
    return pd.DataFrame([summarize(apply_mask(w, random_mask(w, rate, s))) | {"seed": s} for s in range(seeds)])


def recovery(acpl_laya: float, acpl_hybrid: float, acpl_s2: float) -> float:
    """H3: R = (ACPL_laya - ACPL_hybrid) / (ACPL_laya - ACPL_always_s2); NaN if always_s2 is not better."""
    denom = acpl_laya - acpl_s2
    return float((acpl_laya - acpl_hybrid) / denom) if denom > 0 else float("nan")


def paired_diff_ci(w: Wide, mask_a: np.ndarray, mask_b_fn, n: int = 2000, seed: int = 0,
                   alpha: float = 0.05) -> dict[str, float]:
    """ACPL(B) - ACPL(A) with a paired position bootstrap. `mask_b_fn(rng) -> mask` lets B be random
    (a fresh random gate per resample, so the CI includes the control's own randomness)."""
    rng = np.random.default_rng(seed)
    d = w.df
    cpl_a = np.where(mask_a, d.cpl2, d.cpl1)
    point_b = np.where(mask_b_fn(np.random.default_rng(seed + 1)), d.cpl2, d.cpl1)
    point = float(point_b.mean() - cpl_a.mean())
    stats = np.empty(n)
    m = len(d)
    for i in range(n):
        idx = rng.integers(0, m, m)
        cpl_b = np.where(mask_b_fn(rng), d.cpl2, d.cpl1)
        stats[i] = cpl_b[idx].mean() - cpl_a[idx].mean()
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_one_sided = float((np.sum(stats <= 0) + 1) / (n + 1))       # H: B worse than A (diff > 0)
    return {"diff": point, "lo": float(lo), "hi": float(hi), "p_one_sided": p_one_sided}
