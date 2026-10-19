"""Anchored Elo for Study B. Definition: docs/EXPERIMENT_PROTOCOL.md section 3.

Bradley-Terry with draws as half-wins: P(i beats j) = 1 / (1 + 10 ** ((R_j - R_i) / 400)). Maximise the
(pseudo-)log-likelihood with scipy.optimize, anchors fixed at their nominal UCI_Elo. A very weak Gaussian
prior (sd 1000 around the anchor mean) keeps ratings finite for agents that win or lose everything; such
agents are flagged off-scale. 95% CI by cluster bootstrap over openings (both colors of an opening stay
together).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
PRIOR_SD = 1000.0


def anchor_elos(names: list[str]) -> dict[str, float]:
    return {n: float(n.split("_")[-1]) for n in names if n.startswith("sf_elo_")}


def fit_ratings(games: pd.DataFrame, anchors: dict[str, float]) -> dict[str, float]:
    g = games[games.result.isin(SCORE)]
    agents = sorted(set(g.white) | set(g.black))
    free = [a for a in agents if a not in anchors]
    if not free:
        return dict(anchors)
    if not anchors:                                   # no anchors: fix the first agent at 0
        anchors = {free[0]: 0.0}
        free = free[1:]
    idx = {a: i for i, a in enumerate(free)}
    w = np.array([idx.get(a, -1) for a in g.white])
    b = np.array([idx.get(a, -1) for a in g.black])
    w_fix = np.array([anchors.get(a, 0.0) for a in g.white])
    b_fix = np.array([anchors.get(a, 0.0) for a in g.black])
    s = g.result.map(SCORE).to_numpy()
    mu = float(np.mean(list(anchors.values())))

    def ratings(x: np.ndarray, which: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        return np.where(which >= 0, x[np.maximum(which, 0)], fixed)

    def nll(x: np.ndarray) -> float:
        d = (ratings(x, w, w_fix) - ratings(x, b, b_fix)) * math.log(10) / 400.0
        ll = s * -np.logaddexp(0, -d) + (1 - s) * -np.logaddexp(0, d)
        return float(-ll.sum() + np.sum((x - mu) ** 2) / (2 * PRIOR_SD**2))

    res = minimize(nll, np.full(len(free), mu), method="L-BFGS-B")
    out = dict(anchors)
    out.update({a: float(res.x[i]) for a, i in idx.items()})
    return out


def score_vs_anchors(games: pd.DataFrame, agent: str, anchors: dict[str, float]) -> list[float]:
    rows = []
    for anc in anchors:
        g1 = games[(games.white == agent) & (games.black == anc)].result.map(SCORE)
        g2 = 1 - games[(games.white == anc) & (games.black == agent)].result.map(SCORE)
        both = pd.concat([g1, g2])
        if len(both):
            rows.append(float(both.mean()))
    return rows


def bootstrap_ratings(games: pd.DataFrame, anchors: dict[str, float], n: int = 1000, seed: int = 0,
                      offscale_high: float = 0.9, offscale_low: float = 0.1) -> pd.DataFrame:
    """Columns: agent, elo, ci_low, ci_high, games, score, off_scale."""
    point = fit_ratings(games, anchors)
    rng = np.random.default_rng(seed)
    openings = games.opening_id.unique()
    groups = {o: games[games.opening_id == o] for o in openings}
    boots: dict[str, list[float]] = {a: [] for a in point}
    for _ in range(n):
        pick = rng.choice(openings, size=len(openings), replace=True)
        sample = pd.concat([groups[o] for o in pick], ignore_index=True)
        r = fit_ratings(sample, anchors)
        for a in point:
            if a in r:
                boots[a].append(r[a])
    rows = []
    for a, e in sorted(point.items(), key=lambda kv: -kv[1]):
        gs = games[(games.white == a) | (games.black == a)]
        sc = pd.concat([gs[gs.white == a].result.map(SCORE), 1 - gs[gs.black == a].result.map(SCORE)])
        vs = score_vs_anchors(games, a, anchors) if a not in anchors else []
        off = bool(vs) and (all(v > offscale_high for v in vs) or all(v < offscale_low for v in vs))
        lo, hi = (np.percentile(boots[a], [2.5, 97.5]) if a not in anchors and boots[a] else (e, e))
        rows.append({"agent": a, "elo": round(e, 1), "ci_low": round(float(lo), 1), "ci_high": round(float(hi), 1),
                     "games": int(len(gs)), "score": round(float(sc.mean()), 3) if len(sc) else math.nan,
                     "anchor": a in anchors, "off_scale": off})
    return pd.DataFrame(rows)


def summarize(cfg: dict[str, Any], games: pd.DataFrame) -> pd.DataFrame:
    names = sorted(set(games.white) | set(games.black))
    t = cfg["tournament"]
    return bootstrap_ratings(games, anchor_elos(names), t["bootstrap_resamples"], cfg["seed"],
                             t["offscale_high"], t["offscale_low"])
