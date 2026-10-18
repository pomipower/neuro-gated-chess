"""Metrics for Studies A and C. Definitions: docs/EXPERIMENT_PROTOCOL.md sections 2.1 and 2.5."""

from __future__ import annotations

from typing import Callable, Sequence

import chess
import numpy as np

_PIECE_VALUE = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def non_pawn_material(board: chess.Board) -> int:
    """Combined N/B/R/Q material of both sides (N=B=3, R=5, Q=9)."""
    return sum(_PIECE_VALUE.get(p.piece_type, 0) for p in board.piece_map().values())


def game_phase(board: chess.Board, ply: int, endgame_material_max: int = 20, opening_max_ply: int = 20) -> str:
    """'endgame' if non-pawn material <= endgame_material_max, else 'opening' if ply <= 20, else 'middlegame'."""
    if non_pawn_material(board) <= endgame_material_max:
        return "endgame"
    if ply <= opening_max_ply:
        return "opening"
    return "middlegame"


def clip_score(cp: float, clip: float = 1000.0) -> float:
    return float(max(-clip, min(clip, cp)))


def auroc(confidence: Sequence[float], positive: Sequence[bool]) -> float:
    """AUROC of `confidence` for predicting `positive`. NaN if only one class is present."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(positive, dtype=bool)
    s = np.asarray(confidence, dtype=float)
    ok = ~np.isnan(s)
    if ok.sum() == 0 or y[ok].all() or (~y[ok]).all():
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def ece(confidence: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    c = np.clip(np.asarray(confidence, dtype=float), 0.0, 1.0)
    y = np.asarray(correct, dtype=float)
    idx = np.minimum((c * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - c[m].mean())
    return float(total)


def reliability_table(confidence: Sequence[float], correct: Sequence[bool], bins: int = 10) -> list[dict[str, float]]:
    c = np.clip(np.asarray(confidence, dtype=float), 0.0, 1.0)
    y = np.asarray(correct, dtype=float)
    idx = np.minimum((c * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "lo": b / bins, "hi": (b + 1) / bins, "n": int(m.sum()),
                         "mean_conf": float(c[m].mean()), "frac_correct": float(y[m].mean())})
    return rows


def bootstrap_ci(stat_fn: Callable[[np.ndarray], float], index: np.ndarray, n: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Paired percentile bootstrap over positions: resample `index`, call stat_fn(resampled_index).
    Returns (point_estimate, low, high)."""
    rng = np.random.default_rng(seed)
    index = np.asarray(index)
    point = float(stat_fn(index))
    stats = np.array([stat_fn(rng.choice(index, size=len(index), replace=True)) for _ in range(n)])
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def bootstrap_pvalue_greater(diff_fn: Callable[[np.ndarray], float], index: np.ndarray, n: int = 2000,
                             seed: int = 0) -> float:
    """One-sided bootstrap p-value for H1: diff > 0 (fraction of resamples with diff <= 0, +1 smoothing)."""
    rng = np.random.default_rng(seed)
    index = np.asarray(index)
    stats = np.array([diff_fn(rng.choice(index, size=len(index), replace=True)) for _ in range(n)])
    return float((np.sum(stats <= 0) + 1) / (n + 1))


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
