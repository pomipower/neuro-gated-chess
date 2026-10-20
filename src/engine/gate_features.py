"""v3: features for a LEARNED search gate, and the gate itself.

Every feature is available at decision time, before any search, from things the v2 agent computes anyway
(Laya's scores, the criticality test, the blunder check of Laya's own ranking). The gate is a
linear model (logistic or ridge, chosen by cross-validation) trained on validation positions to predict
whether / how much search improves the move
(scripts/train_gate.py); at play time it searches when the model's score is >= a threshold that was
set on validation to give a target search rate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import chess
import numpy as np

from src.engine.tactics import criticality, material, non_pawn_material, safe_choice

FEATURES = [
    "w_best", "margin12", "margin13", "laya_entropy", "confidence",
    "n_legal", "n_captures", "in_check", "hanging", "opportunity", "few_moves", "endgame",
    "npm_total", "material_balance", "ply",
    "safety_override", "top_tactical_vs_now", "best_tactical_vs_top",
]


def gate_features(board: chess.Board, scores: dict[chess.Move, float], confidence: float,
                  safety_opts: dict[str, Any] | None = None, few_legal_moves: int = 12,
                  endgame_npm: int = 13) -> dict[str, float]:
    s = safety_opts or {}
    w = sorted(scores.values(), reverse=True)
    w = w + [w[-1]] * (3 - len(w)) if len(w) < 3 else w
    p = np.clip(np.array(list(scores.values()), dtype=float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) / 0.5
    q = np.exp(z - z.max())
    q /= q.sum()
    entropy = float(-(q * np.log(q + 1e-12)).sum() / max(math.log(len(q)), 1e-9)) if len(q) > 1 else 0.0
    crit = criticality(board, few_legal_moves, endgame_npm)
    ranking = sorted(scores, key=lambda m: (-scores[m], m.uci()))
    r = safe_choice(board, ranking, s.get("candidates", 6), s.get("full_plies", 2), s.get("margin_cp", 150),
                    s.get("max_nodes", 6000), s.get("check_plies", 0))
    now = material(board, board.turn)
    top_t = r.scores.get(ranking[0], now)
    best_t = max(r.scores.values()) if r.scores else top_t
    return {
        "w_best": w[0], "margin12": w[0] - w[1], "margin13": w[0] - w[2], "laya_entropy": entropy,
        "confidence": float(confidence) if confidence == confidence else 0.0,
        "n_legal": float(len(scores)), "n_captures": float(sum(1 for _ in board.generate_legal_captures())),
        "in_check": float(crit.in_check), "hanging": float(crit.hanging), "opportunity": float(crit.opportunity),
        "few_moves": float(crit.few_moves), "endgame": float(crit.endgame),
        "npm_total": (non_pawn_material(board, chess.WHITE) + non_pawn_material(board, chess.BLACK)) / 100.0,
        "material_balance": now / 100.0, "ply": float(board.ply()),
        "safety_override": float(r.overridden),
        "top_tactical_vs_now": max(-20.0, min(20.0, (top_t - now) / 100.0)),
        "best_tactical_vs_top": max(-20.0, min(20.0, (best_t - top_t) / 100.0)),
    }


class LearnedGate:
    """Linear gate loaded from configs/gate_learned.json (written by scripts/train_gate.py)."""

    def __init__(self, path: str | Path, rate_pct: int) -> None:
        d = json.loads(Path(path).read_text())
        self.features = d["features"]
        self.mean = np.array(d["mean"])
        self.std = np.array(d["std"])
        self.coef = np.array(d["coef"])
        self.intercept = float(d["intercept"])
        key = str(rate_pct)
        if key not in d["thresholds"]:
            raise SystemExit(f"no threshold for search rate {rate_pct}% in {path} (have {sorted(d['thresholds'])})")
        self.threshold = float(d["thresholds"][key])

    def score(self, feats: dict[str, float]) -> float:
        """Linear score (logit for the logistic model, predicted gain in cp for the ridge model)."""
        x = (np.array([feats[f] for f in self.features]) - self.mean) / self.std
        return float(x @ self.coef + self.intercept)

    def use_search(self, feats: dict[str, float]) -> bool:
        return self.score(feats) >= self.threshold
