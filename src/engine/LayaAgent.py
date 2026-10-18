"""System 1: LayaChess move scorer. Contract: docs/SPEC.md, section `src/engine/LayaAgent.py`.

LayaChess (`datafreak/laya-chess`, Apache-2.0, ~0.4B params) is a chess fine-tune of the Laya base model.
Reference code: github.com/devroopsaha744/LayaChess (clone to third_party/LayaChess and add
third_party/LayaChess/engine to PYTHONPATH; it also needs `laya` from github.com/NandhaKishorM/laya).

Confirmed from the LayaChess source (engine/laya_chess/model.py):
- `LayaChessModel(checkpoint, revision=None, device="auto", dtype=None, batch_size=96, cache_size=200_000)`
- `evaluate(boards) -> [(legal_moves: list[chess.Move], q: np.ndarray float32)]`, one entry per board, q[i] is
  the win chance for the side to move after legal_moves[i]. q is the expectation over 10 equal-width win-chance
  levels, so it lies in [0.05, 0.95]. All legal moves of all boards are batched (batch_size sequences per pass).
- `score_moves(board) -> [(move, win)]` sorted best-first.
- It keeps an LRU cache keyed by position, so repeated positions are free. We call `evaluate` and clear
  that cache before timed runs (otherwise wall times are not comparable).
- One position = one forward pass per legal move (~35), fp16 on CUDA.

This module must not call Stockfish or any search. Falls back to CPU if CUDA is unavailable.
`model_name: "stand-in:material"` loads MaterialStandIn (a cheap fake with the same interface) for smoke tests;
never report results from it.
"""

from __future__ import annotations

import math
import time
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import chess
import numpy as np

from src.engine.gating import margin, norm_entropy_conf, top1_softmax


class MoveScorer(Protocol):
    def evaluate(self, boards: Sequence[chess.Board]) -> list[tuple[list[chess.Move], np.ndarray]]: ...


@dataclass(frozen=True)
class LayaOutput:
    scores: dict[chess.Move, float]     # win chance in [0, 1] for every legal move
    best_move: chess.Move
    confidence: float
    n_scored: int
    elapsed_ms: float


def confidence_fn(name: str, temperature: float) -> Callable[[Sequence[float]], float]:
    if name == "margin":
        return margin
    if name == "top1_softmax":
        return lambda w: top1_softmax(w, temperature)
    if name in ("norm_entropy", "norm_entropy_conf"):
        return lambda w: norm_entropy_conf(w, temperature)
    raise ValueError(f"unknown confidence function {name!r}")


_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


class MaterialStandIn:
    """Fake LayaChess for smoke tests only: win chance after each move = logistic(material + noise).

    Same `evaluate` interface as LayaChessModel. Deterministic per position.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.n_sequences = 0

    def evaluate(self, boards: Sequence[chess.Board]) -> list[tuple[list[chess.Move], np.ndarray]]:
        out = []
        for b in boards:
            moves = list(b.legal_moves)
            rng = np.random.default_rng(zlib.crc32(f"{b.board_fen()} {b.turn} {self.seed}".encode()))
            q = []
            for mv in moves:
                me = b.turn
                b.push(mv)
                if b.is_checkmate():
                    m = 30.0
                else:
                    m = sum(_VAL[p.piece_type] * (1 if p.color == me else -1) for p in b.piece_map().values())
                b.pop()
                q.append(0.05 + 0.9 / (1 + math.exp(-0.5 * m + rng.normal(0, 0.6))))
            self.n_sequences += len(moves)
            out.append((moves, np.asarray(q, dtype=np.float32)))
        return out


def load_scorer(model_name: str, revision: str | None, device: str, batch_size: int = 96) -> MoveScorer:
    if model_name.startswith("stand-in"):
        return MaterialStandIn()
    try:
        from laya_chess import LayaChessModel
    except ImportError as e:
        raise ImportError(
            "laya_chess not importable. Clone https://github.com/devroopsaha744/LayaChess to third_party/LayaChess, "
            "pip install -r third_party/LayaChess/engine/requirements.txt, and add third_party/LayaChess/engine "
            "to PYTHONPATH (see README)."
        ) from e
    return LayaChessModel(model_name, revision=revision, device=device, batch_size=batch_size, verbose=True)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


class LayaAgent:
    def __init__(
        self,
        model_name: str = "datafreak/laya-chess",
        revision: str | None = None,
        device: str = "cuda",
        confidence: str = "margin",
        temperature: float = 0.1,
        scorer: MoveScorer | None = None,
        batch_size: int = 96,
        use_cache: bool = False,
    ) -> None:
        from src.utils import pick_device

        self.device = pick_device(device)
        self.model_name = model_name
        self.scorer = scorer if scorer is not None else load_scorer(model_name, revision, self.device, batch_size)
        self.confidence_name, self.temperature = confidence, temperature
        self._conf = confidence_fn(confidence, temperature)
        self.use_cache = use_cache

    @classmethod
    def from_config(cls, cfg: dict[str, Any], gate_cfg: dict[str, Any] | None = None,
                    scorer: MoveScorer | None = None) -> "LayaAgent":
        lc = cfg["laya"]
        conf = (gate_cfg or {}).get("confidence", lc["confidence"])
        temp = (gate_cfg or {}).get("temperature", lc["temperature"])
        return cls(lc["model_name"], lc.get("revision"), lc["device"], conf, temp, scorer,
                   lc.get("batch_size", 96), lc.get("use_cache", False))

    def _clear_cache(self) -> None:
        cache = getattr(self.scorer, "cache", None)
        if cache is not None and not self.use_cache:
            cache.clear()

    def score_many(self, boards: Sequence[chess.Board]) -> list[dict[chess.Move, float]]:
        """Untimed batched scoring for calibration (cache allowed)."""
        return [dict(zip(moves, map(float, q))) for moves, q in self.scorer.evaluate(list(boards))]

    def output_from_scores(self, scores: dict[chess.Move, float], elapsed_ms: float = float("nan")) -> LayaOutput:
        best = max(scores, key=lambda m: (scores[m], m.uci()))
        return LayaOutput(scores, best, float(self._conf(list(scores.values()))), len(scores), elapsed_ms)

    def evaluate(self, board: chess.Board) -> LayaOutput:
        if board.is_game_over():
            raise ValueError("evaluate() called on a finished game")
        self._clear_cache()
        _sync(self.device)
        t0 = time.perf_counter()
        moves, q = self.scorer.evaluate([board])[0]
        _sync(self.device)
        elapsed = (time.perf_counter() - t0) * 1000.0
        legal = set(board.legal_moves)
        scores = {m: float(v) for m, v in zip(moves, q) if m in legal}
        if not scores:
            raise RuntimeError("Laya returned no legal moves")
        return self.output_from_scores(scores, elapsed)
