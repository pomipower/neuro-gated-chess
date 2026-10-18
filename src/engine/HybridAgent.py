"""The confidence-gated router. Contract: docs/SPEC.md, `src/engine/HybridAgent.py and factory.py`.

Rule: if laya is given, call laya.evaluate(board). If search is None or not gate.use_system2(confidence),
play Laya's best move (system 1). Otherwise run search.search(board, root_scores=out.scores if
use_laya_prior else None) and play its move (system 2). The Laya computation is reused as the root prior.

All agents in the protocol are HybridAgent instances with different (laya, search, gate) settings,
built by src/engine/factory.py. Never hard-code tau.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import chess

from src.engine.gating import Gate
from src.engine.LayaAgent import LayaAgent, LayaOutput
from src.mcts.SearchTree import SearchResult, SearchTree


@dataclass
class MoveInfo:
    system: int                  # 1 or 2 (0 = Stockfish anchor)
    confidence: float            # NaN for mcts_uniform
    time_elapsed_ms: float       # wall time of the whole decision (cuda-synchronized)
    laya_calls: int
    laya_moves_scored: int
    cnn_evals: int
    mcts_sims: int
    max_depth: int               # 0 if system == 1
    mean_leaf_depth: float       # 0.0 if system == 1
    laya_ms: float = 0.0         # part of time_elapsed_ms spent in Laya
    search_ms: float = 0.0       # part spent in the search

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class HybridAgent:
    def __init__(self, laya: LayaAgent | None, search: SearchTree | None, gate: Gate,
                 use_laya_prior: bool = True, name: str = "hybrid") -> None:
        if laya is None and search is None:
            raise ValueError("need at least one of laya / search")
        self.laya, self.search, self.gate = laya, search, gate
        self.use_laya_prior = use_laya_prior
        self.name = name

    def new_game(self) -> None:
        pass

    def decide(self, board: chess.Board, laya_out: LayaOutput | None = None
               ) -> tuple[chess.Move, MoveInfo, LayaOutput | None, SearchResult | None]:
        """Full decision with intermediate results (used by the benchmark to log both systems)."""
        _sync()
        t0 = time.perf_counter()
        out = laya_out
        laya_ms = 0.0
        if self.laya is not None and out is None:
            out = self.laya.evaluate(board)
            laya_ms = out.elapsed_ms
        elif out is not None:
            laya_ms = out.elapsed_ms
        conf = out.confidence if out is not None else math.nan
        if self.search is None or (out is not None and not self.gate.use_system2(conf)):
            assert out is not None
            elapsed = (time.perf_counter() - t0) * 1000.0 if laya_out is None else laya_ms
            info = MoveInfo(1, conf, elapsed, 1, out.n_scored, 0, 0, 0, 0.0, laya_ms, 0.0)
            return out.best_move, info, out, None
        values_before = getattr(self.search.value_fn, "calls", None)
        ts = time.perf_counter()
        res = self.search.search(board, root_scores=out.scores if (out is not None and self.use_laya_prior) else None)
        _sync()
        search_ms = (time.perf_counter() - ts) * 1000.0
        elapsed = laya_ms + search_ms if laya_out is not None else (time.perf_counter() - t0) * 1000.0
        if values_before is not None:
            assert self.search.value_fn.calls - values_before == res.cnn_evals
        info = MoveInfo(2, conf, elapsed, int(out is not None), out.n_scored if out else 0, res.cnn_evals,
                        res.sims, res.max_depth, res.mean_leaf_depth, laya_ms, search_ms)
        return res.move, info, out, res

    def select_move(self, board: chess.Board) -> tuple[chess.Move, MoveInfo]:
        move, info, _, _ = self.decide(board)
        return move, info


class StockfishAnchorAgent:
    """Wraps StockfishPlayer with the select_move interface (system = 0)."""

    def __init__(self, player: Any, name: str) -> None:
        self.player, self.name = player, name

    def new_game(self) -> None:
        self.player.new_game()

    def select_move(self, board: chess.Board) -> tuple[chess.Move, MoveInfo]:
        t0 = time.perf_counter()
        move = self.player.play(board)
        ms = (time.perf_counter() - t0) * 1000.0
        return move, MoveInfo(0, math.nan, ms, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0)
