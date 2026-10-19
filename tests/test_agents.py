from __future__ import annotations

import math

import chess
import pytest

from src.engine.gating import AlwaysGate, ConfidenceGate, NeverGate
from src.engine.HybridAgent import HybridAgent
from src.engine.LayaAgent import LayaAgent, MaterialStandIn
from src.mcts.SearchTree import SearchTree


class CountingValue:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, board: chess.Board) -> float:
        self.calls += 1
        return 0.0


def laya() -> LayaAgent:
    return LayaAgent(model_name="stand-in:material", device="cpu", scorer=MaterialStandIn())


def test_laya_agent_scores_every_legal_move_and_picks_the_best() -> None:
    board = chess.Board("4k3/8/8/8/8/8/3q4/3QK3 w - - 0 1")     # Qxd2 or Kxd2 wins the black queen
    out = laya().evaluate(board)
    assert set(out.scores) == set(board.legal_moves)
    assert out.best_move.to_square == chess.D2
    assert laya().evaluate(board).scores == out.scores           # stand-in is deterministic

    assert 0.0 <= out.confidence <= 1.0 and out.n_scored == board.legal_moves.count()


def test_gate_routes_and_logs() -> None:
    board = chess.Board()
    vf = CountingValue()
    tree = SearchTree(vf, simulations=20)
    s1 = HybridAgent(laya(), tree, NeverGate())
    _, info1 = s1.select_move(board)
    assert info1.system == 1 and info1.cnn_evals == 0 and info1.mcts_sims == 0 and info1.laya_calls == 1
    s2 = HybridAgent(laya(), tree, AlwaysGate())
    _, info2 = s2.select_move(board)
    assert info2.system == 2 and info2.mcts_sims == 20 and info2.cnn_evals > 0 and info2.laya_calls == 1
    pure = HybridAgent(None, tree, AlwaysGate(), use_laya_prior=False)
    _, info3 = pure.select_move(board)
    assert info3.system == 2 and math.isnan(info3.confidence) and info3.laya_calls == 0


def test_confidence_threshold_boundary() -> None:
    board = chess.Board()
    agent = laya()
    conf = agent.evaluate(board).confidence
    tree = SearchTree(CountingValue(), simulations=5)
    assert HybridAgent(agent, tree, ConfidenceGate(conf)).select_move(board)[1].system == 2      # conf <= tau
    assert HybridAgent(agent, tree, ConfidenceGate(conf - 1e-9)).select_move(board)[1].system == 1


def test_forced_move_is_system1_under_confidence_gate() -> None:
    board = chess.Board("7k/8/8/8/8/8/6q1/7K w - - 0 1")          # only Kxg2
    agent = HybridAgent(laya(), SearchTree(CountingValue(), simulations=5), ConfidenceGate(0.99))
    move, info = agent.select_move(board)
    assert move == chess.Move.from_uci("h1g2") and info.system == 1 and info.confidence == 1.0
