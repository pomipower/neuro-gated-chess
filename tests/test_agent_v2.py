from __future__ import annotations

import chess
import numpy as np

from src.engine.AgentV2 import Tablebase, V2Agent, make_search_factory, repeats
from src.engine.LayaAgent import LayaAgent, LayaOutput, MaterialStandIn

CFG = {
    "mcts": {"c_puct": 1.5, "fpu_reduction": 0.2, "prior_temperature": 0.3},
    "v2": {"sims_critical": 60, "sims_normal": 30, "interior_prior_temperature": 1.0, "quiescence_leaf": True,
           "qleaf_scale": 0.25, "contempt": 0.1, "contempt_min_winchance": 0.55, "few_legal_moves": 12,
           "endgame_npm": 13},
}


class FixedLaya:
    """Laya stand-in with hand-set win chances (unlisted moves get 0.3)."""

    def __init__(self, prefs: dict[str, float]) -> None:
        self.prefs = prefs

    def evaluate(self, board: chess.Board) -> LayaOutput:
        scores = {m: self.prefs.get(m.uci(), 0.3) for m in board.legal_moves}
        best = max(scores, key=lambda m: (scores[m], m.uci()))
        return LayaOutput(scores, best, 0.5, len(scores), 1.0)


def agent(laya, gate="critical", search=True, **over) -> V2Agent:
    o = {**CFG["v2"], "gate": gate, "safety": True, "safety_opts": {}, **over}
    f = make_search_factory(lambda b: 0.0, CFG, None, True) if search else None
    return V2Agent(laya, f, o, Tablebase(None), name="t")


def test_safety_net_refuses_hanging_queen() -> None:
    board = chess.Board("4k3/8/4p3/8/8/8/3Q4/4K3 w - - 0 1")
    move, info = agent(FixedLaya({"d2d5": 0.9, "d2d3": 0.6}), gate="never").select_move(board)
    assert move == chess.Move.from_uci("d2d3") and info.safety_override == 1 and info.system == 1


def test_mate_in_one_shortcut() -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    move, info = agent(FixedLaya({}), gate="never").select_move(board)
    assert move == chess.Move.from_uci("a1a8") and info.system == 3 and info.special == "mate1"


def test_critical_gate_searches_sharp_positions_only() -> None:
    quiet = chess.Board()
    sharp = chess.Board("4k3/8/8/3q4/8/4N3/8/4K3 w - - 0 1")        # free queen on d5
    a = agent(LayaAgent(model_name="stand-in", device="cpu", scorer=MaterialStandIn()))
    assert a.select_move(quiet)[1].system == 1
    move, info = a.select_move(sharp)
    assert info.system == 2 and info.critical == 1 and move == chess.Move.from_uci("e3d5")


def test_contempt_avoids_repetition_when_better() -> None:
    board = chess.Board()
    for u in ["g1f3", "g8f6", "f3g1", "f6g8"]:      # back to the start position once
        board.push_uci(u)
    assert repeats(board, chess.Move.from_uci("g1f3"))
    laya = FixedLaya({"g1f3": 0.62, "b1c3": 0.60})
    move, info = agent(laya, gate="never").select_move(board)
    assert move == chess.Move.from_uci("b1c3") and info.rep_avoided == 1
    worse = FixedLaya({"g1f3": 0.45, "b1c3": 0.44})           # not better: repeating is fine
    assert agent(worse, gate="never").select_move(board)[0] == chess.Move.from_uci("g1f3")


def test_v2_agent_plays_a_short_legal_game() -> None:
    a = agent(LayaAgent(model_name="stand-in", device="cpu", scorer=MaterialStandIn()))
    board = chess.Board()
    for _ in range(30):
        if board.is_game_over():
            break
        move, info = a.select_move(board.copy())
        assert move in board.legal_moves
        board.push(move)
    assert board.ply() > 10


def test_tablebase_disabled_without_files(tmp_path) -> None:
    tb = Tablebase(tmp_path)
    assert tb.tb is None and tb.best_move(chess.Board("8/8/8/4k3/8/8/8/3QK3 w - - 0 1")) is None
