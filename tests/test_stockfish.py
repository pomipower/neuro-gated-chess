from __future__ import annotations

import shutil

import chess
import pytest

from src.engine.Stockfish import StockfishJudge, StockfishPlayer, find_stockfish

try:
    find_stockfish("stockfish")
    HAVE_SF = True
except FileNotFoundError:
    HAVE_SF = False

pytestmark = pytest.mark.skipif(not HAVE_SF, reason="Stockfish binary not installed")


@pytest.fixture(scope="module")
def judge() -> StockfishJudge:
    j = StockfishJudge(nodes=20_000)
    yield j
    j.close()


def test_best_move_has_zero_cpl_and_blunder_is_large(judge: StockfishJudge) -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    res = judge.analyse(board)
    assert res.best_move == chess.Move.from_uci("a1a8")
    assert res.score_cp == 1000.0                        # mate mapped to +1000
    assert judge.cpl(board, res.best_move) == 0.0
    assert judge.cpl(board, chess.Move.from_uci("a1a2")) >= 200


def test_scores_are_side_to_move_perspective(judge: StockfishJudge) -> None:
    up_a_queen_white = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    up_a_queen_black_to_move = chess.Board("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")
    assert judge.analyse(up_a_queen_white).score_cp > 500
    assert judge.analyse(up_a_queen_black_to_move).score_cp < -500
    assert judge.eval_white_cp(up_a_queen_black_to_move) > 500


def test_judge_is_deterministic(judge: StockfishJudge) -> None:
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    j1, j2 = StockfishJudge(nodes=20_000), StockfishJudge(nodes=20_000)
    j1.analyse(chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1"))   # different hash history before the query
    a, b = j1.analyse(chess.Board(fen)), j2.analyse(chess.Board(fen))
    j1.close(); j2.close()
    assert a == b


def test_player_plays_legal_moves_and_rejects_bad_elo() -> None:
    with pytest.raises(ValueError):
        StockfishPlayer("stockfish", elo=1000)
    p = StockfishPlayer("stockfish", elo=1500, movetime_ms=20)
    board = chess.Board()
    assert p.play(board) in board.legal_moves
    p.close()


def test_process_exits_with_engines_left_open() -> None:
    """Regression: an unclosed python-chess engine used to hang interpreter exit (threads join before atexit)."""
    import subprocess
    import sys

    code = ("import chess; from src.engine.Stockfish import StockfishJudge, StockfishPlayer; "
            "StockfishJudge(nodes=2000).analyse(chess.Board()); StockfishPlayer('stockfish', 1500, 10).play(chess.Board())")
    proc = subprocess.run([sys.executable, "-c", code], timeout=60, capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()[-500:]
