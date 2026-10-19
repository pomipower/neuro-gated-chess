from __future__ import annotations

import time

import chess

from src.engine.tactics import (adjust_value, criticality, mate_in_one, material, qsearch, quiescence_shift,
                                safe_choice, tactical_prior, tactical_score, _Budget, INF)


def test_material_and_qsearch_hanging_queen() -> None:
    # White to move can take an undefended queen on d5 with the knight.
    b = chess.Board("4k3/8/8/3q4/8/4N3/8/4K3 w - - 0 1")
    assert material(b, chess.WHITE) == 320 - 900
    assert qsearch(b, -INF, INF, _Budget(1000)) == 320          # after Nxd5 White is a knight up
    assert quiescence_shift(b) == 900


def test_tactical_score_sees_hanging_piece_after_move() -> None:
    # Qd2-d5?? walks into ...exd5 (pawn e6 covers d5). Qd2-d3 is safe.
    b = chess.Board("4k3/8/4p3/8/8/8/3Q4/4K3 w - - 0 1")
    bad, good = chess.Move.from_uci("d2d5"), chess.Move.from_uci("d2d3")
    assert tactical_score(b, bad) < tactical_score(b, good) - 500


def test_tactical_score_sees_knight_fork() -> None:
    # After the quiet h2-h3, ...Nc2+ forks king e1 and rook a1. Moving the rook first avoids it.
    b = chess.Board("4k3/8/8/8/1n6/8/7P/R3K3 w - - 0 1")
    quiet = chess.Move.from_uci("h2h3")
    rook_away = chess.Move.from_uci("a1a8")     # Ra8+ also escapes the fork (check, then rook safe)
    assert tactical_score(b, quiet) <= tactical_score(b, rook_away) - 400     # loses the rook to the fork


def test_safe_choice_overrides_blunder_and_keeps_safe_top() -> None:
    b = chess.Board("4k3/8/4p3/8/8/8/3Q4/4K3 w - - 0 1")
    blunder, ok = chess.Move.from_uci("d2d5"), chess.Move.from_uci("d2d3")
    pref = [blunder, ok] + [m for m in b.legal_moves if m not in (blunder, ok)]
    r = safe_choice(b, pref)
    assert r.overridden and r.move == ok
    r2 = safe_choice(b, [ok] + [m for m in b.legal_moves if m != ok])
    assert not r2.overridden and r2.move == ok


def test_safe_choice_takes_free_queen_missing_from_preferences() -> None:
    b = chess.Board("4k3/8/8/3q4/8/4N3/8/4K3 w - - 0 1")
    take = chess.Move.from_uci("e3d5")
    pref = [m for m in b.legal_moves if m != take] + [take]     # capture ranked last
    r = safe_choice(b, pref, candidates=3)
    assert r.move == take


def test_mate_in_one() -> None:
    assert mate_in_one(chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")) == chess.Move.from_uci("a1a8")
    assert mate_in_one(chess.Board()) is None


def test_tactical_prior_prefers_good_capture() -> None:
    b = chess.Board("4k3/8/8/3q4/8/4N3/8/4K3 w - - 0 1")
    moves = list(b.legal_moves)
    p = tactical_prior(b, moves)
    assert moves[int(p.argmax())] == chess.Move.from_uci("e3d5")
    assert abs(p.sum() - 1) < 1e-9


def test_criticality() -> None:
    assert not criticality(chess.Board()).critical
    assert criticality(chess.Board("4k3/8/8/3q4/8/4N3/8/4K3 w - - 0 1")).opportunity
    assert criticality(chess.Board("4k3/8/8/8/8/8/3r4/3QK3 w - - 0 1")).hanging       # queen attacked by a rook
    assert criticality(chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")).endgame


def test_adjust_value_monotone() -> None:
    assert adjust_value(0.0, 0) == 0.0
    assert adjust_value(0.0, 300) > 0.5
    assert adjust_value(0.2, -900) < -0.8


def test_safety_speed_middlegame() -> None:
    b = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8")
    t = time.perf_counter()
    safe_choice(b, list(b.legal_moves))
    assert time.perf_counter() - t < 5.0
