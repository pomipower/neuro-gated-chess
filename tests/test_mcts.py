from __future__ import annotations

import chess

from tests.helpers import skip_if_not_implemented

MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"   # Ra8#


@skip_if_not_implemented
def test_finds_mate_in_one_with_uninformative_value_fn() -> None:
    """Checks terminal handling and per-ply sign flipping together (value_fn always says 0)."""
    from src.mcts.SearchTree import SearchTree

    tree = SearchTree(value_fn=lambda board: 0.0, simulations=50)
    result = tree.search(chess.Board(MATE_IN_ONE))
    assert result.move == chess.Move.from_uci("a1a8")


@skip_if_not_implemented
def test_black_to_move_mate_in_one() -> None:
    from src.mcts.SearchTree import SearchTree

    tree = SearchTree(value_fn=lambda board: 0.0, simulations=50)
    mirrored = chess.Board(MATE_IN_ONE).mirror()        # same puzzle, colors swapped: ...Ra1#
    assert tree.search(mirrored).move == chess.Move.from_uci("a8a1")


@skip_if_not_implemented
def test_value_fn_is_white_perspective() -> None:
    """value_fn always says +1 (white is winning). With black to move, the root value from the side to
    move's perspective must therefore be negative. Catches a missing or doubled sign conversion."""
    from src.mcts.SearchTree import SearchTree

    tree = SearchTree(value_fn=lambda board: 1.0, simulations=30)
    result = tree.search(chess.Board("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1"))
    assert result.root_q < 0


@skip_if_not_implemented
def test_avoids_hanging_queen_with_material_value() -> None:
    """With a material-count value_fn the search must not play a move that loses the queen for nothing."""
    from src.mcts.SearchTree import SearchTree

    val = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

    def material(board: chess.Board) -> float:
        m = sum(val[p.piece_type] * (1 if p.color == chess.WHITE else -1) for p in board.piece_map().values())
        return max(-1.0, min(1.0, m / 10))

    # White is in check from the rook on e6. Blocking with Qe2 or Qe3 loses the queen to the rook; a king move does not.
    board = chess.Board("4k3/8/4r3/8/8/8/3Q4/4K3 w - - 0 1")
    result = SearchTree(value_fn=material, simulations=300).search(board)
    after = board.copy()
    after.push(result.move)
    replies = list(after.legal_moves)
    assert not any(after.is_capture(r) and after.piece_at(r.to_square).piece_type == chess.QUEEN
                   for r in replies), board.san(result.move)


@skip_if_not_implemented
def test_root_prior_shapes_search() -> None:
    from src.mcts.SearchTree import SearchTree

    board = chess.Board()
    favourite = chess.Move.from_uci("g1f3")
    scores = {m: (0.9 if m == favourite else 0.3) for m in board.legal_moves}
    result = SearchTree(value_fn=lambda b: 0.0, simulations=60).search(board, root_scores=scores)
    assert result.move == favourite
    assert result.visits[favourite] == max(result.visits.values())
    assert result.cnn_evals >= 1 and result.max_depth >= 1


@skip_if_not_implemented
def test_principal_variation_and_q_are_reported() -> None:
    from src.mcts.SearchTree import SearchTree

    board = chess.Board(MATE_IN_ONE)
    result = SearchTree(value_fn=lambda b: 0.0, simulations=50).search(board)
    assert result.pv[0] == result.move
    assert result.q[result.move] == 1.0                      # mate found: the mover wins
    replay = board.copy()
    for mv in result.pv:
        assert mv in replay.legal_moves
        replay.push(mv)


@skip_if_not_implemented
def test_v2_contempt_avoids_repetition_when_better() -> None:
    """With contempt, a root move that repeats is scored as a draw worth -contempt to the mover."""
    from src.mcts.SearchTree import SearchTree

    board = chess.Board()
    rep = {chess.Move.from_uci("g1f3")}
    scores = {m: (0.6 if m in rep else 0.5) for m in board.legal_moves}      # Laya prefers the repeating move
    plain = SearchTree(value_fn=lambda b: 0.3, simulations=200, prior_temperature=0.3).search(board, root_scores=scores)
    avoid = SearchTree(value_fn=lambda b: 0.3, simulations=200, prior_temperature=0.3).search(
        board, root_scores=scores, repeat_moves=rep, contempt=0.5)
    assert plain.move in rep
    assert avoid.move not in rep


@skip_if_not_implemented
def test_v2_interior_prior_and_leaf_adjust_are_used() -> None:
    from src.engine.tactics import adjust_value, quiescence_shift, tactical_prior
    from src.mcts.SearchTree import SearchTree

    calls = {"prior": 0, "leaf": 0}

    def prior(b, moves):
        calls["prior"] += 1
        return tactical_prior(b, moves)

    def leaf(b, v):
        calls["leaf"] += 1
        return adjust_value(v, quiescence_shift(b))

    from src.engine.tactics import material

    def value(b):           # white-perspective material value, like a (perfect) material-only network
        return max(-1.0, min(1.0, material(b, chess.WHITE) / 1000))

    # White can win the hanging queen: the search must find Nxd5.
    board = chess.Board("4k3/8/8/3q4/8/4N3/8/4K3 w - - 0 1")
    res = SearchTree(value, simulations=100, interior_prior_fn=prior, leaf_adjust_fn=leaf).search(board)
    assert res.move == chess.Move.from_uci("e3d5")
    assert calls["prior"] > 0 and calls["leaf"] > 0
