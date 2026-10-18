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
