from __future__ import annotations

import chess
import numpy as np

from src.model.encoding import (
    N_PLANES,
    PLANE_CASTLE_BK,
    PLANE_CASTLE_BQ,
    PLANE_CASTLE_WK,
    PLANE_CASTLE_WQ,
    PLANE_EP,
    PLANE_STM,
    board_to_planes,
    flip_color,
)


def test_shape_dtype_and_piece_count() -> None:
    p = board_to_planes(chess.Board())
    assert p.shape == (N_PLANES, 8, 8) and p.dtype == np.float32
    assert p[:12].sum() == 32


def test_piece_planes_follow_documented_order_and_orientation() -> None:
    p = board_to_planes(chess.Board())
    assert p[3, 0, 0] == 1.0          # white rook on a1  (plane R=3, rank index 0, file a=0)
    assert p[0, 1].sum() == 8.0       # white pawns on rank 2
    assert p[11, 7, 4] == 1.0         # black king on e8  (plane k=11)
    assert p[6, 6].sum() == 8.0       # black pawns on rank 7


def test_side_to_move_is_encoded() -> None:
    board = chess.Board()
    assert board_to_planes(board)[PLANE_STM].min() == 1.0
    board.push_san("e4")
    assert board_to_planes(board)[PLANE_STM].max() == 0.0


def test_same_pieces_different_side_to_move_encode_differently() -> None:
    """Guards the v1 bug: without a side-to-move plane the net could not tell these apart."""
    white_to_move = chess.Board()
    black_to_move = chess.Board()
    black_to_move.turn = chess.BLACK
    a, b = board_to_planes(white_to_move), board_to_planes(black_to_move)
    assert np.array_equal(a[:12], b[:12])
    assert not np.array_equal(a, b)


def test_castling_planes() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1")
    p = board_to_planes(board)
    assert p[PLANE_CASTLE_WK].min() == 1.0 and p[PLANE_CASTLE_BQ].min() == 1.0
    assert p[PLANE_CASTLE_WQ].max() == 0.0 and p[PLANE_CASTLE_BK].max() == 0.0


def test_en_passant_plane_only_when_capture_is_legal() -> None:
    quiet = chess.Board()
    quiet.push_san("e4")                     # ep square e3 exists, but no black pawn can capture
    assert board_to_planes(quiet)[PLANE_EP].sum() == 0.0

    board = chess.Board()
    for san in ["e4", "a6", "e5", "d5"]:     # white pawn e5 can capture d5 en passant, target d6
        board.push_san(san)
    p = board_to_planes(board)
    assert p[PLANE_EP].sum() == 1.0 and p[PLANE_EP, 5, 3] == 1.0


def test_encoding_is_a_pure_function_of_the_position() -> None:
    a, b = chess.Board(), chess.Board()
    for san in ["e4", "e5", "Nf3", "Nc6"]:
        a.push_san(san)
    for san in ["Nf3", "Nc6", "e4", "e5"]:   # transposition
        b.push_san(san)
    assert np.array_equal(board_to_planes(a), board_to_planes(b))
    assert np.array_equal(board_to_planes(a), board_to_planes(chess.Board(a.fen())))


def test_color_flip_swaps_colors_and_mirrors_ranks() -> None:
    board = chess.Board()
    for san in ["e4", "a6", "e5", "d5", "Nf3"]:
        board.push_san(san)
    board.set_castling_fen("Kq")
    pb, pm = board_to_planes(board), board_to_planes(flip_color(board))
    assert np.array_equal(pm[0:6], pb[6:12][:, ::-1, :])
    assert np.array_equal(pm[6:12], pb[0:6][:, ::-1, :])
    assert np.array_equal(pm[PLANE_STM], 1.0 - pb[PLANE_STM])
    assert np.array_equal(pm[PLANE_CASTLE_WK], pb[PLANE_CASTLE_BK])
    assert np.array_equal(pm[PLANE_CASTLE_WQ], pb[PLANE_CASTLE_BQ])
    assert np.array_equal(pm[PLANE_CASTLE_BK], pb[PLANE_CASTLE_WK])
    assert np.array_equal(pm[PLANE_CASTLE_BQ], pb[PLANE_CASTLE_WQ])
    assert np.array_equal(pm[PLANE_EP], pb[PLANE_EP][::-1, :])


def test_flip_planes_matches_board_mirror() -> None:
    from src.model.encoding import flip_planes

    board = chess.Board()
    for san in ["e4", "a6", "e5", "d5"]:
        board.push_san(san)
    assert np.array_equal(flip_planes(board_to_planes(board)), board_to_planes(flip_color(board)))


def test_pack_unpack_roundtrip() -> None:
    from src.model.encoding import pack_planes, unpack_planes

    boards = [chess.Board(), chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1")]
    planes = np.stack([board_to_planes(b) for b in boards])
    packed = pack_planes(planes)
    assert packed.shape == (2, 144) and packed.dtype == np.uint8
    assert np.array_equal(unpack_planes(packed), planes)
