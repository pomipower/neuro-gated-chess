"""Board encoding for the value network. Pure numpy: no torch dependency, so it is easy to test.

Layout of the (18, 8, 8) float32 array (see docs/SPEC.md):
    planes 0-11 : P N B R Q K p n b r q k   (white first); index [plane, rank, file], rank 0 = rank 1
    plane  12   : side to move (all ones if white to move, else all zeros)
    planes 13-16: castling rights  white-kingside, white-queenside, black-kingside, black-queenside
    plane  17   : en-passant target square (a single 1), only if an en-passant capture is *legal*

The encoding is a pure function of the position (what a FEN describes). Move history, halfmove clock and
repetition do not enter. The en-passant plane uses legality (like FEN output), so the same position reached
by different move orders encodes identically.
"""

from __future__ import annotations

import chess
import numpy as np

PIECE_ORDER = "PNBRQKpnbrqk"
N_PLANES = 18
PLANE_STM = 12
PLANE_CASTLE_WK, PLANE_CASTLE_WQ, PLANE_CASTLE_BK, PLANE_CASTLE_BQ = 13, 14, 15, 16
PLANE_EP = 17

_PIECE_TO_PLANE: dict[tuple[chess.PieceType, chess.Color], int] = {
    (chess.Piece.from_symbol(sym).piece_type, chess.Piece.from_symbol(sym).color): i
    for i, sym in enumerate(PIECE_ORDER)
}


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Encode `board` as a float32 array of shape (18, 8, 8)."""
    planes = np.zeros((N_PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = _PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        planes[plane, chess.square_rank(square), chess.square_file(square)] = 1.0

    if board.turn == chess.WHITE:
        planes[PLANE_STM] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[PLANE_CASTLE_WK] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[PLANE_CASTLE_WQ] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[PLANE_CASTLE_BK] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[PLANE_CASTLE_BQ] = 1.0
    if board.ep_square is not None and board.has_legal_en_passant():
        planes[PLANE_EP, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1.0
    return planes


def flip_planes(planes: np.ndarray) -> np.ndarray:
    """Array version of board_to_planes(flip_color(board)) for (..., 18, 8, 8) arrays (augmentation)."""
    out = np.empty_like(planes)
    out[..., 0:6, :, :] = planes[..., 6:12, ::-1, :]
    out[..., 6:12, :, :] = planes[..., 0:6, ::-1, :]
    out[..., PLANE_STM, :, :] = 1.0 - planes[..., PLANE_STM, :, :]
    out[..., PLANE_CASTLE_WK, :, :] = planes[..., PLANE_CASTLE_BK, :, :]
    out[..., PLANE_CASTLE_WQ, :, :] = planes[..., PLANE_CASTLE_BQ, :, :]
    out[..., PLANE_CASTLE_BK, :, :] = planes[..., PLANE_CASTLE_WK, :, :]
    out[..., PLANE_CASTLE_BQ, :, :] = planes[..., PLANE_CASTLE_WQ, :, :]
    out[..., PLANE_EP, :, :] = planes[..., PLANE_EP, ::-1, :]
    return out


def pack_planes(planes: np.ndarray) -> np.ndarray:
    """(N, 18, 8, 8) binary float -> (N, 144) uint8 (8x smaller; used for stored training data)."""
    return np.packbits(planes.reshape(planes.shape[0], -1).astype(np.uint8), axis=1)


def unpack_planes(packed: np.ndarray) -> np.ndarray:
    """Inverse of pack_planes: (N, 144) uint8 -> (N, 18, 8, 8) float32."""
    bits = np.unpackbits(packed, axis=1, count=N_PLANES * 64)
    return bits.reshape(packed.shape[0], N_PLANES, 8, 8).astype(np.float32)


def flip_color(board: chess.Board) -> chess.Board:
    """Mirror the board vertically and swap colors (turn, castling, en passant included).

    Use for optional color-flip augmentation: the encoded position is equivalent modulo color, so the
    white-perspective label must be negated.
    """
    return board.mirror()
