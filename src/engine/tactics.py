"""v2 tactical helpers (pure python-chess, CPU): material search, blunder check, move priors, criticality.

All scores are centipawns of MATERIAL from the perspective stated in each function. Nothing here uses a
neural network; these are deliberately simple, classical tools that cover the failure we measured in v1
(17% of Laya's moves lose >= 2 pawns, mostly to short tactics).

- qsearch / tactical_score: negamax with alpha-beta over `full_plies` plies of all legal moves, then captures
  (and check evasions) until quiet. Checkmate = -MATE for the side to move. A node budget bounds the cost.
- safe_choice: the blunder check. Given moves in order of preference, keep the first one whose tactical score
  is within `margin_cp` of the best candidate; candidates are the top-K preferences plus every capture.
- tactical_prior: cheap PUCT priors for interior search nodes (captures by MVV-LVA, promotions, checks;
  moves that put a piece en prise are discouraged).
- criticality: is the position sharp (in check, something hanging, a winning capture available, few legal
  moves, endgame)? Used by the v2 gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess
import numpy as np

VAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
MATE = 100_000
INF = 10 * MATE


def material(board: chess.Board, color: chess.Color) -> int:
    """Material balance in cp for `color`."""
    s = 0
    for pt, v in VAL.items():
        if v:
            s += v * (chess.popcount(board.pieces_mask(pt, color)) - chess.popcount(board.pieces_mask(pt, not color)))
    return s


def non_pawn_material(board: chess.Board, color: chess.Color) -> int:
    return sum(VAL[pt] * chess.popcount(board.pieces_mask(pt, color))
               for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))


def _victim_value(board: chess.Board, m: chess.Move) -> int:
    if board.is_en_passant(m):
        return VAL[chess.PAWN]
    p = board.piece_at(m.to_square)
    return VAL[p.piece_type] if p else 0


def _mvv_lva(board: chess.Board, m: chess.Move) -> int:
    attacker = board.piece_at(m.from_square)
    return 10 * (_victim_value(board, m) + (VAL[m.promotion] if m.promotion else 0)) - (VAL[attacker.piece_type] if attacker else 0) // 10


class _Budget:
    __slots__ = ("left",)

    def __init__(self, n: int) -> None:
        self.left = n


def qsearch(board: chess.Board, alpha: int, beta: int, budget: _Budget, depth: int = 0, checks: int = 0) -> int:
    """Quiescence: material score for the side to move after resolving captures (and check evasions).
    With checks > 0, quiet checking moves are also tried for that many more quiescence plies (finds forcing
    mating attacks and check-then-capture tactics beyond the full-width horizon)."""
    budget.left -= 1
    in_check = board.is_check()
    if in_check:
        moves = list(board.legal_moves)
        if not moves:
            return -MATE + depth
        best = -INF
    else:
        stand = material(board, board.turn)
        if stand >= beta or budget.left <= 0 or depth > 12:
            return stand
        alpha = max(alpha, stand)
        best = stand
        moves = [m for m in board.generate_legal_captures()] + \
                [m for m in board.generate_legal_moves(to_mask=chess.BB_RANK_1 | chess.BB_RANK_8)
                 if m.promotion == chess.QUEEN and not board.is_capture(m)]
        if checks > 0:
            moves += [m for m in board.generate_legal_moves()
                      if not board.is_capture(m) and not m.promotion and board.gives_check(m)]
    if budget.left <= 0 and in_check:
        return material(board, board.turn)
    moves.sort(key=lambda m: -_mvv_lva(board, m))
    for m in moves:
        board.push(m)
        score = -qsearch(board, -beta, -alpha, budget, depth + 1, max(0, checks - 1))
        board.pop()
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best


def _search(board: chess.Board, plies: int, alpha: int, beta: int, budget: _Budget, depth: int,
            checks: int = 0) -> int:
    budget.left -= 1
    moves = list(board.legal_moves)
    if not moves:
        return -MATE + depth if board.is_check() else 0
    if plies <= 0 or budget.left <= 0:
        return qsearch(board, alpha, beta, budget, depth, checks if budget.left > 0 else 0)
    caps = [m for m in moves if board.is_capture(m) or m.promotion]
    caps.sort(key=lambda m: -_mvv_lva(board, m))
    quiet = [m for m in moves if not (board.is_capture(m) or m.promotion)]
    best = -INF
    for m in caps + quiet:
        board.push(m)
        score = -_search(board, plies - 1, -beta, -alpha, budget, depth + 1, checks)
        board.pop()
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best


def tactical_score(board: chess.Board, move: chess.Move, full_plies: int = 2, max_nodes: int = 6000,
                   check_plies: int = 0) -> int:
    """Material outcome (cp, mover's perspective) of `move` after `full_plies` full-width replies + quiescence
    (with `check_plies` extra plies of checking moves in the quiescence search)."""
    b = board.copy(stack=False)
    b.push(move)
    return -_search(b, full_plies, -INF, INF, _Budget(max_nodes), 1, check_plies)


def mate_in_one(board: chess.Board) -> chess.Move | None:
    for m in board.legal_moves:
        board.push(m)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return m
    return None


@dataclass
class SafetyResult:
    move: chess.Move
    overridden: bool
    scores: dict[chess.Move, int] = field(default_factory=dict)


def safe_choice(board: chess.Board, preference: list[chess.Move], candidates: int = 6, full_plies: int = 2,
                margin_cp: int = 150, max_nodes: int = 6000, check_plies: int = 0) -> SafetyResult:
    """Blunder check. `preference` = legal moves, most preferred first (Laya ranking or search visits)."""
    top = preference[0]
    scores: dict[chess.Move, int] = {}

    def score(m: chess.Move) -> int:
        if m not in scores:
            scores[m] = tactical_score(board, m, full_plies, max_nodes, check_plies)
        return scores[m]

    score(top)
    captures = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    for m in captures:
        score(m)
    now = material(board, board.turn)
    best = max(scores.values())
    if scores[top] >= best - margin_cp and scores[top] >= now - margin_cp:
        return SafetyResult(top, False, scores)                 # the usual case: Laya's move is tactically fine
    shortlist = list(preference[:candidates])
    for m in shortlist:
        score(m)
    best = max(scores.values())
    rank = {m: i for i, m in enumerate(preference)}
    ordered = shortlist + sorted((m for m in scores if m not in shortlist), key=lambda m: rank.get(m, len(rank)))
    for m in ordered:                                           # most preferred move that is (nearly) the best
        if scores[m] >= best - margin_cp:
            return SafetyResult(m, m != top, scores)
    return SafetyResult(top, False, scores)                     # unreachable (best itself qualifies)


def _attacked_by_cheaper_or_hanging(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    """Is `color`'s piece on `square` attacked by a cheaper enemy piece, or attacked and undefended?"""
    p = board.piece_at(square)
    if p is None or p.piece_type == chess.KING:
        return False
    attackers = board.attackers(not color, square)
    if not attackers:
        return False
    if not board.attackers(color, square):
        return True
    cheapest = min(VAL[board.piece_type_at(sq)] or 10_000 for sq in attackers)
    return cheapest < VAL[p.piece_type]


def tactical_prior(board: chess.Board, moves: list[chess.Move], temperature: float = 1.0) -> np.ndarray:
    """PUCT prior for interior nodes: favour good captures, promotions and checks; discourage en prise moves.

    Written for speed (it runs at every expanded search node): no push/pop, attack maps only.
    """
    logits = np.zeros(len(moves))
    me, them = board.turn, not board.turn
    their_pawns = board.pieces_mask(chess.PAWN, them)
    king_sq = board.king(them)
    for i, m in enumerate(moves):
        z = 0.0
        pt = board.piece_type_at(m.from_square)
        av = VAL[pt] if pt else 0
        to = m.to_square
        victim = _victim_value(board, m)
        if victim:
            z += 1.0 + max(0.0, (victim - av) / 300.0) + victim / 900.0
        if m.promotion:
            z += 2.0 if m.promotion == chess.QUEEN else -1.0
        if king_sq is not None and pt in (chess.KNIGHT, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.PAWN):
            # cheap direct-check test: does the moved piece attack the enemy king from its new square?
            if pt == chess.KNIGHT and chess.BB_KNIGHT_ATTACKS[to] & chess.BB_SQUARES[king_sq]:
                z += 0.8
            elif pt == chess.PAWN and chess.BB_PAWN_ATTACKS[me][to] & chess.BB_SQUARES[king_sq]:
                z += 0.8
        if pt not in (chess.KING, chess.PAWN, None) and victim < av:
            attacked_by_pawn = bool(chess.BB_PAWN_ATTACKS[me][to] & their_pawns)
            if attacked_by_pawn or (board.is_attacked_by(them, to) and not board.is_attacked_by(me, to)):
                z -= 1.5
        logits[i] = z
    logits /= max(temperature, 1e-3)
    logits -= logits.max()
    p = np.exp(logits)
    return p / p.sum()


@dataclass
class Criticality:
    in_check: bool
    hanging: bool          # one of our pieces is attacked by a cheaper piece or undefended
    opportunity: bool      # we can capture something for profit
    few_moves: bool
    endgame: bool

    @property
    def sharp(self) -> bool:
        return self.in_check or self.hanging or self.opportunity

    @property
    def critical(self) -> bool:
        return self.sharp or self.few_moves or self.endgame


def criticality(board: chess.Board, few_legal_moves: int = 12, endgame_npm_pawns: int = 13) -> Criticality:
    me = board.turn
    hanging = any(_attacked_by_cheaper_or_hanging(board, sq, me)
                  for sq in chess.scan_forward(board.occupied_co[me]))
    opportunity = False
    for m in board.generate_legal_captures():
        victim = _victim_value(board, m)
        attacker = board.piece_at(m.from_square)
        defended = bool(board.attackers(not me, m.to_square))
        if not defended or victim > VAL[attacker.piece_type]:
            opportunity = True
            break
    n_legal = board.legal_moves.count()
    npm = max(non_pawn_material(board, chess.WHITE), non_pawn_material(board, chess.BLACK)) / 100
    return Criticality(board.is_check(), hanging, opportunity, n_legal <= few_legal_moves, npm <= endgame_npm_pawns)


def quiescence_shift(board: chess.Board, max_nodes: int = 60) -> int:
    """How much material (cp, side to move) the pending captures are worth: qsearch - static material."""
    if not board.is_check() and not any(True for _ in board.generate_legal_captures()):
        return 0
    return qsearch(board, -INF, INF, _Budget(max_nodes)) - material(board, board.turn)


def adjust_value(v_stm: float, shift_cp: int, scale_per_pawn: float = 0.25) -> float:
    """Shift a [-1, 1] value by a material swing, in the logit-like tanh(cp/400) space of the labels."""
    if shift_cp == 0:
        return v_stm
    if abs(shift_cp) >= MATE // 2:
        return 1.0 if shift_cp > 0 else -1.0
    v = max(-0.999, min(0.999, v_stm))
    return math.tanh(math.atanh(v) + scale_per_pawn * shift_cp / 100.0)
