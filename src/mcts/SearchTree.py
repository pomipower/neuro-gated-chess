"""System 2: PUCT Monte Carlo Tree Search with value-network leaf evaluation (no rollouts).

Contract: docs/SPEC.md, section `src/mcts/SearchTree.py`. Tests: tests/test_mcts.py.

Key rules (do not change without updating the spec and docs/DEVIATIONS.md):
- `value_fn(board)` returns a WHITE-perspective value in [-1, 1]. This module is the only place where it is
  converted to side-to-move perspective:  v_stm = v if board.turn == chess.WHITE else -v.
- PUCT: score(edge) = Q(edge) + c_puct * P(edge) * sqrt(N_node) / (1 + N_edge), with Q from the perspective
  of the player who makes the move. Unvisited edges: Q = Q_node - fpu_reduction.
- Root prior = softmax(logit(w) / prior_temperature) over `root_scores` (Laya win chances); uniform if None.
  Interior nodes use uniform priors.
- Terminal nodes are scored exactly without the network: checkmate -> -1 for the side to move there;
  stalemate, insufficient material, 75-move and fivefold -> 0. Boards inside the tree are copied with
  stack=False, so threefold repetition and 50-move *claims* are not detected inside the tree (documented
  limitation).
- Backpropagation flips the sign at every ply.
- Final move: most visits; ties -> higher Q -> higher prior. No Dirichlet noise, no tree reuse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import chess
import numpy as np

_EPS = 1e-6


@dataclass
class SearchResult:
    move: chess.Move
    visits: dict[chess.Move, int]
    root_q: float            # mean value at the root, side-to-move perspective
    sims: int
    cnn_evals: int           # non-terminal leaf evaluations
    max_depth: int
    mean_leaf_depth: float
    q: dict[chess.Move, float] = field(default_factory=dict)      # mean value per root move, mover's view
    pv: list[chess.Move] = field(default_factory=list)            # most-visited line from the root


def terminal_value(board: chess.Board) -> float | None:
    """Exact value for the side to move if the game is over (no claims), else None."""
    outcome = board.outcome(claim_draw=False)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def root_prior(moves: list[chess.Move], root_scores: dict[chess.Move, float] | None,
               temperature: float) -> np.ndarray:
    n = len(moves)
    if root_scores is None:
        return np.full(n, 1.0 / n)
    lo = min(root_scores.values()) if root_scores else 0.5
    w = np.array([root_scores.get(m, lo) for m in moves], dtype=np.float64)
    w = np.clip(w, _EPS, 1 - _EPS)
    z = np.log(w / (1 - w)) / temperature
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


class Node:
    __slots__ = ("board", "terminal", "moves", "prior", "N", "W", "children", "visits", "vsum")

    def __init__(self, board: chess.Board) -> None:
        self.board = board
        self.terminal = terminal_value(board)
        self.moves: list[chess.Move] | None = None
        self.prior: np.ndarray | None = None
        self.N: np.ndarray | None = None          # visits per edge
        self.W: np.ndarray | None = None          # summed value per edge, mover's perspective
        self.children: dict[int, Node] = {}
        self.visits = 0
        self.vsum = 0.0                           # summed value, this node's side-to-move perspective

    def expand(self, prior: np.ndarray | None = None) -> None:
        self.moves = list(self.board.legal_moves)
        n = len(self.moves)
        self.prior = prior if prior is not None else np.full(n, 1.0 / n)
        self.N = np.zeros(n)
        self.W = np.zeros(n)

    @property
    def q(self) -> float:
        return self.vsum / self.visits if self.visits else 0.0


class SearchTree:
    def __init__(
        self,
        value_fn: Callable[[chess.Board], float],
        simulations: int = 400,
        c_puct: float = 1.5,
        fpu_reduction: float = 0.2,
        prior_temperature: float = 0.3,
    ) -> None:
        if simulations < 1:
            raise ValueError("simulations must be >= 1")
        self.value_fn = value_fn
        self.simulations = simulations
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.prior_temperature = prior_temperature

    def _leaf_value(self, node: Node) -> tuple[float, bool]:
        """(value for the node's side to move, used_network)."""
        if node.terminal is not None:
            return node.terminal, False
        v = float(self.value_fn(node.board))
        return (v if node.board.turn == chess.WHITE else -v), True

    def _select(self, node: Node) -> int:
        n_visits = node.N
        q = np.where(n_visits > 0, node.W / np.maximum(n_visits, 1), node.q - self.fpu_reduction)
        u = self.c_puct * node.prior * math.sqrt(max(node.visits, 1)) / (1.0 + n_visits)
        score = q + u
        return int(np.argmax(score))

    def search(
        self, board: chess.Board, root_scores: dict[chess.Move, float] | None = None
    ) -> SearchResult:
        if board.is_game_over():
            raise ValueError("search() called on a finished game")
        root = Node(board.copy(stack=False))
        v, used = self._leaf_value(root)
        cnn_evals = int(used)
        root.expand(root_prior(list(board.legal_moves), root_scores, self.prior_temperature))
        root.visits, root.vsum = 1, v
        max_depth, depth_sum = 0, 0

        for _ in range(self.simulations):
            node, path = root, []
            while True:
                if node.terminal is not None or node.moves is None:
                    break
                a = self._select(node)
                path.append((node, a))
                child = node.children.get(a)
                if child is None:
                    b = node.board.copy(stack=False)
                    b.push(node.moves[a])
                    child = Node(b)
                    node.children[a] = child
                    node = child
                    break
                node = child
            v, used = self._leaf_value(node)
            cnn_evals += int(used)
            if node.terminal is None and node.moves is None:
                node.expand()
            node.visits += 1
            node.vsum += v
            for parent, a in reversed(path):
                v = -v                              # flip: value for the player to move at `parent` (the mover)
                parent.N[a] += 1
                parent.W[a] += v                    # edge stats are from the mover's perspective
                parent.visits += 1
                parent.vsum += v                    # parent's side to move IS the mover
            depth = len(path)
            max_depth = max(max_depth, depth)
            depth_sum += depth

        q_edges = np.where(root.N > 0, root.W / np.maximum(root.N, 1), -np.inf)
        order = sorted(range(len(root.moves)), key=lambda i: (root.N[i], q_edges[i], root.prior[i]), reverse=True)
        best = order[0]
        pv, node = [root.moves[best]], root.children.get(best)
        while node is not None and node.moves is not None and node.N is not None and node.N.sum() > 0 and len(pv) < 8:
            i = int(np.argmax(node.N))
            pv.append(node.moves[i])
            node = node.children.get(i)
        return SearchResult(
            move=root.moves[best],
            visits={m: int(n) for m, n in zip(root.moves, root.N)},
            root_q=root.q,
            sims=self.simulations,
            cnn_evals=cnn_evals,
            max_depth=max_depth,
            mean_leaf_depth=depth_sum / self.simulations,
            q={m: float(q_edges[i]) for i, m in enumerate(root.moves) if root.N[i] > 0},
            pv=pv,
        )
