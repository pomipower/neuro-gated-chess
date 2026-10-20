"""v2 agent: Laya (System 1) + classical safety layers + a better-aimed, better-informed search (System 2).

Decision order for one move (each layer can be switched off in the config, which is how the ablations are built):
  1. Endgame tablebase (<= 5 pieces, Syzygy): play the tablebase-best move.            -> system 3
  2. Mate in one: play it.                                                            -> system 3
  3. Laya scores every legal move (System 1).
  4. Gate: search if the position is critical (in check, a piece hanging, a profitable capture, few legal
     moves, endgame), or always / never / by confidence, or (v3) by a learned gate
     (src/engine/gate_features.py). Sharp positions get more simulations.
  5. System 2 (if gated in): PUCT with Laya's root prior, tactical priors at interior nodes, a quiescence
     correction of the value network at the leaves, tablebase leaves, and repetition contempt at the root.
  5b. (option laya_lookahead = k) When not searching, re-rank Laya's top-k moves by a 2-ply Laya look-ahead:
     value(m) = 1 - (Laya's best win chance for the opponent after m).
  6. Repetition contempt: when we are better (Laya win chance >= contempt_min_winchance), prefer a near-equal
     move that does not repeat a previous position.
  7. Blunder check (safe_choice): never play a move that loses material to a short forcing line when a
     preferred alternative does not.
`MoveInfoV2.system` is 1 (Laya's choice survived), 2 (search was used), 3 (tablebase / mate in one).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import chess
import chess.syzygy

from src.engine.HybridAgent import MoveInfo, _sync
from src.engine.LayaAgent import LayaAgent, LayaOutput
from src.engine.tactics import adjust_value, criticality, mate_in_one, quiescence_shift, safe_choice, tactical_prior
from src.mcts.SearchTree import SearchResult, SearchTree


@dataclass
class MoveInfoV2(MoveInfo):
    critical: int = 0             # gate fired (1) or not (0)
    safety_override: int = 0      # the blunder check replaced the preferred move
    rep_avoided: int = 0          # contempt replaced a repeating move
    special: str = ""             # "tablebase" | "mate1" | ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class HeuristicPolicy:
    """Classical stand-in for Laya (the 'is Laya even needed?' baseline): deterministic win chances from
    material after the move plus the tactical-prior signals (captures, promotions, checks, en prise).
    Same interface as LayaChessModel.evaluate."""

    def __init__(self) -> None:
        self.n_sequences = 0

    def evaluate(self, boards):
        import numpy as np

        from src.engine.tactics import material, tactical_prior

        out = []
        for b in boards:
            moves = list(b.legal_moves)
            pri = np.log(tactical_prior(b, moves) + 1e-9)
            pri -= pri.mean()
            q = []
            for m, z in zip(moves, pri):
                me = b.turn
                b.push(m)
                mate = b.is_checkmate()
                mat = material(b, me) / 100.0
                b.pop()
                x = 30.0 if mate else 0.5 * mat + 0.3 * z
                q.append(0.05 + 0.9 / (1.0 + math.exp(-max(-30.0, min(30.0, x)))))
            self.n_sequences += len(moves)
            out.append((moves, np.asarray(q, dtype=np.float32)))
        return out


class Tablebase:
    """Thin wrapper over python-chess Syzygy probing; silently disabled if no tables are found."""

    def __init__(self, directory: str | Path | None) -> None:
        self.tb = None
        self.max_pieces = 0
        files = list(Path(directory).glob("*.rtbw")) if directory and Path(directory).is_dir() else []
        if files:
            self.tb = chess.syzygy.open_tablebase(str(directory))
            self.max_pieces = max(len(f.stem.replace("v", "")) for f in files)   # "KRPvK" -> 4 pieces

    def usable(self, board: chess.Board) -> bool:
        return (self.tb is not None and chess.popcount(board.occupied) <= self.max_pieces
                and not board.castling_rights)

    def wdl_value(self, board: chess.Board) -> float | None:
        """Exact value for the side to move: +1 win, 0 draw (incl. cursed/blessed), -1 loss; None if unknown."""
        if not self.usable(board):
            return None
        try:
            wdl = self.tb.probe_wdl(board)
        except (KeyError, chess.syzygy.MissingTableError):
            return None
        return 1.0 if wdl == 2 else (-1.0 if wdl == -2 else 0.0)

    def best_move(self, board: chess.Board) -> chess.Move | None:
        if not self.usable(board):
            return None
        best, best_key = None, None
        try:
            for m in board.legal_moves:
                b = board.copy(stack=False)
                b.push(m)
                if b.is_checkmate():
                    return m
                wdl = -self.tb.probe_wdl(b)                     # our result after m
                dtz = self.tb.probe_dtz(b)                      # opponent's distance to zeroing
                zeroing = board.is_zeroing(m)
                if wdl > 0:     # winning: shortest path to a zeroing move (pawn move / capture) keeps progress
                    key = (wdl, 1 if zeroing else 0, -abs(dtz))
                elif wdl < 0:   # losing: resist as long as possible
                    key = (wdl, 0, abs(dtz))
                else:
                    key = (wdl, 0, 0)
                if best_key is None or key > best_key:
                    best, best_key = m, key
        except (KeyError, chess.syzygy.MissingTableError):
            return None
        return best


def repeats(board: chess.Board, move: chess.Move) -> bool:
    """Would `move` recreate a position that already occurred in this game? (needs the move stack)."""
    board.push(move)
    try:
        return board.is_repetition(2)
    finally:
        board.pop()


class V2Agent:
    def __init__(self, laya: LayaAgent, search_factory: Callable[[int], SearchTree] | None, opts: dict[str, Any],
                 tablebase: Tablebase | None = None, name: str = "v2") -> None:
        self.laya, self.search_factory, self.o, self.tb, self.name = laya, search_factory, opts, tablebase, name
        self._trees: dict[int, SearchTree] = {}

    def new_game(self) -> None:
        pass

    def _tree(self, sims: int) -> SearchTree:
        if sims not in self._trees:
            self._trees[sims] = self.search_factory(sims)
        return self._trees[sims]

    def _use_search(self, board: chess.Board, out: LayaOutput) -> tuple[bool, int, int]:
        """(search?, simulations, critical flag)."""
        if self.search_factory is None:
            return False, 0, 0
        gate = self.o.get("gate", "critical")
        crit = criticality(board, self.o.get("few_legal_moves", 12), self.o.get("endgame_npm", 13))
        sims = self.o["sims_critical"] if crit.sharp else self.o["sims_normal"]
        if gate == "always":
            return True, sims, int(crit.critical)
        if gate == "never":
            return False, 0, int(crit.critical)
        if gate == "confidence":
            return out.confidence <= self.o["tau"], sims, int(crit.critical)
        if gate == "learned":
            from src.engine.gate_features import gate_features

            f = gate_features(board, out.scores, out.confidence, self.o.get("safety_opts", {}),
                              self.o.get("few_legal_moves", 12), self.o.get("endgame_npm", 13))
            return self.o["learned_gate"].use_search(f), sims, int(crit.critical)
        return crit.critical, sims, int(crit.critical)

    def decide(self, board: chess.Board, laya_out: LayaOutput | None = None
               ) -> tuple[chess.Move, MoveInfoV2, LayaOutput | None, SearchResult | None]:
        _sync()
        t0 = time.perf_counter()

        def info(system: int, conf: float = math.nan, **kw: Any) -> MoveInfoV2:
            base = dict(system=system, confidence=conf, time_elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                        laya_calls=0, laya_moves_scored=0, cnn_evals=0, mcts_sims=0, max_depth=0,
                        mean_leaf_depth=0.0)
            base.update(kw)
            return MoveInfoV2(**base)

        if self.o.get("tablebase", True) and self.tb is not None:
            m = self.tb.best_move(board)
            if m is not None:
                return m, info(3, special="tablebase"), None, None
        if self.o.get("mate_in_one", True):
            m = mate_in_one(board)
            if m is not None:
                return m, info(3, special="mate1"), None, None

        out = laya_out if laya_out is not None else self.laya.evaluate(board)
        laya_ms = out.elapsed_ms
        ranking = sorted(out.scores, key=lambda m: (-out.scores[m], m.uci()))
        w_best = out.scores[ranking[0]]
        use, sims, crit = self._use_search(board, out)
        res, search_ms, preference = None, 0.0, ranking
        better = w_best >= self.o.get("contempt_min_winchance", 0.55)
        contempt = self.o.get("contempt", 0.0) if better else 0.0
        rep = {m for m in board.legal_moves if repeats(board, m)} if board.move_stack else set()
        if use:
            ts = time.perf_counter()
            res = self._tree(sims).search(board, root_scores=out.scores, repeat_moves=rep or None, contempt=contempt)
            _sync()
            search_ms = (time.perf_counter() - ts) * 1000.0
            preference = sorted(board.legal_moves,
                                key=lambda m: (-res.visits.get(m, 0), -res.q.get(m, -9.0), -out.scores.get(m, 0.0), m.uci()))
        k = int(self.o.get("laya_lookahead", 0) or 0)
        if k > 1 and not use and self.laya is not None and hasattr(self.laya, "score_many"):
            # 2-ply Laya: value of our move m = 1 - (opponent's best win chance after m), on Laya's top-k moves
            top = preference[:k]
            boards = []
            for m in top:
                b = board.copy(stack=False)
                b.push(m)
                boards.append(b)
            live = [i for i, b in enumerate(boards) if not b.is_game_over()]
            replies = self.laya.score_many([boards[i] for i in live]) if live else []
            val = {}
            for i, m in enumerate(top):
                if i in live:
                    sc = replies[live.index(i)]
                    val[m] = 1.0 - max(sc.values())
                else:
                    o = boards[i].outcome(claim_draw=False)
                    val[m] = 1.0 if (o and o.winner == board.turn) else 0.5
            top = sorted(top, key=lambda m: (-val[m], -out.scores.get(m, 0.0), m.uci()))
            preference = top + preference[k:]
        rep_avoided = 0
        if contempt > 0 and rep and preference[0] in rep:
            tol = self.o.get("contempt_tolerance", 0.06)
            alt = next((m for m in preference[1:] if m not in rep and out.scores.get(m, 0.0) >= w_best - tol), None)
            if alt is not None:
                preference = [alt] + [m for m in preference if m != alt]
                rep_avoided = 1
        override = 0
        move = preference[0]
        if self.o.get("safety", True):
            s = self.o.get("safety_opts", {})
            r = safe_choice(board, preference, s.get("candidates", 6), s.get("full_plies", 2),
                            s.get("margin_cp", 150), s.get("max_nodes", 6000), s.get("check_plies", 0))
            move, override = r.move, int(r.overridden)
        mi = info(2 if use else 1, out.confidence, laya_calls=1, laya_moves_scored=out.n_scored,
                  cnn_evals=res.cnn_evals if res else 0, mcts_sims=res.sims if res else 0,
                  max_depth=res.max_depth if res else 0, mean_leaf_depth=res.mean_leaf_depth if res else 0.0,
                  laya_ms=laya_ms, search_ms=search_ms, critical=crit, safety_override=override,
                  rep_avoided=rep_avoided)
        if laya_out is not None:          # benchmark reuse: count Laya's measured time, not the cache hit
            mi.time_elapsed_ms += laya_ms
        return move, mi, out, res

    def select_move(self, board: chess.Board) -> tuple[chess.Move, MoveInfoV2]:
        move, mi, _, _ = self.decide(board)
        return move, mi


def make_search_factory(value_fn: Any, cfg: dict[str, Any], tb: Tablebase | None,
                        improved: bool = True) -> Callable[[int], SearchTree]:
    """Search trees for the v2 agent. improved=False gives the v1 search (uniform interior priors, raw leaves)."""
    m, o = cfg["mcts"], cfg["v2"]
    temp = o.get("prior_temperature") or m["prior_temperature"]
    scale = o.get("qleaf_scale", 0.25)

    def leaf(board: chess.Board, v: float) -> float:
        return adjust_value(v, quiescence_shift(board), scale)

    def factory(sims: int) -> SearchTree:
        if not improved:
            return SearchTree(value_fn, sims, m["c_puct"], m["fpu_reduction"], temp)
        prior_t = o.get("interior_prior_temperature", 1.0)
        return SearchTree(
            value_fn, sims, m["c_puct"], m["fpu_reduction"], temp,
            interior_prior_fn=lambda b, moves: tactical_prior(b, moves, prior_t),
            leaf_adjust_fn=leaf if o.get("quiescence_leaf", True) else None,
            tb_probe=(tb.wdl_value if (tb is not None and tb.tb is not None) else None),
        )

    return factory
