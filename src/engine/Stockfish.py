"""Stockfish as JUDGE (scores every move) and as calibrated rating ANCHOR. Not a main competitor.

Contract: docs/SPEC.md, section `src/engine/Stockfish.py`. Uses chess.engine (python-chess).

Score conventions (protocol section 2.1): side-to-move perspective, mate -> +/-1000 cp, clip to +/-1000.
CPL(m) = max(0, s* - s(m)); s(m) is the judge's score of m searched as the only root move (from the
mover's view); checkmate after m -> +1000, a draw after m -> 0; CPL = 0 if m is the judge's best move.

Determinism: fixed nodes, Threads=1, and a fresh `game` per analysis (ucinewgame clears the hash), so the
same position always gets the same score with the same binary.

UCI_Elo valid range is 1320-3190; Stockfish documents it as calibrated at 120s+1s against CCRL 40/4, so at
our movetime the anchor ratings are nominal.
"""

from __future__ import annotations

import atexit
import shutil
import threading
from dataclasses import dataclass
from typing import Any, Callable

import chess
import chess.engine

UCI_ELO_MIN, UCI_ELO_MAX = 1320, 3190


def find_stockfish(path: str) -> str:
    for candidate in (path, "stockfish", "/usr/games/stockfish", "/usr/local/bin/stockfish",
                      "/opt/homebrew/bin/stockfish"):
        exe = shutil.which(candidate)
        if exe:
            return exe
    raise FileNotFoundError(f"Stockfish binary not found (tried {path!r}); set judge.path in the config")


def score_to_cp(score: chess.engine.PovScore, color: chess.Color, clip: float) -> float:
    s = score.pov(color)
    if s.is_mate():
        m = s.mate()
        return clip if (m is not None and m > 0) or s == chess.engine.MateGiven else -clip
    cp = s.score()
    return float(max(-clip, min(clip, cp if cp is not None else 0)))


_LIVE: "set[_Engine]" = set()       # strong refs: an engine stays registered until closed


def _close_all() -> None:
    """Engines left open keep a python-chess background thread alive and hang interpreter exit."""
    for eng in list(_LIVE):
        eng.close()


# Python joins non-daemon threads BEFORE running atexit handlers, so a plain atexit hook never runs while an
# engine thread is alive. threading._register_atexit runs before that join; atexit is a fallback.
if hasattr(threading, "_register_atexit"):
    threading._register_atexit(_close_all)
atexit.register(_close_all)


class _Engine:
    """SimpleEngine with restart-on-crash; closed automatically at interpreter exit."""

    def __init__(self, path: str, options: dict[str, Any]) -> None:
        self.path, self.options = find_stockfish(path), options
        self.engine: chess.engine.SimpleEngine | None = None
        self._start()
        _LIVE.add(self)

    def _start(self) -> None:
        self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
        self.engine.configure(self.options)

    def call(self, fn: Callable[[chess.engine.SimpleEngine], Any]) -> Any:
        try:
            return fn(self.engine)
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError):
            self.close()
            self._start()
            return fn(self.engine)

    @property
    def name(self) -> str:
        return str(self.engine.id.get("name")) if self.engine else "?"

    def close(self) -> None:
        _LIVE.discard(self)
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None


@dataclass(frozen=True)
class JudgeResult:
    best_move: chess.Move
    score_cp: float              # best line, side to move, clipped
    second_move: chess.Move | None
    second_score_cp: float       # second line (MultiPV=2); equals score_cp if only one legal move

    @property
    def gap_cp(self) -> float:
        return self.score_cp - self.second_score_cp


class StockfishJudge:
    def __init__(self, path: str = "stockfish", nodes: int = 200_000, threads: int = 1,
                 hash_mb: int = 128, score_clip_cp: int = 1000) -> None:
        self._eng = _Engine(path, {"Threads": threads, "Hash": hash_mb})
        self.limit = chess.engine.Limit(nodes=nodes)
        self.clip = float(score_clip_cp)
        self._analysis: dict[str, JudgeResult] = {}
        self._move_scores: dict[tuple[str, str], float] = {}

    @classmethod
    def from_config(cls, cfg: dict[str, Any], nodes: int | None = None) -> "StockfishJudge":
        j = cfg["judge"]
        return cls(j["path"], nodes or j["nodes"], j["threads"], j["hash_mb"], j["score_clip_cp"])

    @property
    def version(self) -> str:
        return self._eng.name

    def analyse(self, board: chess.Board) -> JudgeResult:
        """MultiPV=2 at fixed nodes. Cached by FEN. The board must not be game-over."""
        key = board.fen()
        if key in self._analysis:
            return self._analysis[key]
        infos = self._eng.call(lambda e: e.analyse(board, self.limit, multipv=2, game=object()))
        if isinstance(infos, dict):
            infos = [infos]
        first = infos[0]
        best = first["pv"][0]
        s1 = score_to_cp(first["score"], board.turn, self.clip)
        if len(infos) > 1 and infos[1].get("pv"):
            m2, s2 = infos[1]["pv"][0], score_to_cp(infos[1]["score"], board.turn, self.clip)
        else:
            m2, s2 = None, s1
        res = JudgeResult(best, s1, m2, s2)
        self._analysis[key] = res
        self._move_scores[(key, best.uci())] = s1
        if m2 is not None:
            self._move_scores[(key, m2.uci())] = s2
        return res

    def move_score(self, board: chess.Board, move: chess.Move) -> float:
        """Score of `move` for the side to move in `board` (clipped cp)."""
        key = (board.fen(), move.uci())
        if key in self._move_scores:
            return self._move_scores[key]
        after = board.copy(stack=False)
        after.push(move)
        if after.is_checkmate():
            s = self.clip
        elif after.is_game_over(claim_draw=False):
            s = 0.0
        else:
            info = self._eng.call(lambda e: e.analyse(board, self.limit, root_moves=[move], game=object()))
            s = score_to_cp(info["score"], board.turn, self.clip)
        self._move_scores[key] = s
        return s

    def cpl(self, board: chess.Board, move: chess.Move) -> float:
        res = self.analyse(board)
        if move == res.best_move:
            return 0.0
        return max(0.0, res.score_cp - self.move_score(board, move))

    def eval_white_cp(self, board: chess.Board) -> float:
        """Best-line score from WHITE's perspective (for blend labels and the Spearman check)."""
        if board.is_checkmate():
            return -self.clip if board.turn == chess.WHITE else self.clip
        if board.is_game_over(claim_draw=False):
            return 0.0
        s = self.analyse(board).score_cp
        return s if board.turn == chess.WHITE else -s

    def close(self) -> None:
        self._eng.close()

    def __enter__(self) -> "StockfishJudge":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class StockfishPlayer:
    def __init__(self, path: str, elo: int, movetime_ms: int = 100, threads: int = 1,
                 hash_mb: int = 64) -> None:
        """Set UCI_LimitStrength=true and UCI_Elo=elo (1320 <= elo <= 3190)."""
        if not UCI_ELO_MIN <= elo <= UCI_ELO_MAX:
            raise ValueError(f"UCI_Elo must be in [{UCI_ELO_MIN}, {UCI_ELO_MAX}], got {elo}")
        self.elo = elo
        self._eng = _Engine(path, {"Threads": threads, "Hash": hash_mb, "UCI_LimitStrength": True,
                                   "UCI_Elo": elo})
        self.limit = chess.engine.Limit(time=movetime_ms / 1000.0)
        self._game: object = object()

    def new_game(self) -> None:
        self._game = object()

    def play(self, board: chess.Board) -> chess.Move:
        return self._eng.call(lambda e: e.play(board, self.limit, game=self._game)).move

    def close(self) -> None:
        self._eng.close()
