"""PGN months -> labelled positions -> torch Dataset. Contract: docs/SPEC.md, `dataset.py and train.py`.

- Reads `.pgn`, `.pgn.zst` (Lichess) and `.zip` (Lichess Elite monthly archives) files.
- Keep games with Termination == "Normal" (if the header exists and data.require_normal_termination),
  a result in {1-0, 0-1, 1/2-1/2}, and >= data.min_plies plies.
- Sample <= n positions per game with ply >= min_ply; de-duplicate by the first four FEN fields.
- Label z = game result from WHITE's perspective (+1 / 0 / -1).
- Splits are by MONTH, never by random position. `month_of` reads YYYY-MM from the file name.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset

from src.model.encoding import board_to_planes, flip_planes, pack_planes, unpack_planes

_MONTH_RE = re.compile(r"(\d{4})-(\d{2})")
_RESULTS = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


def month_of(path: str | Path) -> str | None:
    m = _MONTH_RE.search(Path(path).name)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def _text_streams(path: Path) -> Iterator[io.TextIOBase]:
    name = path.name.lower()
    if name.endswith(".zst"):
        import zstandard

        with open(path, "rb") as fh:
            reader = zstandard.ZstdDecompressor().stream_reader(fh)
            yield io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    elif name.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            for member in sorted(zf.namelist()):
                if member.lower().endswith(".pgn"):
                    with zf.open(member) as fh:
                        yield io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
    else:
        with open(path, encoding="utf-8", errors="replace") as fh:
            yield fh


def iter_games(path: Path) -> Iterator[chess.pgn.Game]:
    for stream in _text_streams(Path(path)):
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            yield game


def white_result(game: chess.pgn.Game) -> float | None:
    """+1.0 white won, -1.0 black won, 0.0 draw, None for '*' or anything else."""
    return _RESULTS.get(game.headers.get("Result", "*"))


def keep_game(game: chess.pgn.Game, min_plies: int, require_normal: bool) -> bool:
    if white_result(game) is None:
        return False
    if game.headers.get("Variant", "Standard") not in ("Standard", "standard", ""):
        return False
    if "FEN" in game.headers or game.headers.get("SetUp") == "1":
        return False
    if require_normal and game.headers.get("Termination", "Normal") != "Normal":
        return False
    return game.end().ply() >= min_plies


def mainline_boards(game: chess.pgn.Game) -> list[chess.Board]:
    """Board after each ply (index i = position after i plies; index 0 = start)."""
    board = game.board()
    out = [board.copy(stack=False)]
    for move in game.mainline_moves():
        board.push(move)
        out.append(board.copy(stack=False))
    return out


def fen_key(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def sample_positions(game: chess.pgn.Game, n: int, min_ply: int, rng: np.random.Generator,
                     max_ply: int | None = None) -> list[tuple[int, chess.Board]]:
    """Return up to n (ply, board) pairs from the game's mainline, excluding game-over positions."""
    boards = mainline_boards(game)
    hi = len(boards) - 1 if max_ply is None else min(max_ply, len(boards) - 1)
    candidates = [i for i in range(min_ply, hi + 1) if not boards[i].is_game_over()]
    if not candidates:
        return []
    picks = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
    return [(int(i), boards[int(i)]) for i in sorted(picks)]


@dataclass
class PositionSet:
    fens: list[str] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    game_ids: list[str] = field(default_factory=list)
    plies: list[int] = field(default_factory=list)

    def to_npz(self, path: Path, sf_cp_white: np.ndarray | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        planes = np.stack([board_to_planes(chess.Board(f)) for f in self.fens]) if self.fens else \
            np.zeros((0, 18, 8, 8), np.float32)
        arrays = dict(
            planes=pack_planes(planes),
            y=np.asarray(self.y, dtype=np.float32),
            fens=np.asarray(self.fens),
            game_ids=np.asarray(self.game_ids),
            plies=np.asarray(self.plies, dtype=np.int32),
        )
        if sf_cp_white is not None:
            arrays["sf_cp_white"] = sf_cp_white.astype(np.float32)
        np.savez_compressed(path, **arrays)


def build_training_set(paths: Sequence[Path], per_game: int, min_ply: int, min_plies: int,
                       require_normal: bool, rng: np.random.Generator, max_positions: int | None = None,
                       seen: set[str] | None = None) -> PositionSet:
    """Sample labelled positions (white-perspective game result) from the given PGN files."""
    out, seen = PositionSet(), (seen if seen is not None else set())
    for path in paths:
        for gi, game in enumerate(iter_games(path)):
            if not keep_game(game, min_plies, require_normal):
                continue
            z = white_result(game)
            gid = game.headers.get("Site") or f"{Path(path).name}#{gi}"
            for ply, board in sample_positions(game, per_game, min_ply, rng):
                key = fen_key(board)
                if key in seen:
                    continue
                seen.add(key)
                out.fens.append(board.fen())
                out.y.append(float(z))
                out.game_ids.append(gid)
                out.plies.append(ply)
            if max_positions is not None and len(out.fens) >= max_positions:
                return out
    return out


def blend_labels(z: np.ndarray, sf_cp_white: np.ndarray | None, lam: float) -> np.ndarray:
    """label.mode == 'blend': y = lam * z + (1 - lam) * tanh(cp / 400); positions without cp keep z."""
    if sf_cp_white is None:
        return z
    t = np.tanh(np.nan_to_num(sf_cp_white, nan=0.0) / 400.0)
    y = lam * z + (1.0 - lam) * t
    return np.where(np.isnan(sf_cp_white), z, y).astype(np.float32)


class PositionDataset(Dataset):
    """Packed planes + white-perspective labels. __getitem__ -> (planes (18,8,8) float32, label (1,) float32).

    With color_flip, each item is color-flipped with probability 0.5 and its label negated.
    """

    def __init__(self, planes_packed: np.ndarray, y: np.ndarray, color_flip: bool = False,
                 seed: int = 0) -> None:
        self.packed = planes_packed
        self.y = np.asarray(y, dtype=np.float32)
        self.color_flip = color_flip
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = unpack_planes(self.packed[idx:idx + 1])[0]
        y = self.y[idx]
        if self.color_flip and self._rng.random() < 0.5:
            x, y = flip_planes(x), -y
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor([y], dtype=torch.float32)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}
