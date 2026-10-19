"""v2: add Stockfish evaluations to the value-net training data, in parallel (for label.mode = blend).

Reads <src>/train.npz and <src>/val.npz (the v1 sets, unchanged), evaluates every position with Stockfish at a
fixed node count in several worker processes, and writes <dst>/train.npz and <dst>/val.npz with an extra
`sf_cp_white` array (white perspective, mate -> +/-1000, clipped to +/-1000, same convention as the judge).
The v1 files are never modified. Progress is saved every chunk, so an interrupted run resumes where it stopped.

Run from the repo root:
    python -m scripts.label_sf --src data/processed --dst data/processed_v2 --nodes 10000 --workers 6
"""

from __future__ import annotations

import argparse
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.engine
import numpy as np
from tqdm import tqdm

from src.engine.Stockfish import find_stockfish

CHUNK = 5000
_ENGINE = None
_LIMIT = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="data/processed")
    p.add_argument("--dst", default="data/processed_v2")
    p.add_argument("--nodes", type=int, default=10000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--stockfish", default="stockfish")
    p.add_argument("--limit", type=int, default=None, help="label only the first N positions (smoke test)")
    return p.parse_args()


def _init(path: str, nodes: int) -> None:
    global _ENGINE, _LIMIT
    _ENGINE = chess.engine.SimpleEngine.popen_uci(path)
    _ENGINE.configure({"Threads": 1, "Hash": 16})
    _LIMIT = chess.engine.Limit(nodes=nodes)


def _eval(fen: str) -> float:
    board = chess.Board(fen)
    outcome = board.outcome(claim_draw=False)
    if outcome is not None:
        return 0.0 if outcome.winner is None else (1000.0 if outcome.winner == chess.WHITE else -1000.0)
    info = _ENGINE.analyse(board, _LIMIT, game=object())
    score = info["score"].white()
    if score.is_mate():
        m = score.mate()
        return 1000.0 if (m is not None and m > 0) or score == chess.engine.MateGiven else -1000.0
    return float(max(-1000, min(1000, score.score())))


def label_file(src: Path, dst: Path, path: str, nodes: int, workers: int, limit: int | None) -> None:
    with np.load(src) as z:
        data = {k: z[k] for k in z.files}
    fens = [str(f) for f in data["fens"]][:limit] if limit else [str(f) for f in data["fens"]]
    partial = dst.with_suffix(".partial.npy")
    done = np.load(partial) if partial.exists() else np.zeros(0, np.float32)
    out = list(done)
    with Pool(workers, initializer=_init, initargs=(path, nodes)) as pool:
        bar = tqdm(total=len(fens), initial=len(out), desc=f"SF {src.name}", unit="pos")
        for start in range(len(out), len(fens), CHUNK):
            out.extend(pool.map(_eval, fens[start:start + CHUNK], chunksize=50))
            np.save(partial, np.asarray(out, np.float32))
            bar.update(min(CHUNK, len(fens) - start))
        bar.close()
    sf = np.asarray(out, np.float32)
    if limit:
        for k in ("planes", "y", "fens", "game_ids", "plies"):
            data[k] = data[k][:limit]
    data["sf_cp_white"] = sf
    np.savez_compressed(dst, **data)
    partial.unlink(missing_ok=True)
    print(f"wrote {dst} ({len(sf):,} positions; mean |cp| {np.abs(sf).mean():.0f}, mates/clipped {np.mean(np.abs(sf) >= 1000):.1%})")


def main() -> None:
    args = parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    path = find_stockfish(args.stockfish)
    print(f"Stockfish {path}, {args.nodes} nodes/position, {args.workers} workers")
    for name in ("val.npz", "train.npz"):                     # val first: small, quick sanity check
        if (dst / name).exists():
            print(f"{dst / name} exists, skipping")
            continue
        label_file(src / name, dst / name, path, args.nodes, args.workers, args.limit)
    for extra in ("split_summary.json",):
        if (src / extra).exists():
            shutil.copy(src / extra, dst / extra)


if __name__ == "__main__":
    main()
