"""Study C: tactical puzzles by rating bucket (protocol section 4).

Lichess puzzle convention: FEN is the position BEFORE the opponent's move; Moves[0] is that move and
Moves[1] is the solution's first move. The agent sees the position after Moves[0]; it solves the puzzle if it
plays Moves[1], or any move that gives checkmate.

Writes data/results/<run_id>/puzzles_results.csv
(puzzle_id, rating, bucket, themes, agent, move_uci, solved, system, confidence, time_elapsed_ms).

Run from the repo root:  python -m scripts.run_puzzles --config configs/default.yaml --run-id puzzles_1
"""

from __future__ import annotations

import argparse
import csv
import math

import chess
import pandas as pd
from tqdm import tqdm

from src.engine.factory import Resources, build_agent
from src.utils import load_config, load_gate_config, new_run_id, prepare_run_dir, resolve, seed_everything

COLS = ["puzzle_id", "rating", "bucket", "themes", "agent", "move_uci", "solved", "system", "confidence",
        "time_elapsed_ms"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run-id", default=None)
    p.add_argument("--agents", default="laya_only,hybrid_s2_50,always_s2,mcts_uniform")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def puzzle_position(fen: str, moves: str) -> tuple[chess.Board, chess.Move]:
    seq = moves.split()
    board = chess.Board(fen)
    board.push_uci(seq[0])
    return board, chess.Move.from_uci(seq[1])


def solved(board: chess.Board, move: chess.Move, solution: chess.Move) -> bool:
    if move == solution:
        return True
    after = board.copy(stack=False)
    after.push(move)
    return after.is_checkmate()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    run_id = args.run_id or new_run_id("puzzles")
    out_dir = prepare_run_dir(cfg, run_id)
    pz = pd.read_csv(resolve(cfg["data"]["benchmark_dir"]) / "puzzles.csv")
    if args.limit:
        pz = pz.head(args.limit)
    res = Resources(cfg, load_gate_config())
    names = [a.strip() for a in args.agents.split(",") if a.strip()]
    agents = {n: build_agent(n, res, seed=cfg["seed"]) for n in names}
    out = out_dir / "puzzles_results.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in tqdm(pz.itertuples(), total=len(pz), desc="puzzles"):
            board, sol = puzzle_position(r.fen, r.moves)
            for n, agent in agents.items():
                move, info = agent.select_move(board.copy())
                w.writerow({"puzzle_id": r.puzzle_id, "rating": r.rating, "bucket": r.bucket, "themes": r.themes,
                            "agent": n, "move_uci": move.uci(), "solved": solved(board, move, sol),
                            "system": info.system,
                            "confidence": "" if math.isnan(info.confidence) else round(info.confidence, 6),
                            "time_elapsed_ms": round(info.time_elapsed_ms, 2)})
    df = pd.read_csv(out)
    print(df.pivot_table(index="bucket", columns="agent", values="solved", aggfunc="mean").round(3))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
