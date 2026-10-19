"""Study A: run laya_only, always_s2, mcts_uniform on every benchmark position and judge each move.

Writes data/results/<run_id>/benchmark_moves.csv (columns below). Hybrids and random-gate controls are
composed offline from this file (src/eval/compose.py, protocol section 2.3). always_s2 reuses the laya_only
evaluation of the same position as its root prior (so Laya runs once per position); its wall_ms is
laya_ms + search_ms.

Resumable (--resume skips positions already complete) and shardable (--shard i/n, e.g. 0/2 and 1/2 in two
processes; each shard writes its own file, merge them with --merge).

Run from the repo root:  python -m scripts.run_position_benchmark --config configs/default.yaml --split test
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import chess
import pandas as pd
from tqdm import tqdm

from src.engine.factory import Resources, build_agent
from src.engine.Stockfish import StockfishJudge
from src.utils import load_config, load_gate_config, new_run_id, prepare_run_dir, resolve, seed_everything

COLUMNS = ["run_id", "split", "pos_id", "fen", "ply", "phase", "n_legal", "sf_best_uci", "sf_score_cp",
           "sf_gap_cp", "agent", "move_uci", "system", "confidence", "cpl", "is_best", "wall_ms", "laya_ms",
           "search_ms", "laya_calls", "laya_moves_scored", "cnn_evals", "mcts_sims", "max_depth",
           "mean_leaf_depth", "critical", "safety_override", "special"]
DEFAULT_AGENTS = ("laya_only", "always_s2", "mcts_uniform")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--run-id", default=None)
    p.add_argument("--shard", default="0/1", help="i/n")
    p.add_argument("--agents", default=",".join(DEFAULT_AGENTS))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--merge", action="store_true", help="merge shard files of --run-id and exit")
    return p.parse_args()


def merge(out_dir: Path) -> None:
    parts = sorted(out_dir.glob("benchmark_moves.shard*.csv"))
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).drop_duplicates(["pos_id", "agent"])
    df.sort_values(["pos_id", "agent"]).to_csv(out_dir / "benchmark_moves.csv", index=False)
    print(f"merged {len(parts)} shards -> {len(df)} rows")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    if args.merge:
        if not args.run_id:
            raise SystemExit("--merge needs --run-id")
        merge(resolve(cfg["results_dir"]) / args.run_id)
        return
    gate_cfg = load_gate_config()
    if gate_cfg is None and args.split == "test":
        print("WARNING: configs/gate_frozen.yaml missing; confidence uses the config default. "
              "Protocol: calibrate on val first.")
    run_id = args.run_id or new_run_id(f"bench_{args.split}")
    out_dir = prepare_run_dir(cfg, run_id)
    k, n = (int(x) for x in args.shard.split("/"))
    out_path = out_dir / ("benchmark_moves.csv" if n == 1 else f"benchmark_moves.shard{k}of{n}.csv")

    pos = pd.read_csv(resolve(cfg["data"]["benchmark_dir"]) / f"positions_{args.split}.csv")
    pos = pos.iloc[k::n]
    if args.limit:
        pos = pos.head(args.limit)
    agents_wanted = [a.strip() for a in args.agents.split(",") if a.strip()]
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        if not args.resume:
            raise SystemExit(f"{out_path} exists: pass --resume or a new --run-id")
        prev = pd.read_csv(out_path)
        done = set(zip(prev.pos_id, prev.agent))

    res = Resources(cfg, gate_cfg)
    agents = {a: build_agent(a, res) for a in agents_wanted}
    judge = StockfishJudge.from_config(cfg)
    new_file = not out_path.exists()
    cols = COLUMNS
    if not new_file:                                   # resume: keep the existing header (v1 files lack v2 columns)
        with open(out_path, newline="") as fh:
            cols = next(csv.reader(fh), COLUMNS)
    with open(out_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for _, r in tqdm(pos.iterrows(), total=len(pos), desc=f"Study A {args.split} shard {k}/{n}"):
            todo = [a for a in agents_wanted if (r.pos_id, a) not in done]
            if not todo:
                continue
            board = chess.Board(r.fen)
            sf = judge.analyse(board)
            base = {"run_id": run_id, "split": args.split, "pos_id": r.pos_id, "fen": r.fen, "ply": r.ply,
                    "phase": r.phase, "n_legal": r.n_legal, "sf_best_uci": sf.best_move.uci(),
                    "sf_score_cp": sf.score_cp, "sf_gap_cp": sf.gap_cp}
            laya_out = None
            for name in todo:
                agent = agents[name]
                if (name == "always_s2" or name.startswith("v2_")) and laya_out is not None:
                    move, info, _, _ = agent.decide(board, laya_out=laya_out)    # reuse this position's Laya call
                else:
                    move, info, out, _ = agent.decide(board)
                    if name == "laya_only":
                        laya_out = out
                cpl = judge.cpl(board, move)
                writer.writerow(base | {
                    "agent": name, "move_uci": move.uci(), "system": info.system,
                    "confidence": "" if math.isnan(info.confidence) else round(info.confidence, 6),
                    "cpl": cpl, "is_best": move == sf.best_move, "wall_ms": round(info.time_elapsed_ms, 2),
                    "laya_ms": round(info.laya_ms, 2), "search_ms": round(info.search_ms, 2),
                    "laya_calls": info.laya_calls, "laya_moves_scored": info.laya_moves_scored,
                    "cnn_evals": info.cnn_evals, "mcts_sims": info.mcts_sims, "max_depth": info.max_depth,
                    "mean_leaf_depth": round(info.mean_leaf_depth, 3),
                    "critical": getattr(info, "critical", ""), "safety_override": getattr(info, "safety_override", ""),
                    "special": getattr(info, "special", "")})
            fh.flush()
    judge.close()
    res.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
