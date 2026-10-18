"""Pilot timing: measure per-move cost of each component on this machine and estimate run times.

Runs laya_only, always_s2 and mcts_uniform (timed, Laya cache off) on the first --n val positions, prints
mean/median ms per move and an estimate of the hours each study needs with the current config. Run this
FIRST on the machine that will do the real runs, and copy the output into docs/DEVIATIONS.md.

Run from the repo root:  python -m scripts.pilot_timing --config configs/default.yaml --n 20
"""

from __future__ import annotations

import argparse
import time

import chess
import pandas as pd

from src.engine.factory import Resources, build_agent
from src.utils import env_snapshot, load_config, load_gate_config, resolve, seed_everything

AVG_PLIES_PER_GAME = 100       # assumption for the estimate; replace with the pilot tournament's number


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--split", default="val")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    env = env_snapshot(cfg)
    print({k: env[k] for k in ("platform", "torch", "cuda", "stockfish")})
    pos = pd.read_csv(resolve(cfg["data"]["benchmark_dir"]) / f"positions_{args.split}.csv").head(args.n)
    res = Resources(cfg, load_gate_config())
    agents = {n: build_agent(n, res) for n in ("laya_only", "always_s2", "mcts_uniform")}
    warm = chess.Board(pos.fen.iloc[0])
    for a in agents.values():                     # warm-up (CUDA kernels, model load)
        a.select_move(warm)
    rows = []
    for fen in pos.fen:
        board = chess.Board(fen)
        for n, a in agents.items():
            _, info = a.select_move(board)
            rows.append({"agent": n, "ms": info.time_elapsed_ms, "laya_ms": info.laya_ms,
                         "search_ms": info.search_ms, "n_legal": board.legal_moves.count()})
    df = pd.DataFrame(rows)
    summary = df.groupby("agent")[["ms", "laya_ms", "search_ms"]].agg(["mean", "median"]).round(1)
    print(summary)
    laya_ms = df[df.agent == "laya_only"].ms.mean()
    s2_ms = df[df.agent == "always_s2"].search_ms.mean()
    unif_ms = df[df.agent == "mcts_uniform"].ms.mean()
    per_pos = (laya_ms + s2_ms + unif_ms) / 1000.0
    n_test, n_val = cfg["benchmark"]["n_test"], cfg["benchmark"]["n_val"]
    print(f"\nLaya: {laya_ms:.0f} ms/move ({1000 * df[df.agent == 'laya_only'].n_legal.mean() / laya_ms:.0f} "
          f"move scores/s); search ({cfg['mcts']['simulations']} sims): {s2_ms:.0f} ms/move")
    print(f"Study A test ({n_test} positions, 3 agents, judge excluded): {n_test * per_pos / 3600:.1f} h")
    print(f"gate calibration ({n_val} val positions, Laya only): {n_val * laya_ms / 1000 / 3600:.1f} h")
    t = cfg["tournament"]
    games = len(t["tier1_agents"]) * len(t["anchors"]["elos"]) * t["n_openings"] * 2
    our_moves = games * AVG_PLIES_PER_GAME / 2
    mean_agent_ms = (laya_ms + (laya_ms + s2_ms) + (laya_ms + 0.5 * s2_ms) + unif_ms) / 4
    print(f"tier-1 tournament ({games} games, ~{AVG_PLIES_PER_GAME} plies/game assumed): "
          f"{our_moves * mean_agent_ms / 1000 / 3600:.1f} h + Stockfish moves")
    print("Copy these numbers into docs/DEVIATIONS.md, then size the runs (protocol §3).")


if __name__ == "__main__":
    main()
