"""Recompute the bot's "thought process" for selected tournament games (for the replay viewer).

The tournament logs only each move's outcome. For a handful of demo games this script re-runs the same agent
(same config snapshot as the tournament run, same frozen gate) on every position our bot faced and records:
    - Laya's shortlist: top candidate moves with their win chances, and its confidence
    - for searched moves: how the search split its simulations between moves, each move's value, and the
      principal variation (the line the search expects)
    - whether the final move overrode Laya's favourite
Laya and the search are deterministic, so the recomputed decision should match the logged one; every move is
flagged `matches` so a mismatch (e.g. fp16 GPU noise) is visible rather than hidden.

Writes data/results/<run_id>/explanations.json, which scripts/export_games.py picks up automatically.
Takes ~1-2 minutes per game on an RTX 4060. Never run it while an experiment is using the GPU.

Run from the repo root:
    python -m scripts.explain_game --run-id tour_1 --agents hybrid_s2_50 --max-games 4
    python -m scripts.explain_game --run-id tour_1 --games "o03__hybrid_s2_50__sf_elo_1320"
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import chess
import pandas as pd
import yaml
from tqdm import tqdm

from src.engine.factory import Resources, build_agent
from src.utils import load_config, load_gate_config, resolve

TOP_K = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run-id", required=True)
    p.add_argument("--games", default=None, help="comma-separated game ids")
    p.add_argument("--agents", default=None, help="games involving these agents (comma-separated)")
    p.add_argument("--max-games", type=int, default=4)
    return p.parse_args()


def san_list(board: chess.Board, moves: list[chess.Move]) -> list[str]:
    b, out = board.copy(stack=False), []
    for mv in moves:
        if mv not in b.legal_moves:
            break
        out.append(b.san(mv))
        b.push(mv)
    return out


def explain_move(agent: Any, board: chess.Board, logged: chess.Move, logged_system: int) -> dict[str, Any]:
    thought: dict[str, Any] = {}
    out = None
    if agent.laya is not None:
        out = agent.laya.evaluate(board)
        top = sorted(out.scores.items(), key=lambda kv: (-kv[1], kv[0].uci()))[:TOP_K]
        thought["laya"] = [[m.uci(), board.san(m), round(w, 4)] for m, w in top]
        thought["laya_best"] = out.best_move.uci()
        thought["conf"] = round(out.confidence, 5)
    chosen = out.best_move if out is not None else None
    if logged_system == 2 and agent.search is not None:
        res = agent.search.search(board, root_scores=out.scores if (out is not None and agent.use_laya_prior) else None)
        total = max(1, sum(res.visits.values()))
        top = sorted(res.visits.items(), key=lambda kv: (-kv[1], -res.q.get(kv[0], -9)))[:TOP_K]
        thought["search"] = {
            "visits": [[m.uci(), board.san(m), n, round(n / total, 4), round(res.q.get(m, float("nan")), 4)]
                       for m, n in top if n > 0],
            "pv": san_list(board, res.pv),
            "sims": res.sims,
        }
        chosen = res.move
    thought["chosen"] = chosen.uci() if chosen is not None else None
    thought["matches"] = chosen == logged
    thought["override"] = bool(out is not None and logged != out.best_move)
    return thought


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    run_dir = resolve(base_cfg["results_dir"]) / args.run_id
    snapshot = run_dir / "config.yaml"
    cfg = yaml.safe_load(snapshot.read_text()) if snapshot.exists() else base_cfg
    games = pd.read_csv(run_dir / "games.csv")
    moves = pd.read_csv(run_dir / "moves.csv")
    openings = pd.read_csv(resolve(cfg["data"]["benchmark_dir"]) / "openings.csv").set_index("opening_id")
    if args.games:
        games = games[games.game_id.isin([g.strip() for g in args.games.split(",")])]
    elif args.agents:
        wanted = {a.strip() for a in args.agents.split(",")}
        games = games[games.white.isin(wanted) | games.black.isin(wanted)]
    games = games[games.termination != "error"].head(args.max_games)
    if games.empty:
        raise SystemExit("no matching games")

    res = Resources(cfg, load_gate_config())
    agents: dict[str, Any] = {}
    out_path = run_dir / "explanations.json"
    explanations: dict[str, Any] = json.loads(out_path.read_text()) if out_path.exists() else {}
    mismatches = total = 0
    for g in games.itertuples():
        board = chess.Board()
        for u in openings.loc[g.opening_id, "uci_moves"].split():
            board.push_uci(u)
        rows = moves[moves.game_id == g.game_id].sort_values("ply")
        per_ply: dict[str, Any] = {}
        for m in tqdm(rows.itertuples(), total=len(rows), desc=g.game_id):
            if int(m.system) in (1, 2):
                if m.agent not in agents:
                    agents[m.agent] = build_agent(m.agent, res, seed=0)
                th = explain_move(agents[m.agent], board, chess.Move.from_uci(m.move_uci), int(m.system))
                per_ply[str(int(m.ply))] = th
                total += 1
                mismatches += int(not th["matches"])
            board.push_uci(m.move_uci)
        explanations[g.game_id] = per_ply
        out_path.write_text(json.dumps(explanations))
    res.close()
    print(f"explained {len(games)} games ({total} bot moves, {mismatches} recomputed differently) -> {out_path}")
    if mismatches:
        print("Mismatches are flagged per move in the viewer; small GPU fp16 differences can change near-ties.")


if __name__ == "__main__":
    main()
