"""Study B: play pairings with fixed openings and swapped colors (protocol section 3).

Tier 1 = each of tournament.tier1_agents vs each anchor (sf_elo_N); tier 2 = round-robin among
tier1_agents; tier 3 = hybrid_s2_25 and hybrid_s2_75 vs the anchors. --agents a,b,c with --vs x,y overrides.
Each opening in data/benchmark/openings.csv is played twice, colors swapped. Termination:
board.outcome(claim_draw=True); max_plies -> draw; illegal move or exception -> loss for that agent.

Appends to data/results/<run_id>/games.csv and moves.csv. Resumable: skips (opening_id, white, black)
already in games.csv. random_s2_* agents get a per-game seed derived from the game id.
--judge-moves afterwards adds Stockfish CPL to every logged move (moves_judged.csv) for blunder analysis.
--pilot plays 2 openings per pairing and prints seconds per game, to size the real run (protocol §3).

Run from the repo root:  python -m scripts.run_tournament --config configs/default.yaml --tier 1 --run-id tour_1
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import time
import traceback
import zlib
from pathlib import Path
from typing import Any

import chess
import pandas as pd
from tqdm import tqdm

from src.engine.factory import Resources, build_agent
from src.utils import load_config, load_gate_config, new_run_id, prepare_run_dir, resolve, seed_everything

GAME_COLS = ["run_id", "game_id", "opening_id", "white", "black", "result", "plies", "termination", "seconds",
             "white_error", "black_error"]
MOVE_COLS = ["game_id", "ply", "agent", "move_uci", "fen_before", "system", "confidence", "time_elapsed_ms",
             "laya_ms", "search_ms", "laya_calls", "laya_moves_scored", "cnn_evals", "mcts_sims", "max_depth",
             "mean_leaf_depth"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--tier", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--run-id", default=None)
    p.add_argument("--agents", default=None, help="comma list; overrides the tier's agents")
    p.add_argument("--vs", default=None, help="comma list of opponents for --agents (default: anchors)")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--judge-moves", action="store_true", help="judge moves.csv of --run-id and exit")
    return p.parse_args()


def pairings(cfg: dict[str, Any], tier: int, agents: str | None, vs: str | None) -> list[tuple[str, str]]:
    t = cfg["tournament"]
    anchors = [f"sf_elo_{e}" for e in t["anchors"]["elos"]]
    if agents:
        a_list = [a.strip() for a in agents.split(",") if a.strip()]
        opp = [v.strip() for v in vs.split(",")] if vs else anchors
        return [(a, o) for a in a_list for o in opp if a != o]
    if tier == 1:
        return [(a, o) for a in t["tier1_agents"] for o in anchors]
    if tier == 2:
        return list(itertools.combinations(t["tier1_agents"], 2))
    return [(a, o) for a in ("hybrid_s2_25", "hybrid_s2_75") for o in anchors]


def stable_seed(text: str) -> int:
    return zlib.crc32(text.encode())


def play_game(white: Any, black: Any, opening: list[str], max_plies: int, game_id: str,
              move_rows: list[dict[str, Any]]) -> dict[str, Any]:
    board = chess.Board()
    for u in opening:
        board.push_uci(u)
    for agent in (white, black):
        agent.new_game()
    errors = {chess.WHITE: "", chess.BLACK: ""}
    t0 = time.time()
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            result, termination = outcome.result(), outcome.termination.name.lower()
            break
        if board.ply() >= max_plies:
            result, termination = "1/2-1/2", "max_plies"
            break
        mover = white if board.turn == chess.WHITE else black
        fen = board.fen()
        try:
            move, info = mover.select_move(board.copy())
            if move not in board.legal_moves:
                raise ValueError(f"illegal move {move} in {fen}")
        except Exception as e:  # protocol: an error or illegal move loses the game
            errors[board.turn] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            termination = "error"
            break
        move_rows.append({"game_id": game_id, "ply": board.ply(), "agent": mover.name, "move_uci": move.uci(),
                          "fen_before": fen, **{k: v for k, v in info.as_dict().items() if k in MOVE_COLS}})
        board.push(move)
    return {"result": result, "plies": board.ply(), "termination": termination,
            "seconds": round(time.time() - t0, 1), "white_error": errors[chess.WHITE],
            "black_error": errors[chess.BLACK]}


def judge_moves(cfg: dict[str, Any], out_dir: Path) -> None:
    from src.engine.Stockfish import StockfishJudge

    mv = pd.read_csv(out_dir / "moves.csv")
    with StockfishJudge.from_config(cfg) as judge:
        cpl = [judge.cpl(chess.Board(f), chess.Move.from_uci(u))
               for f, u in tqdm(zip(mv.fen_before, mv.move_uci), total=len(mv), desc="judging moves")]
    mv["cpl"] = cpl
    mv["blunder"] = mv.cpl >= cfg["judge"]["blunder_cpl"]
    mv.to_csv(out_dir / "moves_judged.csv", index=False)
    print(mv.groupby("agent")[["cpl", "blunder"]].mean().round(3))


def append_rows(path: Path, cols: list[str], rows: list[dict[str, Any]]) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if new:
            wr.writeheader()
        for r in rows:
            wr.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()})


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    run_id = args.run_id or new_run_id(f"tour_t{args.tier}")
    out_dir = prepare_run_dir(cfg, run_id)
    if args.judge_moves:
        judge_moves(cfg, out_dir)
        return
    openings = pd.read_csv(resolve(cfg["data"]["benchmark_dir"]) / "openings.csv")
    if args.pilot:
        openings = openings.head(2)
    pairs = pairings(cfg, args.tier, args.agents, args.vs)
    games_path, moves_path = out_dir / "games.csv", out_dir / "moves.csv"
    done: set[tuple[str, str, str]] = set()
    if games_path.exists():
        g = pd.read_csv(games_path)
        done = set(zip(g.opening_id, g.white, g.black))
    schedule = [(o.opening_id, o.uci_moves.split(), w, b)
                for a, c in pairs for o in openings.itertuples() for w, b in ((a, c), (c, a))
                if (o.opening_id, w, b) not in done]
    print(f"{len(pairs)} pairings x {len(openings)} openings x 2 colors; {len(schedule)} games to play")
    res = Resources(cfg, load_gate_config())
    cache: dict[str, Any] = {}

    def agent(name: str, game_id: str) -> Any:
        if name.startswith("random_s2_"):
            return build_agent(name, res, seed=stable_seed(game_id))
        if name not in cache:
            cache[name] = build_agent(name, res)
        return cache[name]

    secs = []
    for oid, moves, w, b in tqdm(schedule, desc="games"):
        gid = f"{oid}__{w}__{b}"
        move_rows: list[dict[str, Any]] = []
        row = play_game(agent(w, gid), agent(b, gid), moves, cfg["tournament"]["max_plies"], gid, move_rows)
        secs.append(row["seconds"])
        append_rows(moves_path, MOVE_COLS, move_rows)            # moves first: a crash never leaves a game
        append_rows(games_path, GAME_COLS, [{"run_id": run_id, "game_id": gid, "opening_id": oid,  # without moves
                                             "white": w, "black": b, **row}])
    res.close()
    if secs:
        mean = sum(secs) / len(secs)
        print(f"mean {mean:.1f} s/game over {len(secs)} games")
        if args.pilot:
            full = len(pairs) * cfg["tournament"]["n_openings"] * 2
            print(f"pilot estimate for the full tier: {full} games x {mean:.0f} s = {full * mean / 3600:.1f} h "
                  "(record this in docs/DEVIATIONS.md)")


if __name__ == "__main__":
    main()
