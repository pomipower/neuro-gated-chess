"""Export tournament games to a self-contained replay page (demo/game_viewer.html) for presentations.

Reads data/results/<run_id>/games.csv and moves.csv (plus configs/gate_frozen.yaml for the gate thresholds),
replays every game with python-chess, and writes one HTML file with the data inlined: open it in any browser,
no server, GPU or internet needed (fonts fall back to system fonts offline).

Run from the repo root:
    python -m scripts.export_games --run-id tour_1                       # all games
    python -m scripts.export_games --run-id tour_1 --agents hybrid_s2_50 --max-games 12
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import chess
import pandas as pd

from src.utils import load_config, load_gate_config, resolve

TEMPLATE = Path(__file__).resolve().parents[1] / "demo" / "viewer_template.html"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run-id", required=True, help="tournament run id (comma-separated to merge several)")
    p.add_argument("--agents", default=None, help="only games involving at least one of these agents")
    p.add_argument("--max-games", type=int, default=None)
    p.add_argument("--out", default="demo/game_viewer.html")
    return p.parse_args()


def agent_tau(agent: str, gate_cfg: dict[str, Any] | None) -> float | None:
    """Gate threshold for hybrid agents (None for agents without a confidence gate)."""
    if not agent.startswith("hybrid_s2_") or not gate_cfg:
        return None
    pct = int(agent.rsplit("_", 1)[1])
    for k, v in gate_cfg["tau"].items():
        if int(round(float(k) * 100)) == pct:
            return float(v)
    return None


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def build_game(g: pd.Series, moves: pd.DataFrame, opening_moves: list[str], gate_cfg: dict[str, Any] | None,
               thoughts: dict[str, Any] | None = None, sacs: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    board = chess.Board()
    plies: list[dict[str, Any]] = []
    for u in opening_moves:
        mv = chess.Move.from_uci(u)
        white = board.turn == chess.WHITE
        san = board.san(mv)
        board.push(mv)
        plies.append({"uci": u, "san": san, "fen": board.fen(), "white": white, "kind": "book",
                      "agent": g.white if white else g.black, "ms": None})
    for m in moves.sort_values("ply").itertuples():
        if m.fen_before.split()[:4] != board.fen().split()[:4]:
            raise ValueError(f"{g.game_id}: move log out of sync at ply {m.ply}")
        mv = chess.Move.from_uci(m.move_uci)
        white = board.turn == chess.WHITE
        san = board.san(mv)
        board.push(mv)
        system = int(m.system)
        kind = {0: "sf", 1: "s1", 2: "s2"}[system]
        plies.append({
            "uci": m.move_uci, "san": san, "fen": board.fen(), "white": white, "kind": kind, "agent": m.agent,
            "ms": _num(m.time_elapsed_ms), "conf": _num(m.confidence), "tau": agent_tau(m.agent, gate_cfg),
            "sims": int(m.mcts_sims) if _num(m.mcts_sims) else 0,
            "depth": int(m.max_depth) if _num(m.max_depth) else 0,
            "thought": (thoughts or {}).get(str(int(m.ply))),
            "sac": (sacs or {}).get(int(m.ply)),
        })
    return {"id": g.game_id, "white": g.white, "black": g.black, "result": g.result,
            "termination": g.termination, "start_fen": chess.STARTING_FEN, "plies": plies}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    gate_cfg = load_gate_config()
    base = resolve(cfg["results_dir"])
    ids = [r.strip() for r in args.run_id.split(",")]
    games = pd.concat([pd.read_csv(base / r / "games.csv") for r in ids], ignore_index=True)
    moves = pd.concat([pd.read_csv(base / r / "moves.csv") for r in ids], ignore_index=True)
    openings = pd.read_csv(resolve(cfg["data"]["benchmark_dir"]) / "openings.csv").set_index("opening_id")
    if args.agents:
        wanted = {a.strip() for a in args.agents.split(",")}
        games = games[games.white.isin(wanted) | games.black.isin(wanted)]
    games = games[games.termination != "error"]
    if args.max_games:
        games = games.head(args.max_games)
    by_game = dict(tuple(moves.groupby("game_id")))
    explanations: dict[str, Any] = {}
    for r in ids:
        f = base / r / "explanations.json"
        if f.exists():
            explanations.update(json.loads(f.read_text()))
    sacrifices: dict[str, dict[int, dict[str, Any]]] = {}
    for r in ids:
        f = base / r / "sacrifices.csv"
        if f.exists():
            for row in pd.read_csv(f).itertuples():
                sacrifices.setdefault(row.game_id, {})[int(row.ply)] = {
                    "piece": row.piece, "loss": int(row.material_loss), "cpl": float(row.cpl), "sound": bool(row.sound)}
    if explanations:   # games with a recorded thought process come first
        games = games.assign(_x=~games.game_id.isin(explanations)).sort_values("_x", kind="stable").drop(columns="_x")
    out_games = []
    for g in games.itertuples():
        op = openings.loc[g.opening_id, "uci_moves"].split() if g.opening_id in openings.index else []
        try:
            gm = build_game(pd.Series(g._asdict()), by_game.get(g.game_id, moves.iloc[0:0]), op, gate_cfg,
                            explanations.get(g.game_id), sacrifices.get(g.game_id))
            gm["explained"] = g.game_id in explanations
            gm["sacs"] = sum(1 for p in gm["plies"] if p.get("sac"))
            out_games.append(gm)
        except ValueError as e:
            print(f"skipping: {e}")
    # gauge range: 98th percentile of observed confidences (+15%), never below the loosest gate threshold
    confs = sorted(p["conf"] for gm in out_games for p in gm["plies"] if p.get("conf") is not None)
    taus = [t for t in (agent_tau(f"hybrid_s2_{q}", gate_cfg) for q in (10, 25, 50, 75)) if t is not None]
    if confs:
        scale = min(1.0, max(confs[int(0.98 * (len(confs) - 1))] * 1.15, *taus, 0.05) * 1.05)
    else:
        scale = 1.0
    note = (f"{len(out_games)} games from tournament run(s) {', '.join(ids)}. Confidence = "
            f"{(gate_cfg or {}).get('confidence', 'n/a')} (T={(gate_cfg or {}).get('temperature', 'n/a')}); "
            "a hybrid searches when confidence is at or below its gate threshold. Thinking times are as measured "
            "during the tournament.")
    pieces_dir = TEMPLATE.parent / "pieces"
    pieces = {p.stem: p.read_text(encoding="utf-8") for p in sorted(pieces_dir.glob("*.svg"))} if pieces_dir.exists() else {}
    data = {"games": out_games, "conf_scale": scale, "note": note, "pieces": pieces}
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", json.dumps(data).replace("</", "<\\/"))
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(out_games)} games, {out.stat().st_size / 1e6:.1f} MB). Open it in a browser.")


if __name__ == "__main__":
    main()
