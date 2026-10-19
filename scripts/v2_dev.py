"""v2 development check on VALIDATION positions only (never the fresh test set).

Runs the given agents on the first N validation positions, judges every move with the Stockfish judge and
prints ACPL, blunder rate, search rate, safety-override rate and seconds per move. Use it to sanity-check the
v2 agent and to choose between a few pre-listed settings BEFORE freezing (docs/V2_PROTOCOL.md section 3).

Run from the repo root:
    python -m scripts.v2_dev --config configs/v2.yaml --n 150 --agents laya_only,v2_safe,v2_full
    python -m scripts.v2_dev --config configs/v2.yaml --n 150 --agents v2_full --set v2.prior_temperature=0.1
"""

from __future__ import annotations

import argparse
import time

import chess
import pandas as pd
import yaml
from tqdm import tqdm

from src.engine.factory import Resources, build_agent
from src.engine.Stockfish import StockfishJudge
from src.utils import load_config, load_gate_config, resolve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/v2.yaml")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--agents", default="laya_only,v2_safe,v2_full")
    p.add_argument("--set", action="append", default=[], help="override, e.g. v2.prior_temperature=0.1")
    return p.parse_args()


def apply_overrides(cfg: dict, items: list[str]) -> None:
    for item in items:
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = yaml.safe_load(val)
        print(f"override {key} = {node[parts[-1]]!r}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    pos = pd.read_csv(resolve(cfg["data"]["benchmark_dir"]) / "positions_val.csv").head(args.n)
    res = Resources(cfg, load_gate_config())
    names = [a.strip() for a in args.agents.split(",")]
    agents = {a: build_agent(a, res) for a in names}
    rows = []
    with StockfishJudge.from_config(cfg) as judge:
        for r in tqdm(pos.itertuples(), total=len(pos), desc="val positions"):
            board = chess.Board(r.fen)
            for a in names:
                t = time.perf_counter()
                move, info = agents[a].select_move(board.copy())
                ms = (time.perf_counter() - t) * 1000
                rows.append({"agent": a, "phase": r.phase, "cpl": judge.cpl(board, move), "ms": ms,
                             "system": info.system, "override": getattr(info, "safety_override", 0),
                             "critical": getattr(info, "critical", 0)})
    res.close()
    df = pd.DataFrame(rows)
    df["blunder"] = df.cpl >= cfg["judge"]["blunder_cpl"]
    t = df.groupby("agent").agg(acpl=("cpl", "mean"), blunder_rate=("blunder", "mean"), sec_per_move=("ms", "mean"),
                                search_rate=("system", lambda s: (s == 2).mean()),
                                override_rate=("override", "mean"), critical_rate=("critical", "mean"))
    t["sec_per_move"] /= 1000
    print(f"\n{len(pos)} validation positions\n", t.round(3).to_string())
    print("\nby phase (ACPL):\n", df.pivot_table(index="agent", columns="phase", values="cpl").round(1).to_string())


if __name__ == "__main__":
    main()
