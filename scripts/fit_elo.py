"""Fit anchored Elo with cluster-bootstrap CIs from games.csv (protocol section 3).

Reads data/results/<run_id>/games.csv (several run ids may be given, comma-separated, to pool tiers),
writes data/results/<first run_id>/ratings.csv (agent, elo, ci_low, ci_high, games, score, anchor, off_scale).
Games that ended in an agent error are kept (they count as losses, per protocol) and reported.

Run from the repo root:  python -m scripts.fit_elo --config configs/default.yaml --run-id tour_1
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.eval.elo import summarize
from src.utils import load_config, resolve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run-id", required=True, help="one id or a comma-separated list")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ids = [r.strip() for r in args.run_id.split(",")]
    base = resolve(cfg["results_dir"])
    games = pd.concat([pd.read_csv(base / r / "games.csv") for r in ids], ignore_index=True)
    errors = games[games.termination == "error"]
    if len(errors):
        print(f"{len(errors)} games ended in an agent error (counted as losses):")
        print(errors[["game_id", "white_error", "black_error"]].to_string(index=False))
    table = summarize(cfg, games)
    out = base / ids[0] / "ratings.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
