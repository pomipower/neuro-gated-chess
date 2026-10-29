"""Paired Elo differences (one shared opening-cluster bootstrap) + in-game search rate / time per agent."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from src.eval.elo import anchor_elos, fit_ratings
from src.utils import load_config, resolve

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/v3.yaml")
    p.add_argument("--run-id", required=True)
    p.add_argument("--pairs", required=True, help="a:b,c:d  (difference = a minus b)")
    p.add_argument("--n", type=int, default=1000)
    args = p.parse_args()
    cfg = load_config(args.config)
    root = resolve(cfg.get("results_dir", "data/results"))
    ids = [r.strip() for r in args.run_id.split(",")]
    games = pd.concat([pd.read_csv(root / r / "games.csv") for r in ids], ignore_index=True)
    moves = pd.concat([pd.read_csv(root / r / "moves.csv") for r in ids], ignore_index=True)
    anchors = anchor_elos(sorted(set(games.white) | set(games.black)))
    pairs = [tuple(x.split(":")) for x in args.pairs.split(",")]
    point = fit_ratings(games, anchors)
    rng = np.random.default_rng(cfg["seed"])
    openings = games.opening_id.unique()
    groups = {o: games[games.opening_id == o] for o in openings}
    diffs = {pr: [] for pr in pairs}
    for _ in range(args.n):
        pick = rng.choice(openings, size=len(openings), replace=True)
        r = fit_ratings(pd.concat([groups[o] for o in pick], ignore_index=True), anchors)
        for a, b in pairs:
            if a in r and b in r:
                diffs[(a, b)].append(r[a] - r[b])
    print(f"paired Elo differences ({args.n} opening-cluster resamples, {len(games)} games)")
    for a, b in pairs:
        d = np.array(diffs[(a, b)])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {a:>15} - {b:<15} {point[a] - point[b]:+7.1f}  95% CI [{lo:+.1f}, {hi:+.1f}]  "
              f"P(diff<=0)={(d <= 0).mean():.3f}  P(diff<=-50)={(d <= -50).mean():.3f}")
    m = moves[~moves.agent.isin(list(anchors))]
    t = m.groupby("agent").agg(moves=("ply", "size"), search_rate=("system", lambda s: (s == 2).mean()),
                               sec_per_move=("time_elapsed_ms", lambda s: s.mean() / 1000),
                               safety_override=("safety_override", "mean"))
    print("\nin-game behaviour\n", t.round(3).to_string())

if __name__ == "__main__":
    main()
