"""v3: compute learned-gate features for every position of a benchmark split (Laya on GPU, ~0.4 s/position).

Writes data/results/<run_id>/gate_features.csv (pos_id + src.engine.gate_features.FEATURES). Resumable.
Run from the repo root:
    python -m scripts.gate_features --config configs/v2.yaml --split val --run-id v3_val
"""

from __future__ import annotations

import argparse
import csv

import chess
import pandas as pd
from tqdm import tqdm

from src.engine.factory import Resources
from src.engine.gate_features import FEATURES, gate_features
from src.utils import load_config, load_gate_config, prepare_run_dir, resolve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/v2.yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--run-id", required=True)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = prepare_run_dir(cfg, args.run_id)
    out = out_dir / "gate_features.csv"
    pos = pd.read_csv(resolve(cfg["data"]["benchmark_dir"]) / f"positions_{args.split}.csv")
    if args.limit:
        pos = pos.head(args.limit)
    done = set(pd.read_csv(out).pos_id) if out.exists() else set()
    res = Resources(cfg, load_gate_config())
    v2 = cfg.get("v2", {})
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pos_id", *FEATURES])
        if new:
            w.writeheader()
        for r in tqdm(pos.itertuples(), total=len(pos), desc=f"gate features {args.split}"):
            if r.pos_id in done:
                continue
            board = chess.Board(r.fen)
            o = res.laya.evaluate(board)
            f = gate_features(board, o.scores, o.confidence, v2.get("safety", {}), v2.get("few_legal_moves", 12),
                              v2.get("endgame_npm", 13))
            w.writerow({"pos_id": r.pos_id, **{k: round(v, 6) for k, v in f.items()}})
            fh.flush()
    res.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
