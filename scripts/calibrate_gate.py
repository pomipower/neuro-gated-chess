"""Choose the confidence function and thresholds on the val split (protocol section 2.4).

1. Laya scores every data/benchmark/positions_val.csv position; the judge labels Laya's move "good" (CPL <= 25).
2. Candidates: margin, top1_softmax(T), norm_entropy(T) for T in the grid. Highest AUROC wins; within 0.01 of
   the best, the simpler `margin` is preferred. 1/n_legal is reported as a trivial baseline (not selectable).
3. tau for each target S2 rate = that quantile of the chosen function's val confidences.
Writes configs/gate_frozen.yaml, and data/results/<run_id>/{gate_candidates.csv, laya_val.csv}.
Commit gate_frozen.yaml BEFORE running anything on the test split.

Run from the repo root:  python -m scripts.calibrate_gate --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt

import chess
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.engine.gating import margin, norm_entropy_conf, quantile_threshold, top1_softmax
from src.engine.LayaAgent import LayaAgent
from src.engine.Stockfish import StockfishJudge
from src.eval.metrics import auroc, ece
from src.utils import load_config, new_run_id, prepare_run_dir, resolve, seed_everything

T_GRID = (0.05, 0.1, 0.2, 0.5)
TIE_MARGIN = 0.01


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run-id", default=None)
    p.add_argument("--out", default="configs/gate_frozen.yaml")
    p.add_argument("--batch", type=int, default=8, help="positions per Laya call")
    p.add_argument("--limit", type=int, default=None, help="use only the first N val positions (debug)")
    return p.parse_args()


def candidates() -> list[tuple[str, float | None]]:
    out: list[tuple[str, float | None]] = [("margin", None)]
    out += [("top1_softmax", t) for t in T_GRID]
    out += [("norm_entropy", t) for t in T_GRID]
    return out


def conf_value(name: str, t: float | None, w: list[float]) -> float:
    if name == "margin":
        return margin(w)
    if name == "top1_softmax":
        return top1_softmax(w, t)
    return norm_entropy_conf(w, t)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    run_id = args.run_id or new_run_id("calib")
    out_dir = prepare_run_dir(cfg, run_id)
    pos = pd.read_csv(resolve(cfg["data"]["benchmark_dir"]) / "positions_val.csv")
    if args.limit:
        pos = pos.head(args.limit)
    laya = LayaAgent.from_config(cfg)
    judge = StockfishJudge.from_config(cfg)
    good_cpl = cfg["judge"]["good_move_cpl"]

    rows = []
    for i in tqdm(range(0, len(pos), args.batch), desc="laya val"):
        chunk = pos.iloc[i:i + args.batch]
        boards = [chess.Board(f) for f in chunk.fen]
        for (_, r), board, scores in zip(chunk.iterrows(), boards, laya.score_many(boards)):
            best = max(scores, key=lambda m: (scores[m], m.uci()))
            cpl = judge.cpl(board, best)
            w = list(scores.values())
            row = {"pos_id": r.pos_id, "phase": r.phase, "n_legal": len(w), "laya_move": best.uci(),
                   "cpl": cpl, "good": cpl <= good_cpl, "inv_n_legal": 1.0 / len(w)}
            for name, t in candidates():
                row[f"{name}" + (f"@{t}" if t else "")] = conf_value(name, t, w)
            rows.append(row)
    judge.close()
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "laya_val.csv", index=False)

    cand_rows = []
    for name, t in candidates():
        col = name + (f"@{t}" if t else "")
        cand_rows.append({"function": name, "temperature": t, "auroc": auroc(df[col], df.good),
                          "ece": ece(df[col], df.good)})
    cand_rows.append({"function": "inv_n_legal (baseline)", "temperature": None,
                      "auroc": auroc(df.inv_n_legal, df.good), "ece": float("nan")})
    cand = pd.DataFrame(cand_rows)
    cand.to_csv(out_dir / "gate_candidates.csv", index=False)
    print(cand.to_string(index=False))

    selectable = cand[~cand.function.str.contains("baseline")].dropna(subset=["auroc"])
    if selectable.empty:
        raise SystemExit("all AUROCs are NaN (Laya's moves were all good or all bad on val): enlarge n_val")
    best = selectable.sort_values("auroc", ascending=False).iloc[0]
    margin_row = selectable[selectable.function == "margin"].iloc[0]
    chosen = margin_row if best.auroc - margin_row.auroc <= TIE_MARGIN else best
    t = None if pd.isna(chosen.temperature) else float(chosen.temperature)
    col = chosen.function + (f"@{t}" if t else "")
    taus = {float(q): quantile_threshold(df[col].tolist(), q) for q in cfg["gate"]["s2_rates"]}
    frozen = {
        "confidence": "norm_entropy" if chosen.function == "norm_entropy" else str(chosen.function),
        "temperature": t if t is not None else cfg["laya"]["temperature"],
        "tau": {f"{q:.2f}": float(v) for q, v in taus.items()},
        "val_auroc": float(chosen.auroc),
        "val_n": int(len(df)),
        "val_laya_acpl": float(df.cpl.mean()),
        "val_good_rate": float(df.good.mean()),
        "calibration_run": run_id,
        "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
        "laya_model": cfg["laya"]["model_name"],
        "laya_revision": cfg["laya"].get("revision"),
    }
    out = resolve(args.out)
    with open(out, "w") as f:
        f.write("# Written by scripts/calibrate_gate.py. Commit this file BEFORE any test-split run.\n")
        yaml.safe_dump(frozen, f, sort_keys=False)
    print(f"chosen: {frozen['confidence']} (T={frozen['temperature']}), AUROC {frozen['val_auroc']:.3f}")
    print(f"tau per S2 rate: {frozen['tau']}")
    achieved = {q: float(np.mean(df[col] <= v)) for q, v in taus.items()}
    print(f"achieved S2 rate on val: {achieved}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
