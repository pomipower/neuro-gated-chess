"""Build splits and benchmark files (docs/SPEC.md, `scripts/prepare_data.py`).

data/raw/*{.pgn,.pgn.zst,.zip} (Lichess Elite months; YYYY-MM must appear in each file name) ->
  data/processed/train.npz, val.npz        value-net data (packed planes, white-perspective labels)
  data/benchmark/positions_val.csv          gate-calibration positions (val month)
  data/benchmark/positions_test.csv         Study A positions (test month)
  data/benchmark/openings.csv               Study B openings (test month; needs the Stockfish judge)
  data/benchmark/puzzles.csv                Study C sample (from data/raw/lichess_db_puzzle.csv[.zst])
  data/benchmark/split_summary.json         which months went where, counts
Deterministic given cfg.seed. Positions in val/test that also occur in train (same FEN) are dropped.
If data.train_months / val_month / test_month are empty, the most recent month is test, the one before
is val, and the rest are train (recorded in split_summary.json; copy it into docs/DEVIATIONS.md).

Run from the repo root:  python -m scripts.prepare_data --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import chess
import numpy as np
from tqdm import tqdm

from src.eval.metrics import game_phase
from src.model.dataset import (
    PositionSet, build_training_set, fen_key, iter_games, keep_game, mainline_boards, month_of,
)
from src.utils import load_config, resolve, seed_everything

PGN_SUFFIXES = (".pgn", ".pgn.zst", ".zip")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--skip-train", action="store_true", help="do not rebuild train.npz / val.npz")
    p.add_argument("--skip-openings", action="store_true")
    p.add_argument("--skip-puzzles", action="store_true")
    p.add_argument("--val-sf-nodes", type=int, default=0,
                   help="Stockfish nodes for val-set evals (Spearman check, protocol §5); 0 = skip")
    p.add_argument("--sf-labels", action="store_true",
                   help="also compute Stockfish evals for train (needed for label.mode=blend; slow)")
    return p.parse_args()


def discover(raw: Path) -> dict[str, list[Path]]:
    by_month: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(raw.iterdir()):
        name = f.name.lower()
        if "puzzle" in name or not name.endswith(PGN_SUFFIXES):
            continue
        m = month_of(f)
        if m is None:
            print(f"  skipping {f.name}: no YYYY-MM in file name")
            continue
        by_month[m].append(f)
    return dict(by_month)


def assign_months(cfg: dict[str, Any], months: list[str]) -> tuple[list[str], str, str]:
    d = cfg["data"]
    if d.get("test_month") and d.get("val_month"):
        train = d.get("train_months") or [m for m in months if m not in (d["val_month"], d["test_month"])]
        roles = (list(train), d["val_month"], d["test_month"])
    else:
        if len(months) < 3:
            raise SystemExit(f"need >= 3 distinct months in data/raw (found {months}) for train/val/test")
        roles = (months[:-2], months[-2], months[-1])
    train, val, test = roles
    overlap = set(train) & {val, test} or ({val} & {test})
    if overlap:
        raise SystemExit(f"months must be disjoint across splits; overlap: {overlap}")
    for m in [*train, val, test]:
        if m not in months:
            raise SystemExit(f"month {m} from the config has no file in data/raw")
    return train, val, test


def benchmark_positions(paths: list[Path], n: int, cfg: dict[str, Any], rng: np.random.Generator,
                        exclude: set[str], prefix: str) -> list[dict[str, Any]]:
    """<= max_per_game positions per game, ply in range, stratified by phase (protocol §2)."""
    b, d = cfg["benchmark"], cfg["data"]
    lo, hi = b["ply_range"]
    cand: dict[str, list[dict[str, Any]]] = {"opening": [], "middlegame": [], "endgame": []}
    seen = set(exclude)
    target = 6 * n                                    # enough candidates to stratify
    gi = 0
    for path in paths:
        for game in iter_games(path):
            gi += 1
            if not keep_game(game, d["min_plies"], d["require_normal_termination"]):
                continue
            boards = mainline_boards(game)
            plies = [i for i in range(lo, min(hi, len(boards) - 1) + 1) if not boards[i].is_game_over()]
            if not plies:
                continue
            pick = rng.choice(plies, size=min(b["max_per_game"], len(plies)), replace=False)
            gid = game.headers.get("Site") or f"{path.name}#{gi}"
            for ply in sorted(int(x) for x in pick):
                board = boards[ply]
                key = fen_key(board)
                if key in seen:
                    continue
                seen.add(key)
                phase = game_phase(board, ply, b["endgame_material_max"])
                cand[phase].append({"fen": board.fen(), "ply": ply, "phase": phase,
                                    "n_legal": board.legal_moves.count(), "game_id": gid})
            if sum(len(v) for v in cand.values()) >= target:
                break
    quotas = {"opening": int(round(n * b["opening_fraction"])), "endgame": int(round(n * b["endgame_fraction"]))}
    quotas["middlegame"] = n - quotas["opening"] - quotas["endgame"]
    chosen: list[dict[str, Any]] = []
    short = 0
    for phase in ("opening", "endgame", "middlegame"):
        pool = cand[phase]
        rng.shuffle(pool)
        take = pool[: quotas[phase]]
        short += quotas[phase] - len(take)
        chosen += take
        cand[phase] = pool[len(take):]
    if short:                                          # fill missing quota from whatever is left
        rest = cand["middlegame"] + cand["opening"] + cand["endgame"]
        rng.shuffle(rest)
        chosen += rest[:short]
        print(f"  {prefix}: phase quotas short by {short}; filled from other phases")
    chosen.sort(key=lambda r: (r["game_id"], r["ply"]))
    for i, r in enumerate(chosen):
        r["pos_id"] = f"{prefix}{i:05d}"
    return chosen


def write_csv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_openings(paths: list[Path], cfg: dict[str, Any], rng: np.random.Generator) -> list[dict[str, Any]]:
    from src.engine.Stockfish import StockfishJudge

    t, d = cfg["tournament"], cfg["data"]
    plies, band = t["opening_plies"], t["opening_eval_band_cp"]
    lines: dict[str, list[str]] = {}
    for path in paths:
        for game in iter_games(path):
            if not keep_game(game, d["min_plies"], False):
                continue
            moves = [m.uci() for m in list(game.mainline_moves())[:plies]]
            if len(moves) == plies:
                lines.setdefault(" ".join(moves), moves)
            if len(lines) >= 50 * t["n_openings"]:
                break
    keys = sorted(lines)
    rng.shuffle(keys)
    out: list[dict[str, Any]] = []
    with StockfishJudge.from_config(cfg) as judge:
        for k in keys:
            board = chess.Board()
            for u in lines[k]:
                board.push_uci(u)
            if board.is_game_over():
                continue
            cp = judge.eval_white_cp(board)
            if abs(cp) <= band:
                out.append({"opening_id": f"o{len(out):02d}", "uci_moves": k, "fen": board.fen(), "sf_cp_white": cp})
            if len(out) >= t["n_openings"]:
                break
    if len(out) < t["n_openings"]:
        print(f"  only {len(out)} openings within ±{band} cp (wanted {t['n_openings']})")
    return out


def _puzzle_stream(path: Path) -> io.TextIOBase:
    if path.name.endswith(".zst"):
        import zstandard

        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")), encoding="utf-8")
    return open(path, encoding="utf-8")


def build_puzzles(raw: Path, cfg: dict[str, Any], rng: np.random.Generator) -> list[dict[str, Any]] | None:
    files = [p for p in (raw / "lichess_db_puzzle.csv.zst", raw / "lichess_db_puzzle.csv") if p.exists()]
    if not files:
        print("  no lichess_db_puzzle.csv[.zst] in data/raw: skipping puzzles")
        return None
    per, buckets = cfg["puzzles"]["per_bucket"], cfg["puzzles"]["buckets"]
    keep: list[list[tuple[float, dict[str, Any]]]] = [[] for _ in buckets]
    with _puzzle_stream(files[0]) as fh:
        reader = csv.DictReader(fh)
        for row in tqdm(reader, desc="puzzles", unit=" rows", mininterval=5):
            rating = int(row["Rating"])
            for bi, (lo, hi) in enumerate(buckets):
                if lo <= rating < hi:
                    key = float(rng.random())                   # reservoir: keep the `per` smallest keys
                    kb = keep[bi]
                    if len(kb) < per:
                        kb.append((key, row))
                    elif key < max(kb, key=lambda t: t[0])[0]:
                        kb[max(range(len(kb)), key=lambda i: kb[i][0])] = (key, row)
                    break
    out = []
    for bi, kb in enumerate(keep):
        for _, row in sorted(kb, key=lambda t: t[1]["PuzzleId"]):
            out.append({"puzzle_id": row["PuzzleId"], "fen": row["FEN"], "moves": row["Moves"],
                        "rating": int(row["Rating"]), "bucket": bi, "themes": row["Themes"]})
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    raw, proc, bench = (resolve(cfg["data"][k]) for k in ("raw_dir", "processed_dir", "benchmark_dir"))
    by_month = discover(raw)
    months = sorted(by_month)
    train_m, val_m, test_m = assign_months(cfg, months)
    print(f"months: train {train_m}, val {val_m}, test {test_m}")
    d = cfg["data"]
    summary: dict[str, Any] = {"train_months": train_m, "val_month": val_m, "test_month": test_m,
                               "seed": cfg["seed"]}
    train_paths = [p for m in train_m for p in by_month[m]]
    train_keys: set[str] = set()

    if not args.skip_train:
        print("building train set ...")
        tr = build_training_set(train_paths, d["positions_per_game_train"], d["min_ply_train"], d["min_plies"],
                                d["require_normal_termination"], rng, d["max_train_positions"], seen=train_keys)
        print("building val (value net) set ...")
        va = build_training_set(by_month[val_m], d["positions_per_game_train"], d["min_ply_train"], d["min_plies"],
                                d["require_normal_termination"], rng, d["n_val_net"], seen=set(train_keys))
        sf_tr = sf_va = None
        if args.sf_labels or args.val_sf_nodes:
            from src.engine.Stockfish import StockfishJudge

            nodes = args.val_sf_nodes or cfg["label"]["sf_nodes"]
            with StockfishJudge.from_config(cfg, nodes=nodes) as judge:
                sf_va = np.array([judge.eval_white_cp(chess.Board(f)) for f in tqdm(va.fens, desc="val SF")])
            if args.sf_labels:
                with StockfishJudge.from_config(cfg, nodes=cfg["label"]["sf_nodes"]) as judge:
                    sf_tr = np.array([judge.eval_white_cp(chess.Board(f)) for f in tqdm(tr.fens, desc="train SF")])
        tr.to_npz(proc / "train.npz", sf_tr)
        va.to_npz(proc / "val.npz", sf_va)
        summary.update(n_train=len(tr.fens), n_val_net=len(va.fens),
                       train_label_mean=float(np.mean(tr.y)) if tr.y else None)
        print(f"  train {len(tr.fens):,}  val {len(va.fens):,}")
    else:
        if (proc / "train.npz").exists():
            with np.load(proc / "train.npz") as z:
                train_keys = {" ".join(f.split()[:4]) for f in z["fens"]}

    cols = ["pos_id", "fen", "ply", "phase", "n_legal", "game_id"]
    for split, month, n in (("val", val_m, cfg["benchmark"]["n_val"]), ("test", test_m, cfg["benchmark"]["n_test"])):
        rows = benchmark_positions(by_month[month], n, cfg, rng, train_keys, prefix=split[0])
        write_csv(bench / f"positions_{split}.csv", rows, cols)
        counts = {ph: sum(r["phase"] == ph for r in rows) for ph in ("opening", "middlegame", "endgame")}
        summary[f"positions_{split}"] = {"n": len(rows), **counts}
        print(f"  positions_{split}: {len(rows)} {counts}")

    if not args.skip_openings:
        ops = build_openings(by_month[test_m], cfg, rng)
        write_csv(bench / "openings.csv", ops, ["opening_id", "uci_moves", "fen", "sf_cp_white"])
        summary["n_openings"] = len(ops)
        print(f"  openings: {len(ops)}")
    if not args.skip_puzzles:
        pz = build_puzzles(raw, cfg, rng)
        if pz is not None:
            write_csv(bench / "puzzles.csv", pz, ["puzzle_id", "fen", "moves", "rating", "bucket", "themes"])
            summary["n_puzzles"] = len(pz)
            print(f"  puzzles: {len(pz)}")
    bench.mkdir(parents=True, exist_ok=True)
    (bench / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {bench / 'split_summary.json'}")


if __name__ == "__main__":
    main()
