"""Download shards of the Hugging Face dataset `nuriyev/lichess-elite` and write monthly PGN files.

The dataset (CC0) is 27 parquet shards (`train-00000.parquet` ... `train-00026.parquet`, ~335 MB each) with one
Lichess Elite game per row: event, site, white, black, result, utcdate, utctime, whiteelo, blackelo, eco, opening,
timecontrol, termination, moves. This script turns the chosen shards into
    data/raw/lichess_elite_YYYY-MM.pgn
which is exactly what scripts/prepare_data.py expects (month taken from `utcdate`).

Default: the 3 newest shards (24-26). Shards are assumed to be in date order; the script prints the games per
month it found, so check that output. A month cut off at the start of the first downloaded shard is usually
incomplete: either download one more shard or leave that month out of the config.

Run from the repo root:
    pip install pyarrow huggingface_hub
    python -m scripts.convert_hf_elite --shards 24 25 26
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

REPO = "nuriyev/lichess-elite"
COLUMNS = ["event", "site", "white", "black", "result", "utcdate", "utctime", "whiteelo", "blackelo", "eco",
           "timecontrol", "termination", "moves"]
_TRAILING_RESULT = re.compile(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", type=int, nargs="+", default=[24, 25, 26], help="shard numbers (0-26)")
    p.add_argument("--repo", default=REPO)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--cache", default="data/raw/hf_cache", help="where downloaded parquet shards are kept")
    p.add_argument("--max-games-per-month", type=int, default=150_000,
                   help="cap per month (150k games is far more than the pipeline samples)")
    p.add_argument("--local", nargs="*", default=None, help="use these local parquet files instead of downloading")
    return p.parse_args()


def _quote(v: Any) -> str:
    return str(v if v is not None else "?").replace("\\", "\\\\").replace('"', '\\"')


def to_pgn(row: dict[str, Any]) -> str | None:
    result = row.get("result") or "*"
    moves = (row.get("moves") or "").strip()
    if not moves or result not in ("1-0", "0-1", "1/2-1/2"):
        return None
    moves = _TRAILING_RESULT.sub("", moves)
    headers = [
        ("Event", row.get("event")), ("Site", row.get("site")), ("Date", row.get("utcdate")),
        ("White", row.get("white")), ("Black", row.get("black")), ("Result", result),
        ("UTCDate", row.get("utcdate")), ("UTCTime", row.get("utctime")),
        ("WhiteElo", row.get("whiteelo")), ("BlackElo", row.get("blackelo")), ("ECO", row.get("eco")),
        ("TimeControl", row.get("timecontrol")), ("Termination", row.get("termination") or "Normal"),
    ]
    head = "\n".join(f'[{k} "{_quote(v)}"]' for k, v in headers)
    return f"{head}\n\n{moves} {result}\n\n"


def month_of_date(utcdate: Any) -> str | None:
    m = re.match(r"(\d{4})[.\-/](\d{2})", str(utcdate or ""))
    return f"{m.group(1)}-{m.group(2)}" if m else None


def convert(paths: list[Path], out_dir: Path, cap: int) -> Counter:
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[str, TextIO] = {}
    written: Counter = Counter()
    skipped = 0
    try:
        for path in paths:
            pf = pq.ParquetFile(path)
            cols = [c for c in COLUMNS if c in pf.schema_arrow.names]
            missing = {"result", "utcdate", "moves"} - set(cols)
            if missing:
                raise SystemExit(f"{path.name}: missing columns {missing}; found {pf.schema_arrow.names}")
            for batch in pf.iter_batches(batch_size=20_000, columns=cols):
                for row in batch.to_pylist():
                    month = month_of_date(row.get("utcdate"))
                    if month is None or written[month] >= cap:
                        skipped += 1
                        continue
                    pgn = to_pgn(row)
                    if pgn is None:
                        skipped += 1
                        continue
                    if month not in handles:
                        handles[month] = open(out_dir / f"lichess_elite_{month}.pgn", "w", encoding="utf-8")
                    handles[month].write(pgn)
                    written[month] += 1
            print(f"  {path.name}: running totals {dict(sorted(written.items()))}", flush=True)
    finally:
        for fh in handles.values():
            fh.close()
    print(f"skipped {skipped:,} rows (no result/moves, bad date, or month cap reached)")
    return written


def main() -> None:
    args = parse_args()
    if args.local:
        paths = [Path(p) for p in args.local]
    else:
        from huggingface_hub import hf_hub_download

        paths = []
        for s in args.shards:
            name = f"train-{s:05d}.parquet"
            print(f"downloading {args.repo}/{name} (~335 MB) ...", flush=True)
            paths.append(Path(hf_hub_download(args.repo, name, repo_type="dataset", local_dir=args.cache)))
    written = convert(paths, Path(args.out), args.max_games_per_month)
    print("\ngames written per month:")
    for m, n in sorted(written.items()):
        print(f"  {m}: {n:,}")
    months = sorted(written)
    if len(months) >= 3:
        print(f"\nThe first month ({months[0]}) may be cut off by the shard boundary. Suggested config "
              f"(configs/default.yaml -> data):\n  train_months: {months[1:-2] or months[:-2]}\n"
              f"  val_month: \"{months[-2]}\"\n  test_month: \"{months[-1]}\"")
    else:
        print("\nFewer than 3 months: download more shards (e.g. --shards 23 24 25 26).")
    print(f"\nYou can delete {args.cache} afterwards to free space.")


if __name__ == "__main__":
    main()
