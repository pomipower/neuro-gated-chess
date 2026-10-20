"""Blunder audit: WHY does an agent still lose? (diagnostic; uses already-played games, CPU only)

For every move the agent played in the given tournament run(s), the Stockfish judge (fixed nodes) computes the
centipawn loss. Each blunder (CPL >= 200) is classified:
  mate_allowed      : after the move Stockfish sees a forced mate against the mover
  missed_mate       : the mover had a forced mate (Stockfish best line is mate) and did not play a mating line
  tactic_depth_N    : along Stockfish's best reply line the mover loses >= 2 pawns of material within N plies
                      (N <= 2 should be caught by the v2 safety net; N >= 3 is beyond its horizon)
  positional        : no material loss within 10 plies (evaluation, king safety, pawn structure ...)
plus phase, whether search was used, whether the safety net overrode, and the game result.

Run from the repo root (about 30-60 min for 100 games; runs alongside GPU jobs):
    python -m scripts.blunder_audit --config configs/v2.yaml --run-ids v2_tour_a --agent v2_full
"""

from __future__ import annotations

import argparse

import chess
import chess.engine
import pandas as pd
from tqdm import tqdm

from src.engine.Stockfish import StockfishJudge, score_to_cp
from src.engine.tactics import material, non_pawn_material
from src.utils import load_config, resolve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/v2.yaml")
    p.add_argument("--run-ids", required=True)
    p.add_argument("--agent", required=True)
    p.add_argument("--nodes", type=int, default=100_000)
    p.add_argument("--max-games", type=int, default=None)
    return p.parse_args()


def phase(board: chess.Board) -> str:
    npm = non_pawn_material(board, chess.WHITE) + non_pawn_material(board, chess.BLACK)
    if board.ply() < 20:
        return "opening"
    return "endgame" if npm <= 2600 else "middlegame"


def classify(judge: StockfishJudge, board: chess.Board, move: chess.Move) -> tuple[str, int | None]:
    me = board.turn
    best = judge._eng.call(lambda e: e.analyse(board, judge.limit, game=object()))
    if best["score"].pov(me).is_mate() and (best["score"].pov(me).mate() or 0) > 0:
        after = board.copy(stack=False)
        after.push(move)
        if not after.is_checkmate():
            info = judge._eng.call(lambda e: e.analyse(after, judge.limit, game=object()))
            if not (info["score"].pov(me).is_mate() and (info["score"].pov(me).mate() or 0) > 0):
                return "missed_mate", None
    after = board.copy(stack=False)
    after.push(move)
    if after.is_game_over():
        return "other", None
    info = judge._eng.call(lambda e: e.analyse(after, judge.limit, game=object()))
    sc = info["score"].pov(me)
    if sc.is_mate() and (sc.mate() or 0) < 0:
        return "mate_allowed", abs(sc.mate()) * 2 - 1
    before = material(board, me)
    b = after.copy(stack=False)
    for i, mv in enumerate(info.get("pv", [])[:10], start=1):
        if mv not in b.legal_moves:
            break
        b.push(mv)
        if before - material(b, me) >= 200:
            return f"tactic_depth_{min(i, 6)}{'+' if i >= 6 else ''}", i
    return "positional", None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    base = resolve(cfg["results_dir"])
    ids = [r.strip() for r in args.run_ids.split(",")]
    mv = pd.concat([pd.read_csv(base / r / "moves.csv") for r in ids], ignore_index=True)
    gm = pd.concat([pd.read_csv(base / r / "games.csv") for r in ids], ignore_index=True).set_index("game_id")
    mv = mv[mv.agent == args.agent]
    if args.max_games:
        keep = list(dict.fromkeys(mv.game_id))[: args.max_games]
        mv = mv[mv.game_id.isin(keep)]
    rows = []
    with StockfishJudge.from_config(cfg, nodes=args.nodes) as judge:
        for m in tqdm(mv.itertuples(), total=len(mv), desc=f"auditing {args.agent}"):
            board = chess.Board(m.fen_before)
            move = chess.Move.from_uci(m.move_uci)
            cpl = judge.cpl(board, move)
            r = {"game_id": m.game_id, "ply": m.ply, "cpl": cpl, "phase": phase(board), "system": m.system,
                 "safety_override": getattr(m, "safety_override", ""), "special": getattr(m, "special", ""),
                 "eval_before": judge.analyse(board).score_cp}
            if cpl >= cfg["judge"]["blunder_cpl"]:
                r["kind"], r["depth"] = classify(judge, board, move)
            else:
                r["kind"], r["depth"] = "", None
            g = gm.loc[m.game_id]
            mover_white = board.turn == chess.WHITE
            res = {"1-0": 1.0, "0-1": 0.0}.get(g.result, 0.5)
            r["game_score"] = res if mover_white else 1 - res
            rows.append(r)
    df = pd.DataFrame(rows)
    out = base / ids[0] / f"blunder_audit_{args.agent}.csv"
    df.to_csv(out, index=False)
    bl = df[df.kind != ""]
    pd.set_option("display.width", 200)
    print(f"\n{args.agent}: {len(df)} moves in {df.game_id.nunique()} games; blunders (CPL>=200): {len(bl)} "
          f"({len(bl) / len(df):.1%}); ACPL {df.cpl.mean():.1f}")
    print("\nblunders by kind:\n", bl.kind.value_counts().to_string())
    print("\nblunders by phase:\n", bl.phase.value_counts().to_string())
    print("\nblunders by system (1 = no search, 2 = search, 3 = tablebase/mate1):\n", bl.system.value_counts().to_string())
    first = bl[bl.eval_before > -150].sort_values("ply").groupby("game_id").head(1)
    lost = df.groupby("game_id").game_score.first()
    print(f"\ngames lost: {(lost == 0).sum()}; of those with a blunder from a not-yet-lost position "
          f"(eval > -1.5): {first[first.game_id.isin(lost[lost == 0].index)].game_id.nunique()}")
    print("decisive first blunders (eval was > -1.5 before) by kind:\n", first.kind.value_counts().to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
