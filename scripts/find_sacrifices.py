"""Find real sacrifices our bots played in tournament games (for the demo's commentary mode).

A move by one of our agents is a *sacrifice* when:
  1. it puts a piece worth >= 3 where the opponent can win material (static exchange check), and
  2. along Stockfish's best continuation (judge, fixed nodes) the mover really ends up >= 2 pawns of material down
     a few plies later.
It is *sound* if the judge's centipawn loss for the move is <= SOUND_CPL (Stockfish agrees it was fine).
Unsound ones are kept too (the viewer says so honestly).

Writes data/results/<run_id>/sacrifices.csv (game_id, ply, agent, move_uci, san, piece, material_loss, cpl,
sound, system). Stockfish only, no GPU; runs on CPU alongside anything else.

Run from the repo root:  python -m scripts.find_sacrifices --run-id tour_1
"""

from __future__ import annotations

import argparse

import chess
import pandas as pd
from tqdm import tqdm

from src.engine.Stockfish import StockfishJudge
from src.utils import load_config, resolve

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
NAME = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen"}
SOUND_CPL = 50
MIN_LOSS = 2
PV_PLIES = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run-id", required=True)
    p.add_argument("--nodes", type=int, default=None, help="judge nodes (default: judge.nodes from config)")
    return p.parse_args()


def material(board: chess.Board, color: chess.Color) -> int:
    return sum(VALUE[p.piece_type] * (1 if p.color == color else -1) for p in board.piece_map().values())


def looks_like_sacrifice(board: chess.Board, move: chess.Move) -> tuple[bool, int]:
    """Cheap filter: moved piece (>= 3) lands where a cheaper enemy piece attacks it, or where it is undefended
    and attacked, net of anything it captured. Returns (candidate, piece_type)."""
    piece = board.piece_at(move.from_square)
    if piece is None or VALUE[piece.piece_type] < 3:
        return False, 0
    captured = board.piece_at(move.to_square)
    gain = VALUE[captured.piece_type] if captured else (1 if board.is_en_passant(move) else 0)
    after = board.copy(stack=False)
    after.push(move)
    attackers = after.attackers(after.turn, move.to_square)
    if not attackers:
        return False, 0
    defended = bool(after.attackers(not after.turn, move.to_square))
    cheapest = min(VALUE[after.piece_at(sq).piece_type] or 100 for sq in attackers)
    risk = VALUE[piece.piece_type] - (cheapest if defended else 0)
    return risk - gain >= MIN_LOSS, piece.piece_type


def material_loss_along_pv(judge: StockfishJudge, board: chess.Board, move: chess.Move) -> int:
    """Mover's material change after `move` and up to PV_PLIES of Stockfish's best continuation (positive = lost)."""
    me = board.turn
    before = material(board, me)
    after = board.copy(stack=False)
    after.push(move)
    if after.is_game_over():
        return 0
    info = judge._eng.call(lambda e: e.analyse(after, judge.limit, game=object()))
    worst = 0
    for mv in info.get("pv", [])[:PV_PLIES]:
        if mv not in after.legal_moves:
            break
        after.push(mv)
        worst = max(worst, before - material(after, me))
    return before - material(after, me) if worst >= MIN_LOSS else worst


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = resolve(cfg["results_dir"]) / args.run_id
    moves = pd.read_csv(run_dir / "moves.csv")
    ours = moves[moves.system.isin([1, 2])]
    judged = run_dir / "moves_judged.csv"
    cpl_lookup = {}
    if judged.exists():
        j = pd.read_csv(judged)
        cpl_lookup = {(g, int(p)): c for g, p, c in zip(j.game_id, j.ply, j.cpl)}
    rows = []
    with StockfishJudge.from_config(cfg, nodes=args.nodes) as judge:
        for m in tqdm(ours.itertuples(), total=len(ours), desc="scanning moves"):
            board = chess.Board(m.fen_before)
            move = chess.Move.from_uci(m.move_uci)
            cand, ptype = looks_like_sacrifice(board, move)
            if not cand:
                continue
            loss = material_loss_along_pv(judge, board, move)
            if loss < MIN_LOSS:
                continue
            cpl = cpl_lookup.get((m.game_id, int(m.ply)))
            if cpl is None:
                cpl = judge.cpl(board, move)
            rows.append({"game_id": m.game_id, "ply": int(m.ply), "agent": m.agent, "move_uci": m.move_uci,
                         "san": board.san(move), "piece": NAME.get(ptype, "piece"), "material_loss": int(loss),
                         "cpl": round(float(cpl), 1), "sound": bool(cpl <= SOUND_CPL), "system": int(m.system)})
    out = pd.DataFrame(rows, columns=["game_id", "ply", "agent", "move_uci", "san", "piece", "material_loss", "cpl",
                                      "sound", "system"])
    out.to_csv(run_dir / "sacrifices.csv", index=False)
    print(f"{len(out)} sacrifices found ({int(out.sound.sum()) if len(out) else 0} sound) -> {run_dir / 'sacrifices.csv'}")
    if len(out):
        print(out.sort_values(["sound", "material_loss"], ascending=False).head(10).to_string(index=False))
        print("\nTo show one in the viewer with the full thought process:\n"
              f"  python -m scripts.explain_game --run-id {args.run_id} --games <game_id>\n"
              f"  python -m scripts.export_games --run-id {args.run_id}")


if __name__ == "__main__":
    main()
