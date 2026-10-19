"""Health check for a value-network checkpoint (v1 or v2). CPU or GPU, about a minute.

Reports, on the value-net validation set (val.npz with sf_cp_white):
  - Spearman and Pearson correlation of the prediction with tanh(Stockfish cp / 400)
  - sign agreement where |Stockfish| >= 100 cp
  - dead-head rate: share of positions where every hidden unit of the 256-unit layer is <= 0 (for a ReLU head
    this means the output is a constant, whatever the position)
  - share of positions whose output equals the most common output (constant-output rate)
  - sanity positions: start, White a queen up, Black a queen up (should be ~0, clearly +, clearly -)

Run from the repo root:
    python -m scripts.check_value_net --ckpt src/model/checkpoints/best_model.pt --val data/processed_v2/val.npz
"""

from __future__ import annotations

import argparse

import chess
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from src.model.architecture import load_checkpoint
from src.model.dataset import load_npz
from src.model.encoding import board_to_planes, unpack_planes

SANITY = {
    "start position": chess.STARTING_FEN,
    "White a queen up": "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "Black a queen up": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
    "White a rook up (endgame)": "4k3/pp6/8/8/8/8/PP6/R3K3 w - - 0 1",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val", default="data/processed_v2/val.npz")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max", type=int, default=20000)
    return p.parse_args()


@torch.no_grad()
def run(model: torch.nn.Module, x: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    hidden: list[torch.Tensor] = []
    fc = [m for m in model.head if isinstance(m, torch.nn.Linear)][0]
    h = fc.register_forward_hook(lambda m, i, o: hidden.append(o.detach().cpu()))
    preds = []
    for i in range(0, len(x), 1024):
        preds.append(model(torch.from_numpy(x[i:i + 1024]).to(device)).squeeze(1).float().cpu())
    h.remove()
    return torch.cat(preds).numpy(), torch.cat(hidden).numpy()


def main() -> None:
    args = parse_args()
    model = load_checkpoint(args.ckpt, args.device)
    print(f"checkpoint {args.ckpt}  head_act={model.hparams.get('head_act', 'relu')}")
    va = load_npz(args.val)
    n = min(args.max, len(va["y"]))
    x = unpack_planes(va["planes"][:n]).astype(np.float32)
    pred, hid = run(model, x, args.device)
    dead = (hid <= 0).all(axis=1)
    vals, counts = np.unique(np.round(pred, 5), return_counts=True)
    print(f"positions: {n:,}")
    print(f"dead-head rate (all 256 hidden units <= 0): {dead.mean():.1%}")
    print(f"constant-output rate (output == most common value {vals[counts.argmax()]:+.4f}): {counts.max() / n:.1%}")
    print(f"output std: {pred.std():.3f}")
    if "sf_cp_white" in va:
        t = np.tanh(va["sf_cp_white"][:n] / 400.0)
        print(f"Spearman vs tanh(SF/400): {spearmanr(pred, t)[0]:.3f}   Pearson: {pearsonr(pred, t)[0]:.3f}")
        big = np.abs(va["sf_cp_white"][:n]) >= 100
        print(f"sign agreement where |SF| >= 100 cp: {np.mean(np.sign(pred[big]) == np.sign(t[big])):.1%} (n={big.sum():,})")
    else:
        print("val file has no sf_cp_white: run scripts.label_sf first for the correlation checks")
    xs = np.stack([board_to_planes(chess.Board(f)) for f in SANITY.values()])
    sp, _ = run(model, xs, args.device)
    for (k, _), v in zip(SANITY.items(), sp):
        print(f"  {k:28s} {v:+.3f}")


if __name__ == "__main__":
    main()
