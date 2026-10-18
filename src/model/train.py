"""Train ValueCNN. Contract: docs/SPEC.md, `dataset.py and train.py`.

Usage (from repo root):
    python -m src.model.train --config configs/default.yaml [--epochs 20 --batch-size 256 --lr 1e-3 --device cuda]

Loads data/processed/{train,val}.npz (made by scripts/prepare_data.py). MSELoss, AdamW(lr, weight_decay),
CosineAnnealingLR. Saves model.checkpoint on best validation loss. Prints once the constant-predictor
baseline MSE, then per epoch: train_loss, val_loss, val_sign_acc_excl_draws. After training, prints the
Spearman correlation between predictions and tanh(SF cp/400) on val if val.npz has sf_cp_white
(protocol section 5), and writes a train_log.csv next to the checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model.architecture import model_from_config, save_checkpoint
from src.model.dataset import PositionDataset, blend_labels, load_npz
from src.model.encoding import unpack_planes
from src.utils import load_config, pick_device, resolve, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-train", type=int, default=None, help="use only the first N training positions (debug)")
    return p.parse_args()


def labels(d: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    z = d["y"].astype(np.float32)
    if cfg["label"]["mode"] == "blend":
        if "sf_cp_white" not in d:
            raise SystemExit("label.mode=blend needs sf_cp_white in the npz: rerun prepare_data with --sf-labels")
        return blend_labels(z, d["sf_cp_white"], cfg["label"]["blend_lambda"])
    return z


@torch.no_grad()
def predict(model: nn.Module, packed: np.ndarray, device: str, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(packed), bs):
        x = torch.from_numpy(unpack_planes(packed[i:i + bs])).to(device)
        out.append(model(x).squeeze(1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def sign_acc_excl_draws(pred: np.ndarray, z: np.ndarray) -> float:
    mask = z != 0
    return float(np.mean(np.sign(pred[mask]) == np.sign(z[mask]))) if mask.any() else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    ok = ~(np.isnan(a) | np.isnan(b))
    return float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() > 2 else float("nan")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    tc = cfg["train"]
    epochs = args.epochs or tc["epochs"]
    bs = args.batch_size or tc["batch_size"]
    lr = args.lr or tc["lr"]
    device = pick_device(args.device or tc["device"])
    seed_everything(cfg["seed"])

    proc = resolve(cfg["data"]["processed_dir"])
    tr, va = load_npz(proc / "train.npz"), load_npz(proc / "val.npz")
    if args.max_train:
        tr = {k: v[: args.max_train] for k, v in tr.items()}
    y_tr, y_va = labels(tr, cfg), labels(va, cfg)
    print(f"train {len(y_tr):,} positions, val {len(y_va):,}, device {device}, label mode {cfg['label']['mode']}")
    base = float(np.mean((y_va - y_tr.mean()) ** 2))
    print(f"constant-predictor baseline val MSE (predict train mean {y_tr.mean():+.3f}): {base:.4f}")

    ds = PositionDataset(tr["planes"], y_tr, color_flip=cfg["model"]["color_flip_augmentation"], seed=cfg["seed"])
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=tc["num_workers"],
                    pin_memory=device.startswith("cuda"), drop_last=len(ds) > bs)
    model = model_from_config(cfg).to(device)
    print(f"ValueCNN parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = nn.MSELoss()

    ckpt = resolve(cfg["model"]["checkpoint"])
    log_path = ckpt.parent / "train_log.csv"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    with open(log_path, "w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["epoch", "train_loss", "val_loss", "val_sign_acc_excl_draws", "lr", "seconds"])
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            total, n = 0.0, 0
            for x, y in dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(x)
                n += len(x)
            sched.step()
            pred = predict(model, va["planes"], device)
            val_loss = float(np.mean((pred - y_va) ** 2))
            acc = sign_acc_excl_draws(pred, va["y"])
            secs = time.time() - t0
            print(f"epoch {epoch:3d}  train_loss {total / max(n, 1):.4f}  val_loss {val_loss:.4f}  "
                  f"val_sign_acc_excl_draws {acc:.3f}  ({secs:.0f}s)", flush=True)
            log.writerow([epoch, total / max(n, 1), val_loss, acc, opt.param_groups[0]["lr"], round(secs, 1)])
            fh.flush()
            if val_loss < best:
                best = val_loss
                save_checkpoint(model, ckpt, {"epoch": epoch, "val_loss": val_loss, "label_mode": cfg["label"]["mode"]})

    print(f"best val_loss {best:.4f} (baseline {base:.4f}); checkpoint {ckpt}")
    if "sf_cp_white" in va:
        from src.model.architecture import load_checkpoint

        pred = predict(load_checkpoint(ckpt, device), va["planes"], device)
        rho = spearman(pred, np.tanh(va["sf_cp_white"] / 400.0))
        print(f"Spearman(value, tanh(SF cp/400)) on val: {rho:.3f}"
              + ("   < 0.3: switch label.mode to blend (protocol section 5)" if rho < 0.3 else ""))
    else:
        print("val.npz has no sf_cp_white: run prepare_data with --val-sf-nodes to get the Spearman check")


if __name__ == "__main__":
    main()
