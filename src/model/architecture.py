"""Value network "ValueNet-6x128". Contract: docs/SPEC.md, section `src/model/architecture.py`.

Input  (batch, 18, 8, 8) from src.model.encoding.board_to_planes.
Output (batch, 1) in [-1, 1], WHITE perspective (+1 white winning). Never convert to side-to-move here.

Stem:  Conv3x3(18->128, bias=False) -> BN -> ReLU
Body:  6 residual blocks of [conv3x3 -> BN -> ReLU -> conv3x3 -> BN] + skip -> ReLU   (128 channels)
Head:  Conv1x1(128->32) -> BN -> ReLU -> Flatten(2048) -> Linear(2048->256) -> ReLU -> Linear(256->1) -> Tanh
He (kaiming_normal_) init for convs. About 2.32M parameters. Do NOT global-pool the head to a scalar.
Tests: tests/test_value_cnn.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import chess
import numpy as np
import torch
from torch import nn

from src.model.encoding import board_to_planes


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)


class ValueCNN(nn.Module):
    def __init__(
        self,
        planes: int = 18,
        channels: int = 128,
        res_blocks: int = 6,
        head_channels: int = 32,
        fc_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.hparams = dict(planes=planes, channels=channels, res_blocks=res_blocks,
                            head_channels=head_channels, fc_hidden=fc_hidden)
        self.stem = nn.Sequential(
            nn.Conv2d(planes, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU()
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(res_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(channels, head_channels, 1, bias=False), nn.BatchNorm2d(head_channels), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(head_channels * 64, fc_hidden), nn.ReLU(),
            nn.Linear(fc_hidden, 1), nn.Tanh(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(x)))


def model_from_config(cfg: dict[str, Any]) -> ValueCNN:
    m = cfg["model"]
    return ValueCNN(planes=m["planes"], channels=m["channels"], res_blocks=m["res_blocks"],
                    head_channels=m["head_channels"], fc_hidden=m["fc_hidden"])


def save_checkpoint(model: ValueCNN, path: str | Path, extra: dict[str, Any] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hparams": model.hparams, **(extra or {})}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> ValueCNN:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ValueCNN(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


class ValueFn:
    """Callable board -> white-perspective float in [-1, 1]. Counts evaluations in `calls`."""

    def __init__(self, model: ValueCNN, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.calls = 0

    @torch.no_grad()
    def __call__(self, board: chess.Board) -> float:
        self.calls += 1
        x = torch.from_numpy(board_to_planes(board)).unsqueeze(0).to(self.device)
        return float(self.model(x).item())

    @torch.no_grad()
    def batch(self, boards: Sequence[chess.Board]) -> np.ndarray:
        self.calls += len(boards)
        x = torch.from_numpy(np.stack([board_to_planes(b) for b in boards])).to(self.device)
        return self.model(x).squeeze(1).float().cpu().numpy()


def make_value_fn(model: ValueCNN, device: str | torch.device) -> Callable[[chess.Board], float]:
    """Return value_fn(board) -> white-perspective float, using model.eval() and torch.no_grad().

    The returned object exposes an integer attribute `calls` counting evaluations (for cnn_evals logs).
    """
    return ValueFn(model, device)
