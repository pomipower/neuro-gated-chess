"""Shared helpers: config loading, seeding, run directories, environment snapshots."""

from __future__ import annotations

import datetime as _dt
import json
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    with open(p) as f:
        return yaml.safe_load(f)


def load_gate_config(path: str | Path = "configs/gate_frozen.yaml") -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return None
    with open(p) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def pick_device(name: str) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return name


def new_run_id(prefix: str) -> str:
    return f"{prefix}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _stockfish_version(path: str) -> str | None:
    exe = shutil.which(path) or (path if Path(path).exists() else None)
    if exe is None:
        return None
    try:
        import chess.engine

        with chess.engine.SimpleEngine.popen_uci(exe) as eng:
            return str(eng.id.get("name"))
    except Exception:
        return None


def env_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {
        "time": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_sha": _git_sha(),
        "laya_model": cfg.get("laya", {}).get("model_name"),
        "laya_revision": cfg.get("laya", {}).get("revision"),
        "stockfish": _stockfish_version(cfg.get("judge", {}).get("path", "stockfish")),
    }
    for mod in ("numpy", "chess", "torch", "pandas"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch

        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["cuda"] = None
    return info


def prepare_run_dir(cfg: dict[str, Any], run_id: str) -> Path:
    """Create data/results/<run_id>/ with config.yaml and env.json (kept if they already exist)."""
    d = resolve(cfg.get("results_dir", "data/results")) / run_id
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "config.yaml").exists():
        with open(d / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    if not (d / "env.json").exists():
        (d / "env.json").write_text(json.dumps(env_snapshot(cfg), indent=2))
    return d
