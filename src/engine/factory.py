"""Build any agent named in docs/EXPERIMENT_PROTOCOL.md from the config.

Names: laya_only, mcts_uniform, always_s2, hybrid_s2_{10,25,50,75}, random_s2_{25,50} (seeded),
sf_elo_<N>. Hybrid thresholds come from configs/gate_frozen.yaml (written by scripts/calibrate_gate.py).
Laya and the value network are loaded once (Resources) and shared between agents.
"""

from __future__ import annotations

import re
from typing import Any

from src.engine.gating import AlwaysGate, ConfidenceGate, NeverGate, RandomGate
from src.engine.HybridAgent import HybridAgent, StockfishAnchorAgent

AGENT_NAMES = (
    "laya_only", "mcts_uniform", "always_s2",
    "hybrid_s2_10", "hybrid_s2_25", "hybrid_s2_50", "hybrid_s2_75",
    "random_s2_25", "random_s2_50",
)
_HYBRID = re.compile(r"^hybrid_s2_(\d+)$")
_RANDOM = re.compile(r"^random_s2_(\d+)$")
_SF = re.compile(r"^sf_elo_(\d+)$")


class Resources:
    """Lazily loaded shared models."""

    def __init__(self, cfg: dict[str, Any], gate_cfg: dict[str, Any] | None = None) -> None:
        self.cfg, self.gate_cfg = cfg, gate_cfg
        self._laya = None
        self._value_fn = None
        self.agents: list[Any] = []

    @property
    def laya(self) -> Any:
        if self._laya is None:
            from src.engine.LayaAgent import LayaAgent

            self._laya = LayaAgent.from_config(self.cfg, self.gate_cfg)
        return self._laya

    @property
    def value_fn(self) -> Any:
        if self._value_fn is None:
            from src.model.architecture import load_checkpoint, make_value_fn
            from src.utils import pick_device, resolve

            device = pick_device(self.cfg["train"]["device"])
            self._value_fn = make_value_fn(load_checkpoint(resolve(self.cfg["model"]["checkpoint"]), device), device)
        return self._value_fn

    def close(self) -> None:
        """Close engine subprocesses owned by agents built from these resources."""
        for agent in self.agents:
            player = getattr(agent, "player", None)
            if player is not None:
                player.close()

    def search(self) -> Any:
        from src.mcts.SearchTree import SearchTree

        m = self.cfg["mcts"]
        return SearchTree(self.value_fn, m["simulations"], m["c_puct"], m["fpu_reduction"], m["prior_temperature"])


def tau_for(gate_cfg: dict[str, Any] | None, pct: int) -> float:
    if gate_cfg is None:
        raise SystemExit("configs/gate_frozen.yaml not found: run scripts.calibrate_gate first")
    taus = {int(round(float(k) * 100)): float(v) for k, v in gate_cfg["tau"].items()}
    if pct not in taus:
        raise SystemExit(f"no tau for S2 rate {pct}% in gate_frozen.yaml (have {sorted(taus)})")
    return taus[pct]


def build_agent(name: str, res: Resources, seed: int = 0) -> Any:
    if name == "laya_only":
        return HybridAgent(res.laya, None, NeverGate(), name=name)
    if name == "always_s2":
        return HybridAgent(res.laya, res.search(), AlwaysGate(), use_laya_prior=True, name=name)
    if name == "mcts_uniform":
        return HybridAgent(None, res.search(), AlwaysGate(), use_laya_prior=False, name=name)
    if m := _HYBRID.match(name):
        return HybridAgent(res.laya, res.search(), ConfidenceGate(tau_for(res.gate_cfg, int(m.group(1)))), name=name)
    if m := _RANDOM.match(name):
        return HybridAgent(res.laya, res.search(), RandomGate(int(m.group(1)) / 100, seed), name=name)
    if m := _SF.match(name):
        from src.engine.Stockfish import StockfishPlayer

        a = res.cfg["tournament"]["anchors"]
        player = StockfishPlayer(res.cfg["judge"]["path"], int(m.group(1)), a["movetime_ms"], a["threads"], a["hash_mb"])
        agent = StockfishAnchorAgent(player, name)
        res.agents.append(agent)
        return agent
    raise ValueError(f"unknown agent {name!r}")
