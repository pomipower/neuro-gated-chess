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

    def value_fn_for(self, checkpoint: str) -> Any:
        """Value function for a specific checkpoint (v2 agents choose old or new net)."""
        if checkpoint == self.cfg["model"]["checkpoint"]:
            return self.value_fn
        cache = self.__dict__.setdefault("_vfn_cache", {})
        if checkpoint not in cache:
            from src.model.architecture import load_checkpoint, make_value_fn
            from src.utils import pick_device, resolve

            device = pick_device(self.cfg["train"]["device"])
            cache[checkpoint] = make_value_fn(load_checkpoint(resolve(checkpoint), device), device)
        return cache[checkpoint]

    @property
    def tablebase(self) -> Any:
        if not hasattr(self, "_tb"):
            from src.engine.AgentV2 import Tablebase
            from src.utils import resolve

            d = self.cfg.get("v2", {}).get("tablebase_dir")
            self._tb = Tablebase(resolve(d) if d else None)
            print(f"[v2] tablebases: {'up to ' + str(self._tb.max_pieces) + ' pieces' if self._tb.tb else 'not found (disabled)'}")
        return self._tb

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


V2_AGENTS = {
    # name: (search?, gate, improved search, checkpoint key, overrides)
    "v2_safe": (False, "never", True, "new", {}),
    "v2_full": (True, "critical", True, "new", {}),
    "v2_always": (True, "always", True, "new", {}),
    "v2_full_oldnet": (True, "critical", True, "old", {}),
    "v2_full_plainsearch": (True, "critical", False, "new", {}),
    "v2_full_nosafety": (True, "critical", True, "new", {"safety": False}),
    "v2_full_nocontempt": (True, "critical", True, "new", {"contempt": 0.0}),
    "v2_full_notb": (True, "critical", True, "new", {"tablebase": False}),
}


def build_v2_agent(name: str, res: Resources) -> Any:
    from src.engine.AgentV2 import V2Agent, make_search_factory

    if "v2" not in res.cfg:
        raise SystemExit(f"{name} needs the v2 config: use --config configs/v2.yaml (python -m scripts.make_v2_config)")
    with_search, gate, improved, ck, over = V2_AGENTS[name]
    o = {**res.cfg["v2"], "gate": gate, **over}
    o["safety_opts"] = res.cfg["v2"].get("safety", {})
    o["safety"] = over.get("safety", True)
    tb = res.tablebase
    factory = None
    if with_search:
        ckpt = res.cfg["model"]["checkpoint"] if ck == "new" else res.cfg["v2"]["v1_checkpoint"]
        factory = make_search_factory(res.value_fn_for(ckpt), res.cfg, tb, improved)
    return V2Agent(res.laya, factory, o, tb, name=name)


def build_agent(name: str, res: Resources, seed: int = 0) -> Any:
    if name in V2_AGENTS:
        return build_v2_agent(name, res)
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
