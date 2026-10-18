from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval import compose
from src.eval.elo import bootstrap_ratings, fit_ratings
from src.eval.metrics import auroc, ece, holm


def test_auroc_ece_holm() -> None:
    assert auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert np.isnan(auroc([0.1, 0.2], [True, True]))
    assert ece([0.95] * 100, [True] * 95 + [False] * 5) == pytest.approx(0.0, abs=1e-9)
    assert holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])


def _games(diff_elo: float, n: int, rng: np.random.Generator) -> pd.DataFrame:
    p = 1 / (1 + 10 ** (-diff_elo / 400))
    rows = []
    for i in range(n):
        o = f"o{i % 25:02d}"
        if i % 2 == 0:
            rows.append({"opening_id": o, "white": "agent", "black": "sf_elo_1500",
                         "result": "1-0" if rng.random() < p else "0-1"})
        else:
            rows.append({"opening_id": o, "white": "sf_elo_1500", "black": "agent",
                         "result": "0-1" if rng.random() < p else "1-0"})
    return pd.DataFrame(rows)


def test_elo_recovers_known_difference() -> None:
    g = _games(200, 4000, np.random.default_rng(0))
    r = fit_ratings(g, {"sf_elo_1500": 1500.0})
    assert r["agent"] == pytest.approx(1700, abs=25)
    table = bootstrap_ratings(g[:200], {"sf_elo_1500": 1500.0}, n=50)
    row = table[table.agent == "agent"].iloc[0]
    assert row.ci_low < row.elo < row.ci_high and not row.off_scale


def _bench(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    conf = rng.random(n)
    cpl1 = np.where(conf < 0.3, 150.0, 5.0)              # S1 bad exactly when unconfident
    cpl2 = np.full(n, 20.0)
    rows = []
    for i in range(n):
        base = {"pos_id": f"t{i}", "phase": "middlegame", "ply": 30, "n_legal": 30, "sf_gap_cp": 10.0,
                "laya_moves_scored": 30, "cnn_evals": 0, "mcts_sims": 0, "max_depth": 0, "mean_leaf_depth": 0.0,
                "search_ms": 0.0}
        rows.append(base | {"agent": "laya_only", "move_uci": "e2e4", "confidence": conf[i], "cpl": cpl1[i],
                            "is_best": cpl1[i] == 5.0, "wall_ms": 100.0})
        rows.append(base | {"agent": "always_s2", "move_uci": "d2d4", "confidence": conf[i], "cpl": cpl2[i],
                            "is_best": False, "wall_ms": 600.0, "search_ms": 500.0, "cnn_evals": 400,
                            "mcts_sims": 400, "max_depth": 5, "mean_leaf_depth": 3.0})
    return pd.DataFrame(rows)


def test_composition_hybrid_beats_random_when_confidence_is_informative() -> None:
    w = compose.Wide.from_benchmark(_bench())
    hyb = compose.summarize(compose.apply_mask(w, compose.hybrid_mask(w, 0.3)))
    rnd = compose.random_control(w, hyb["s2_rate"], seeds=50)
    assert hyb["acpl"] < rnd.acpl.mean()
    assert hyb["cost_ms"] == pytest.approx(100 + 500 * hyb["s2_rate"])
    test = compose.paired_diff_ci(w, compose.hybrid_mask(w, 0.3),
                                  lambda rng: rng.random(len(w)) < hyb["s2_rate"], n=300)
    assert test["lo"] > 0 and test["p_one_sided"] < 0.05
    assert compose.recovery(50.0, 30.0, 10.0) == pytest.approx(0.5)
    assert np.isnan(compose.recovery(10.0, 30.0, 50.0))


def test_make_all_figures(tmp_path) -> None:
    from src.eval.figures import make_all
    from src.utils import load_config

    bench = tmp_path / "bench.csv"
    _bench().to_csv(bench, index=False)
    gate = {"tau": {"0.10": 0.1, "0.25": 0.25, "0.50": 0.5, "0.75": 0.75}}
    res = make_all(load_config("configs/default.yaml") | {"gate": {"random_control_seeds": 20},
                                                          "benchmark": {"bootstrap_resamples": 100}},
                   gate, bench, tmp_path / "fig")
    pts, tests = res["F3"]
    assert {"laya_only", "always_s2", "hybrid_s2_50", "random_matched_50"} <= set(pts.agent)
    assert set(tests.rate_pct) == {10, 25, 50, 75} and tests.p_holm.notna().sum() == 2
    for name in ("F1_reliability", "F2_roc", "F3_tradeoff", "F4a_blunder_by_confidence", "F4b_blunder_by_phase",
                 "F5_trigger_rate", "F6_depth"):
        assert (tmp_path / "fig" / f"{name}.png").exists() and (tmp_path / "fig" / f"{name}.pdf").exists()
