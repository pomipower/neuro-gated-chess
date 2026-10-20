"""v3: train the learned search gate on VALIDATION positions, or evaluate a frozen gate on TEST positions.

Data per position (from run_position_benchmark with agents v2_safe,v2_always, and scripts.gate_features):
    gain = CPL(v2_safe) - CPL(v2_always)      # > 0: searching would have improved the move
Two pre-listed linear models, compared by 5-fold cross-validation on validation (grouped by game):
    logistic : P(gain > 0)
    ridge    : E[gain] (gain clipped to +/-300 cp)
The model with the lower cross-validated composed ACPL at a 25% search rate is kept, refit on all validation
positions, and saved to configs/gate_learned.json with score thresholds for search rates 10/25/60%.

Composed ACPL at rate r: search (take v2_always's move) on the r share of positions with the highest score,
otherwise v2_safe's move. Compared with: random gate (expected), critical gate (v2), oracle (upper bound).

    python -m scripts.train_gate --config configs/v2.yaml --bench-run v3_val --features-run v3_val
    python -m scripts.train_gate --config configs/v3.yaml --eval --bench-run v3_test --features-run v3_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold

from src.engine.gate_features import FEATURES
from src.utils import load_config, resolve

RATES = (10, 25, 60)
GATE_PATH = Path("configs/gate_learned.json")
N_BOOT = 4000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/v2.yaml")
    p.add_argument("--bench-run", required=True)
    p.add_argument("--features-run", required=True)
    p.add_argument("--split", default=None, help="positions split for game ids (default: val, or test with --eval)")
    p.add_argument("--eval", action="store_true", help="evaluate the frozen configs/gate_learned.json")
    p.add_argument("--out", default="paper/figures")
    return p.parse_args()


def load(cfg: dict, bench_run: str, feat_run: str, split: str) -> pd.DataFrame:
    base = resolve(cfg["results_dir"])
    bm = pd.read_csv(base / bench_run / "benchmark_moves.csv")
    w = bm.pivot_table(index="pos_id", columns="agent", values="cpl")
    f = pd.read_csv(base / feat_run / "gate_features.csv").set_index("pos_id")
    d = f.join(w[["v2_safe", "v2_always"]], how="inner").dropna(subset=["v2_safe", "v2_always"])
    pos = pd.read_csv(resolve(cfg["data"]["benchmark_dir"]) / f"positions_{split}.csv").set_index("pos_id")
    d = d.join(pos[["game_id"]], how="left")
    d["game_id"] = d.game_id.fillna(pd.Series(d.index, index=d.index))
    d["gain"] = d.v2_safe - d.v2_always
    d["critical"] = (d[["in_check", "hanging", "opportunity", "few_moves", "endgame"]].sum(axis=1) > 0)
    return d


def composed(d: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, d.v2_always.to_numpy(float), d.v2_safe.to_numpy(float))


def top_mask(score: np.ndarray, rate: float) -> np.ndarray:
    k = int(round(rate * len(score)))
    m = np.zeros(len(score), bool)
    m[np.argsort(-score, kind="stable")[:k]] = True
    return m


def fit(kind: str, X: np.ndarray, gain: np.ndarray):
    if kind == "logistic":
        m = LogisticRegression(C=1.0, max_iter=2000).fit(X, (gain > 0).astype(int))
        return m.coef_.ravel(), float(m.intercept_[0])
    m = Ridge(alpha=10.0).fit(X, np.clip(gain, -300, 300))
    return m.coef_.ravel(), float(m.intercept_)


def cv_acpl(d: pd.DataFrame, kind: str, rate: float) -> float:
    X = d[FEATURES].to_numpy(float)
    score = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(X, groups=d.game_id):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        coef, b = fit(kind, (X[tr] - mu) / sd, d.gain.to_numpy()[tr])
        score[te] = ((X[te] - mu) / sd) @ coef + b
    return composed(d, top_mask(score, rate)).mean()


def train(cfg: dict, d: pd.DataFrame, out: Path) -> None:
    print(f"validation positions: {len(d)}; search improves the move on {(d.gain > 0).mean():.1%}, "
          f"worsens on {(d.gain < 0).mean():.1%}")
    rows = []
    for kind in ("logistic", "ridge"):
        for r in RATES:
            rows.append({"model": kind, "rate_pct": r, "cv_acpl": cv_acpl(d, kind, r / 100)})
    cv = pd.DataFrame(rows)
    print("\n5-fold CV (grouped by game) composed ACPL\n", cv.pivot(index="model", columns="rate_pct", values="cv_acpl").round(2))
    best = cv[cv.rate_pct == 25].sort_values("cv_acpl").iloc[0].model
    X = d[FEATURES].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    coef, b = fit(best, (X - mu) / sd, d.gain.to_numpy())
    score = ((X - mu) / sd) @ coef + b
    thresholds = {str(r): float(np.quantile(score, 1 - r / 100)) for r in RATES}
    GATE_PATH.write_text(json.dumps({"model": best, "features": FEATURES, "mean": mu.tolist(), "std": sd.tolist(),
                                     "coef": coef.tolist(), "intercept": b, "thresholds": thresholds,
                                     "trained_on": "validation", "n": len(d)}, indent=2))
    print(f"\nchosen model: {best}; saved {GATE_PATH}")
    print("coefficients (standardized):")
    for f, c in sorted(zip(FEATURES, coef), key=lambda t: -abs(t[1])):
        print(f"  {f:22s} {c:+.3f}")
    cv.to_csv(out / "V3_gate_cv.csv", index=False)


def evaluate(cfg: dict, d: pd.DataFrame, out: Path) -> None:
    g = json.loads(GATE_PATH.read_text())
    X = d[g["features"]].to_numpy(float)
    score = ((X - np.array(g["mean"])) / np.array(g["std"])) @ np.array(g["coef"]) + g["intercept"]
    rng = np.random.default_rng(0)
    rows = []
    safe, alw = d.v2_safe.to_numpy(float), d.v2_always.to_numpy(float)
    gain = safe - alw
    crit = d.critical.to_numpy()
    for r in RATES:
        learned = score >= g["thresholds"][str(r)]
        achieved = learned.mean()
        lc = composed(d, learned)
        stats = np.empty(N_BOOT)
        n = len(d)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, n)
            stats[i] = composed(d, rng.random(n) < achieved)[idx].mean() - lc[idx].mean()
        lo, hi = np.percentile(stats, [2.5, 97.5])
        oracle = composed(d, top_mask(gain, achieved))
        rows.append({"rate_pct": r, "achieved_rate": achieved, "learned_acpl": lc.mean(),
                     "random_acpl_expected": safe.mean() - achieved * gain.mean(),
                     "random_minus_learned": stats.mean(), "ci_lo": lo, "ci_hi": hi,
                     "p_one_sided": float((stats <= 0).mean()), "oracle_acpl": oracle.mean()})
    t = pd.DataFrame(rows)
    print(f"TEST evaluation of the frozen gate ({g['model']}), n={len(d)}")
    print(f"never {safe.mean():.2f}  always {alw.mean():.2f}  critical gate ({crit.mean():.1%}) {composed(d, crit).mean():.2f}")
    print(t.round(3).to_string(index=False))
    print("random_minus_learned > 0 with CI above 0 = the learned gate beats a random gate at the same rate")
    t.to_csv(out / "V3_gate_test.csv", index=False)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(cfg, args.bench_run, args.features_run, args.split or ("test" if args.eval else "val"))
    (evaluate if args.eval else train)(cfg, d, out)


if __name__ == "__main__":
    main()
