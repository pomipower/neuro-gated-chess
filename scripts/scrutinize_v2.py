"""Scrutiny of v2 (post hoc; report as exploratory). CPU only, seconds. Uses existing results only.

1. Gate control: is "search only where critical" better than searching a RANDOM 60% of positions?
   Composed offline from Study A-v2 rows: on each position take v2_always's move if the gate says search,
   else v2_safe's move (v2_full == this composition with the critical gate; checked and reported).
   Random gates at the same rate (500 seeds); paired bootstrap of ACPL(random) - ACPL(critical).
   Also the oracle gate (search only where it helps) at the same rate, as an upper bound.
2. Tail check: mean vs median CPL, good-move rate (CPL <= 25), ACPL with CPL capped at 300.
3. Colour breakdown of every tournament agent's score (as White / as Black) and terminations.

Run from the repo root:
    python -m scripts.scrutinize_v2 --config configs/v2.yaml --bench-run v2_bench --tour-runs v2_tour_a,v2_tour_b
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.utils import load_config, resolve

N_BOOT, N_SEEDS = 4000, 500


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/v2.yaml")
    p.add_argument("--bench-run", default="v2_bench")
    p.add_argument("--tour-runs", default="v2_tour_a,v2_tour_b")
    p.add_argument("--out", default="paper/figures")
    return p.parse_args()


def gate_control(bm: pd.DataFrame, out) -> None:
    w = {a: g.set_index("pos_id") for a, g in bm.groupby("agent")}
    need = {"v2_safe", "v2_always", "v2_full"}
    if not need <= set(w):
        print(f"gate control needs agents {need}; have {sorted(w)}")
        return
    ids = w["v2_safe"].index.intersection(w["v2_always"].index).intersection(w["v2_full"].index)
    safe, alw, full = (w[a].loc[ids] for a in ("v2_safe", "v2_always", "v2_full"))
    c_safe, c_alw = safe.cpl.to_numpy(float), alw.cpl.to_numpy(float)
    crit = pd.to_numeric(full.critical, errors="coerce").fillna(0).to_numpy() > 0
    rate = crit.mean()
    comp = np.where(crit, c_alw, c_safe)
    same = np.mean(np.where(crit, alw.move_uci.to_numpy(), safe.move_uci.to_numpy()) == full.move_uci.to_numpy())
    print(f"== 1. Gate control (n={len(ids)}, critical gate searches {rate:.1%})")
    print(f"composition check: composed critical-gate move == actual v2_full move on {same:.1%} of positions;"
          f" ACPL composed {comp.mean():.2f} vs actual v2_full {full.cpl.mean():.2f}")
    rng = np.random.default_rng(0)
    rand_means = np.array([np.where(rng.random(len(ids)) < rate, c_alw, c_safe).mean() for _ in range(N_SEEDS)])
    # paired bootstrap: fresh random gate per resample, same position resample for both
    stats = np.empty(N_BOOT)
    n = len(ids)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        mask = rng.random(n) < rate
        stats[i] = np.where(mask, c_alw, c_safe)[idx].mean() - comp[idx].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    gain = c_safe - c_alw
    k = int(round(rate * n))
    oracle = c_safe.copy()
    top = np.argsort(-gain, kind="stable")[:k]
    oracle[top] = np.minimum(c_safe[top], c_alw[top])
    rows = [
        {"gate": "never (v2_safe)", "search_rate": 0.0, "acpl": c_safe.mean()},
        {"gate": "critical (v2_full, composed)", "search_rate": rate, "acpl": comp.mean()},
        {"gate": "random, same rate (mean of 500)", "search_rate": rate, "acpl": rand_means.mean()},
        {"gate": "oracle, same rate (upper bound)", "search_rate": rate, "acpl": oracle.mean()},
        {"gate": "always (v2_always)", "search_rate": 1.0, "acpl": c_alw.mean()},
    ]
    t = pd.DataFrame(rows)
    t.to_csv(out / "V2_gate_control.csv", index=False)
    print(t.round(3).to_string(index=False))
    print(f"random minus critical ACPL: {stats.mean():+.2f} cp, 95% CI [{lo:+.2f}, {hi:+.2f}]  "
          f"(> 0 means the critical gate beats random)")
    gc, gn = gain[crit], gain[~crit]
    print(f"mean search gain where the gate fires: {gc.mean():+.2f} cp (n={crit.sum()}); "
          f"where it does not: {gn.mean():+.2f} cp (n={(~crit).sum()})")


def tail_check(bm: pd.DataFrame, out) -> None:
    rows = []
    for a, g in bm.groupby("agent"):
        c = g.cpl.to_numpy(float)
        rows.append({"agent": a, "n": len(c), "acpl": c.mean(), "median_cpl": np.median(c),
                     "good_rate(cpl<=25)": (c <= 25).mean(), "acpl_capped300": np.minimum(c, 300).mean(),
                     "share_of_total_cpl_from_blunders(cpl>=200)": c[c >= 200].sum() / max(c.sum(), 1e-9)})
    t = pd.DataFrame(rows).sort_values("acpl")
    t.to_csv(out / "V2_tail_check.csv", index=False)
    print("\n== 2. Tail check (does v2 improve typical moves or only remove disasters?)")
    print(t.round(3).to_string(index=False))


def colour(cfg: dict, runs: list[str], out) -> None:
    base = resolve(cfg["results_dir"])
    g = pd.concat([pd.read_csv(base / r / "games.csv") for r in runs], ignore_index=True)
    v = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
    rows = []
    for (w, b), d in g.groupby(["white", "black"]):
        for res, term in zip(d.result, d.termination):
            rows.append({"agent": w, "opp": b, "colour": "white", "score": v[res], "term": term})
            rows.append({"agent": b, "opp": w, "colour": "black", "score": 1 - v[res], "term": term})
    df = pd.DataFrame(rows)
    df = df[df.agent != "sf_elo_1320"]
    t = df.pivot_table(index=["agent", "opp"], columns="colour", values="score", aggfunc=["mean", "size"])
    t.to_csv(out / "V2_colour.csv")
    print("\n== 3. Score by colour")
    print(t.round(3).to_string())
    wdl = df.assign(res=np.select([df.score == 1, df.score == 0], ["win", "loss"], "draw"))
    print("\nresults by colour (all agents vs sf_elo_1320):")
    print(wdl[wdl.opp == "sf_elo_1320"].groupby(["agent", "colour"]).res.value_counts().unstack(fill_value=0).to_string())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bm = pd.read_csv(resolve(cfg["results_dir"]) / args.bench_run / "benchmark_moves.csv")
    print("POST HOC SCRUTINY (exploratory)\n")
    gate_control(bm, out)
    tail_check(bm, out)
    colour(cfg, [r.strip() for r in args.tour_runs.split(",")], out)


if __name__ == "__main__":
    main()
