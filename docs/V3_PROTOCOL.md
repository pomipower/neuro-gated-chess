# v3 protocol: a learned "when to think" gate, a second anchor and a classical baseline

Written before any v3 data exists. v1 and v2 stay as reported; v3 targets the open questions that the v2
scrutiny (scripts/scrutinize_v2.py) left:

| v2 finding | v3 response |
|---|---|
| Critical gate is no better than a random gate at the same 60% rate (+1.1 cp, CI -2.4..+4.7); oracle at 60% = 52.6 vs 68.7 ACPL | **Learned gate**: a linear model trained on validation positions to predict whether search improves the move (src/engine/gate_features.py, scripts/train_gate.py) |
| Elo rests on one weak anchor (Stockfish UCI_Elo 1320) | Two anchors: 1320 and 1600 |
| Unknown whether Laya matters once the safety layer exists | **Classical baseline**: the same safety layer + search driven by a heuristic policy instead of Laya (`classical_full`) |
| v2 fresh month (2025-03) has now been seen | New untouched month for every v3 test set (2025-02 or 2025-01, from Hugging Face shard 23) |

## 1. Learned gate (development on validation only)

- Data: the 3000 v1 validation positions (`data/benchmark_v2/positions_val.csv`), each played by `v2_safe` and
  `v2_always` and judged by Stockfish (same judge as v1/v2); per-position gain = CPL(v2_safe) - CPL(v2_always).
- Features (18, all available before searching): Laya win-chance statistics (best, margins to 2nd/3rd, entropy,
  confidence), legal moves, captures, the five criticality flags, total non-pawn material, material balance,
  ply, whether the blunder check overrides Laya, and two tactical-score summaries.
- Models: logistic (P[gain > 0]) and ridge (E[gain], clipped +/-300). 5-fold cross-validation grouped by game;
  the model with lower composed ACPL at a 25% search rate is kept and refit on all validation positions.
  Score thresholds for 10/25/60% search rates are set on validation. Saved to `configs/gate_learned.json`.
- No other parameter of v2 changes.

## 2. Freeze, then final runs on the untouched month

1. Build v3 sets from the new month with `scripts.build_fresh_sets --id-prefix g --exclude-months 2025-03
   --exclude-benchmarks data/benchmark_v2` into `data/benchmark_v3` (2000 positions, 50 openings).
2. Commit `configs/v3.yaml`, `configs/gate_learned.json`, this file, DEVIATIONS -> tag `v3-frozen`.
3. **Study A-v3**: 2000 new positions; agents laya_only, v2_safe, v2_always, v2_full; plus gate features.
   The learned gate is evaluated by exact offline composition (v2_safe move unless the gate fires, then
   v2_always's), validated by the identity v2_full == composition(critical gate) (100.0% on v2 data).
4. **Tournament-v3**: laya_only, v2_full, v3_learned_25, classical_full, each vs sf_elo_1320 and sf_elo_1600,
   50 new openings x 2 colours (100 games per pairing, 800 games).

## 3. Pre-specified questions

- **Q1 (primary, confirmatory)**: does the learned gate beat a random gate at the same search rate?
  ACPL(random) - ACPL(learned) on Study A-v3 at 10/25/60%, paired bootstrap (fresh random gate per resample),
  one-sided p, Holm over the three rates. Supported if the 25% comparison is significant after Holm.
- **Q2**: strength at lower cost: Elo of v3_learned_25 vs v2_full (the learned gate searches ~25% of moves vs
  ~60%). Descriptive with CI; "no meaningful loss" if the CI of the difference excludes -50 Elo.
- **Q3**: does Laya matter? Elo of v2_full vs classical_full.
- **Q4 (replication)**: v2_full vs laya_only on a second fresh month and with two anchors.
- Everything else is exploratory. All results are reported, including nulls.

## 4. Known limitations carried over

Anchors are limited-strength Stockfish, not humans; the safety layer is material-only and shallow; LayaChess
training-data overlap with Lichess is unknown (revision now pinned: 9269e61202b2c3517fef1cdacea83c3c23b92961).
