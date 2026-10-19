# v2 protocol: fixing the shortcomings diagnosed in v1

Written and committed **before** any v2 run on the fresh sets. v1 (docs/EXPERIMENT_PROTOCOL.md) stays the main
pre-registered study and is reported unchanged; v2 is a follow-up that asks whether fixing the weaknesses v1
exposed makes the agent stronger, and which fix matters.

## 1. Shortcomings found in v1 and the v2 change for each

| # | v1 evidence | v2 change (module) |
|---|---|---|
| 1 | Laya blunders (CPL >= 200) on 17.3% of test positions; search only lowers it to 16.6% | Blunder check before every move: candidates = top-6 preferred moves + all captures, scored by a 2-ply full-width + quiescence material search; play the most preferred move within 150 cp of the best (`tactics.safe_choice`) |
| 2 | Search adds nothing on puzzles; leaves are evaluated mid-exchange | Quiescence correction of the value at every leaf (`tactics.quiescence_shift`, `adjust_value`) |
| 3 | Interior search nodes use uniform priors | Tactical priors at interior nodes: captures (MVV-LVA), promotions, direct checks; en prise moves discouraged (`tactics.tactical_prior`) |
| 4 | Confidence does not predict where search helps (Spearman 0.01); gains are in sharp positions and endgames | Criticality gate: search if in check, a piece is hanging, a profitable capture exists, <= 12 legal moves, or endgame; 400 simulations when sharp, 200 otherwise (`tactics.criticality`) |
| 5 | Value net: constant output on 9.2% of test positions (dead 256-unit ReLU layer), Spearman 0.61 with Stockfish | Retrained: LeakyReLU in the head, weight decay 1e-4, labels blended 50/50 game result / tanh(Stockfish cp / 400) at 10k nodes (`label_sf`, `make_v2_config`) |
| 6 | Search helps most in endgames, where the net is weakest | Syzygy tablebases (<= 5 pieces) at the root and as exact search leaves; mate-in-one shortcut |
| 7 | 74% of round-robin games end in threefold repetition, often from better positions | Contempt: when Laya's best win chance >= 0.55, a repeating move is scored as a draw worth -0.10 and replaced by a non-repeating move within 0.06 win chance |
| 8 | 103 of 500 v1 games duplicated another game | 50 fresh openings (100 games per pairing) |
| 9 | The v1 test set is no longer blind | Fresh positions and openings from 2025-03, a month no v1 split used; all development on v1 validation positions / v1 openings only |

## 2. Agents

`laya_only` (v1 System 1, reference), `v2_safe` (Laya + blunder check + tablebases + mate-in-one + contempt, no
search), `v2_full` (v2_safe + critical gate + improved search + new net), `v2_always` (search on every move).
Ablations: `v2_full_oldnet` (v1 value net), `v2_full_plainsearch` (v1 search: uniform interior priors, no
quiescence), `v2_full_nosafety` (no blunder check), `v2_full_nocontempt`, `v2_full_notb`.

## 3. Development (validation positions and v1 openings only), then freeze

1. Value net: accept the new checkpoint only if `check_value_net` shows dead-head rate < 1% and validation
   Spearman with Stockfish higher than the v1 checkpoint's on the same set. Otherwise keep the v1 net and say so.
2. One pre-listed choice, decided on the first 300 validation positions with `scripts.v2_dev`: the root prior
   temperature of the v2 search, 0.01 (the v1-tuned value) vs 0.1. Pick the lower ACPL for `v2_full`; ties
   (within 1 cp) keep 0.01. No other parameter is tuned.
3. A practice match (`v2_full` vs `sf_elo_1320`, 10 v1 openings) checks for errors and measures seconds/game.
4. Commit `configs/v2.yaml` (+ this file, + DEVIATIONS entry) = **freeze**. Only then build/run the fresh sets.

## 4. Final runs (fresh sets, run once)

- **Study A-v2:** 2000 fresh positions; agents laya_only, v2_safe, v2_full, v2_always; judged as in v1
  (Stockfish, 200k nodes, MultiPV 2). Single process (timings are reported).
- **Tournament-v2:** each of laya_only, v2_safe, v2_full, v2_full_oldnet, v2_full_nosafety vs `sf_elo_1320`,
  50 fresh openings x 2 colours (100 games each); plus v2_full vs laya_only head-to-head (100 games).
  May run as parallel processes (moves do not depend on time).

## 5. Pre-specified questions and analysis

- **Q1 (main):** does v2_full score higher than laya_only against sf_elo_1320? Score difference with a
  bootstrap CI over openings; Elo from the anchored Bradley-Terry fit (scripts.fit_elo).
- **Q2:** does the blunder check reduce blunders? v2_safe vs laya_only blunder rate and ACPL on Study A-v2,
  paired bootstrap 95% CI.
- **Q3:** does the improved search add anything on top of the safety layers? v2_full vs v2_safe (Study A-v2 and
  tournament).
- **Q4 (ablations):** v2_full vs v2_full_oldnet (value net), vs v2_full_nosafety (blunder check).
- Report every result, including nulls and reversals. Descriptive: search rate, override rate, s/move,
  terminations (repetition share vs v1's 74%).
- The target of "1000-1100 Elo" is a hope, not a hypothesis; Elo stays relative to the 1320 anchor.

## 6. Known limitations carried over

Elo is relative to one weak anchor; LayaChess's training data overlap with Lichess positions is unknown; the
safety layer is material-only (it can reject sound sacrifices); v2 development used positions already used in v1
validation.
