# Experiment Protocol (pre-registered)

**Commit this file before running any experiment.** Anything you change after seeing results goes in `docs/DEVIATIONS.md` with a date and a reason, and is reported in the paper. Negative results are acceptable and reportable; the protocol below is written so that every outcome is publishable.

---

## 1. Research question

Dual-process chess agents pair a fast, intuitive policy (System 1) with slow search (System 2). **Is a System 1 model's own confidence a good enough signal for deciding *when* to invoke System 2, and does that gating buy a better quality-per-compute trade-off than the obvious alternatives?**

We do **not** claim Stockfish-level play. Stockfish is used as (a) the *judge* that scores every move and (b) a *calibrated rating anchor*.

### Agents (names are used verbatim in configs and CSVs)

| Name | System 1 (Laya) | System 2 (MCTS + value CNN) | Role |
|---|---|---|---|
| `laya_only` | always | never | pure intuition |
| `mcts_uniform` | not used | always, uniform root prior | pure calculation by our own search (ablation: no Laya at all) |
| `always_s2` | scores root | always, **Laya root prior** | upper compute bound of the hybrid family |
| `hybrid_s2_{10,25,50,75}` | scores root | if confidence ≤ τ, else S1 move; τ set so that ≈ that % of validation positions go to S2 | **the proposed method** |
| `random_s2_{25,50}` | scores root | sent to S2 by a seeded coin flip at the same rate | **control: is the *confidence* informative, or does any gating work?** |
| `sf_elo_{…}` | — | Stockfish with `UCI_LimitStrength`, `UCI_Elo` = N | rating anchors (Study B only) |

`laya_only` and `always_s2` are the two ends of the hybrid family (τ = 0 and τ = ∞), so the sweep is a continuum.

The hybrid decision rule is exactly: **play the S1 move if `confidence > τ`, otherwise run S2.** S2 receives Laya's root scores as its prior, so the S1 computation is never wasted.

### Hypotheses

- **H1 (informative gate).** Laya's confidence predicts the quality of Laya's own move: AUROC for "good move" (CPL ≤ 25 cp) is > 0.5 with a 95% CI that excludes 0.5, and exceeds the AUROC of the trivial baseline "fewer legal moves = more confident".
- **H2 (beats random gating).** At matched S2 rates (25% and 50%), mean centipawn loss (ACPL) of `hybrid_s2_*` is lower than `random_s2_*`. One-sided paired bootstrap over positions, α = 0.05, Holm-corrected over the two rates.
- **H3 (trade-off).** Define recovery R = (ACPL_laya − ACPL_hybrid) / (ACPL_laya − ACPL_always_s2). At S2 rate 50%, R ≥ 0.5 while using ≤ 50% of `always_s2`'s S2 compute. *Only meaningful if `always_s2` has lower ACPL than `laya_only`; if it does not, report that and skip H3.*
- **H4 (errors concentrate where confidence is low).** `laya_only`'s blunder rate (CPL ≥ 200 cp) in the lowest-confidence quartile exceeds the highest-confidence quartile; report the rate ratio with a bootstrap CI.
- **H5 (tactical sensitivity).** The S2 trigger rate on tactical puzzle positions (Study C) exceeds that on ordinary game positions (Study A).
- **H6 (playing strength, confirmatory and underpowered).** `hybrid_s2_50` has a higher Elo than `laya_only`. Report the point estimate and CI; make no claim if the CI overlaps.

If the value network turns out too weak for `always_s2` to beat `laya_only` (a real possibility with a ~2M-parameter net), the paper becomes an analysis of *why* and what gating still tells us (H1, H4, H5 do not depend on S2 being good).

---

## 2. Study A: position benchmark (main quantitative results)

Cheap, large-N, statistically powerful. No games are played.

**Data.** Three disjoint splits, separated by *month* of Lichess Elite games (no game or position can leak across splits):

| Split | Use | Size |
|---|---|---|
| train months | value-net training | ~500k positions |
| `val` month | value-net early stopping, **gate selection** (§2.3) | 20k positions (net) / 3,000 positions (gate) |
| `test` month | **all reported Study A numbers** | 3,000 positions (scale up if time allows) |

Source: Lichess Elite Database (https://database.nikonoel.fr/, games by 2400+ vs 2200+ players, no bullet; maintained by nikonoel). Use the most recent available months for `val` and `test` to reduce overlap with whatever data the Laya checkpoint was trained on (LayaChess was fine-tuned on ChessBench, which is built from Lichess data; **check and state this overlap risk in the paper's limitations**).

**Position sampling (test and val).** ≤ 3 positions per game, ply ∈ [10, 120], game result `Termination = Normal`. Stratify: ~20% opening (ply ≤ 20), ~20% endgame (combined N/B/R/Q value of both sides ≤ 20, with N=B=3, R=5, Q=9), rest middlegame. Fixed seed. Position lists are committed to `data/benchmark/` so everyone evaluates the same positions.

### 2.1 Judge and metrics

Stockfish `StockfishJudge`: `Threads=1`, `Hash=128`, `MultiPV=2`, `go nodes N_eval` (default 200,000, deterministic for a fixed binary and a single thread). Record the Stockfish version.

For position P and agent move m (all scores from the side to move's point of view, mate scores mapped to ±1000 cp, all scores clipped to ±1000):

- `s*` = Stockfish score of P (best line); `s(m)` = −(score of the position after m); if m ends the game, checkmate = +1000, stalemate = 0.
- **CPL(m) = max(0, s* − s(m))**; if m equals Stockfish's best move, CPL = 0.
- **ACPL** = mean CPL. **Blunder** = CPL ≥ 200; **mistake** = 100 ≤ CPL < 200; **good move** = CPL ≤ 25. Also report top-1 agreement with Stockfish.
- **Forcedness** `sf_gap_cp` = score(best) − score(second best) from MultiPV=2 (large gap = "only move" positions).

### 2.2 Cost accounting (the "compute" axis)

Per move, log: `wall_ms` (primary; `time.perf_counter()` with `torch.cuda.synchronize()` before stopping the clock), `laya_calls`, `laya_moves_scored`, `cnn_evals`, `mcts_sims`. Report both **wall time** and the **counts** — counting only MCTS nodes would hide that one Laya move costs a 421M-parameter inference per legal move. If feasible, also derive an analytic `est_gflops` (document the formula). Run everything on one machine, log its hardware, and fix the device across agents.

### 2.3 Composition trick (why Study A is cheap)

The gate depends only on the S1 output, and S2 does not depend on the gate. So run each component **once per position** (`laya_only`, `always_s2`, `mcts_uniform`) and **compose any hybrid offline**: hybrid move = S1 move if `conf > τ` else the `always_s2` move; hybrid cost = Laya cost + (S2 cost if triggered). This lets us sweep τ continuously and draw every random-gate control with many seeds at no extra search cost. Composed cost uses the sum of measured component times; state that as an approximation. (Study B plays the real agents.)

### 2.4 Gate selection (done on `val`, frozen before touching `test`)

1. Compute Laya scores on `val`. Candidate confidence functions: `margin` (top-1 minus top-2 win chance), `top1_softmax` (T ∈ {0.05, 0.1, 0.2, 0.5} over win-chance logits), `norm_entropy` (same T grid).
2. Choose the one with the highest AUROC for "good move" on `val`; if the best two are within 0.01 AUROC, take the simpler (`margin`).
3. For each target S2 rate q ∈ {0.10, 0.25, 0.50, 0.75} set τ_q = q-quantile of `val` confidences. Report the S2 rate actually achieved on `test`.
4. Write the chosen function, T, and τ_q to `configs/gate_frozen.yaml` and commit it.

### 2.5 Analyses and figures

| Figure | Content | Hypothesis |
|---|---|---|
| F1 | Reliability diagram (confidence bins vs fraction of good moves) + ECE | H1 |
| F2 | ROC of confidence vs trivial baselines (legal-move count, random) | H1 |
| F3 | ACPL vs mean cost per move: all agents, hybrids as a curve, random-gate controls as points | H2, H3 |
| F4 | Blunder rate by confidence quartile; by game phase × agent | H4 |
| F5 | S2 trigger rate by phase, forcedness bin, and puzzle vs game positions | H5 |
| F6 | MCTS depth distribution when S2 fires (`max_depth`, `mean_leaf_depth`) | descriptive |

All intervals: 95% percentile bootstrap, 2,000 resamples, resampling **positions** (paired across agents). Per-game dependence is limited by taking ≤ 3 positions per game.

---

## 3. Study B: tournament (confirmatory)

**Design.** Games between real agents with fixed openings and swapped colors.

- **Openings:** 25 distinct 8-ply openings sampled from the `test` month (fixed seed), kept if Stockfish's eval after the 8 plies is within ±100 cp. Each opening is played twice with colors swapped, so a pairing is 50 games. Because our agents are deterministic, this is what prevents duplicate games.
- **Termination:** `board.outcome(claim_draw=True)`; cap at 300 plies = draw; an exception or illegal move = loss for that agent (logged).
- **No wall-clock limit** for our agents (their budget is simulations); Stockfish anchors use `go movetime 100`.
- **Anchors:** `sf_elo_N` for three or four values of N chosen by a **pilot** (10 games of `laya_only` and `always_s2` against N ∈ {1320, 1500, 1700, 1900, 2100}) so that our agents score between ~25% and ~75% against at least two anchors. Stockfish documents its `UCI_Elo` as calibrated at 120s+1s time control, anchored to CCRL 40/4, so at our time budget the anchor ratings are **nominal**; call the result "Elo on the Stockfish-anchor scale", not a CCRL rating.

**Tiers (by priority, pick what the compute allows):**

1. *Required.* Gauntlet: {`laya_only`, `hybrid_s2_50`, `always_s2`, `mcts_uniform`} × 3 anchors × 50 games = 600 games.
2. Round-robin among our agents (6 pairings × 50 = 300 games).
3. Extra hybrids (`hybrid_s2_25`, `hybrid_s2_75`) against the same anchors.

Measure seconds per game in the pilot and **write the estimate into `DEVIATIONS.md` before launching**; do not guess.

**Rating model.** Bradley–Terry with draws counted as half-wins (pseudo-likelihood, solved with `scipy.optimize`), anchors fixed at their nominal Elo, everything else estimated. 95% CI by **cluster bootstrap over openings** (resample openings with replacement, keep both colors together), 1,000 replicates. An agent that scores > 90% or < 10% against every anchor is flagged "off-scale". With 50 games per pairing expect CIs of roughly ±60–100 Elo: Study B can corroborate Study A but cannot, by itself, support small effects. Say so in the paper.

**Per-move log for games** (`moves.csv`) is needed for phase-level S1/S2 usage and blunders. Blunders in games are labelled by the judge from the game's positions (Study A's definitions).

---

## 4. Study C: tactical puzzles (optional, cheap)

- 1,000 puzzles from the Lichess puzzle database (`lichess_db_puzzle.csv.zst`; Lichess database exports are CC0), 200 per rating bucket: <1200, 1200–1599, 1600–1999, 2000–2399, ≥2400.
- Position = the puzzle FEN after applying the first move in `Moves` (the opponent's move); solved if the agent plays the next move in `Moves`. (Also accept any move that gives checkmate.)
- Report solve rate per agent by rating bucket, and the S2 trigger rate vs Study A (H5). Check the model card/paper for overlap between LayaChess's training data and Lichess puzzles and state it as a caveat.

---

## 5. Value-network checks (go/no-go for build day 1)

| Check | Pass condition |
|---|---|
| Validation MSE | beats the constant-predictor baseline MSE, and both are reported |
| Sign accuracy | reported **excluding draws** (a draw has no sign) |
| Spearman(value, tanh(SF cp / 400)) on `val` positions | report; **if < 0.3, switch `label.mode` to `blend`** (outcome + Stockfish-eval labels) and retrain |
| Mate-in-1 unit test (`tests/test_mcts.py`) | must pass |
| Lichess puzzle theme `mateIn2`, 100 puzzles | report solve rate |

---

## 6. Reproducibility

- One `configs/default.yaml` holds every hyperparameter; each run copies it plus `env.json` (python/torch/chess versions, Stockfish version, Laya HF `revision`, git SHA, hardware) into `data/results/<run_id>/`.
- Seeds: python, numpy, torch, plus the data-sampling seed in the config. MCTS uses no Dirichlet noise and breaks ties deterministically (visits, then Q, then prior).
- Result CSVs are **committed** (they are small); raw PGN, processed training data and `third_party/` are not. Trained weights (`best_model.pt`) are shared by release/drive; record the file's SHA-256 in `docs/ARTIFACTS.md`.
- Pin the Laya checkpoint with a Hugging Face `revision` hash once chosen.

## 7. Threats to validity (put in the paper)

- Elite-game positions and a single-engine judge (Stockfish at fixed nodes) define "quality"; CPL is a proxy for strength.
- Possible train/test overlap between LayaChess's training data and Lichess games/puzzles.
- Nominal anchor Elo; small game counts in Study B.
- One hardware setup; wall-time numbers will not transfer.
- The value CNN is small and trained on game outcomes (or blended labels); S2 here is a weak searcher by design. Conclusions are about the *gate*, not about beating strong engines.

## 8. Notes added during the build (2026-10-18, before any real data was run)

These were added after reading the LayaChess source and smoke-testing the pipeline, and before any experiment on real data, so they are part of the pre-registration rather than deviations.

1. **System 1 is expensive.** LayaChess runs one 421M-parameter forward pass *per legal move* (~35 per position); its README reports ~14 move scores/s on an Apple M4. Our System 2 (2.3M-parameter value net, 400 simulations) may cost about the same or less. Therefore:
   - "compute" in H3 and F3 is the **measured wall time** (and model-call counts), never an assumption that S1 is cheap;
   - the gate is framed as deciding **when to add search on top of System 1** (the hybrid always pays for Laya). If S2 turns out cheaper than S1, report it: it changes the practical story (the gate saves search time, not total time) but not H1, H2, H4, H5.
2. **Closest software baseline.** The LayaChess repo already contains an MCTS that uses Laya itself for priors *and* leaf values. We do not run it (each node costs a full Laya pass), but the paper must cite it and state the difference: we gate to a separate, cheap searcher, and we measure *when* to search rather than *how*.
3. **Laya output range.** LayaChess win chances are expectations over 10 levels and lie in [0.05, 0.95]. Confidence functions use them as-is; `margin` is in [0, 0.9].
4. **Run sizes come from `scripts/pilot_timing.py`** on the actual machine (RTX 4060 8 GB). Record its output in `docs/DEVIATIONS.md` before launching Study A or B, and shrink Study B tiers (never Study A's test set below 1,500 positions) if needed.
5. **Matched rates.** H2 compares each hybrid to random gating at the S2 rate the hybrid **actually achieved on test** (not the target rate), so the comparison is at equal compute.
