# Deviations from the pre-registered protocol

Log every change to `docs/EXPERIMENT_PROTOCOL.md` or `configs/` made after the protocol was committed. Each entry is reported in the paper.

| Date | Who | What changed | Why | Seen test results before deciding? |
|---|---|---|---|---|
| | | | | |

Also record here: pilot timings (seconds per position per agent, seconds per game), the chosen anchor Elos, the final train/val/test months, and the Laya revision hash.
| 2026-10-18 | Rohan | Value net: color_flip_augmentation true, weight_decay 1e-3, 10 epochs | Run 1 (defaults) overfit after epoch 5: best val MSE 0.703, Spearman 0.574. Run 2: best val MSE 0.686 (baseline 0.811), Spearman 0.614. Kept run 2. | No (validation data only) |

Pilot timing (2026-10-18, RTX 4060 Laptop, WSL2, torch 2.14.1+cu130, Stockfish 16, n=20 val positions):
laya_only 392 ms/move (81 move scores/s); always_s2 1343 ms (laya 309 + search 1034); mcts_uniform 1092 ms. 400 simulations kept; full study sizes kept.
| 2026-10-18 | Rohan | mcts.prior_temperature 0.3 -> 0.01 | Pilot: always_s2 lost 16/16 to SF anchors. Val sweep (150 val positions, 2025-10) ACPL: laya_only 108, mcts_uniform 258, always_s2 T=0.3/0.1/0.05/0.03/0.01 = 136/106/102/100/98. Chose lowest val ACPL. Gate thresholds unaffected (depend on Laya only). | No (validation only) |
| 2026-10-18 | Rohan | Tournament anchors [1500,1700,1900] -> [1320]; add tier-2 round-robin among our agents | Pilot: laya_only scored 12.5% even vs UCI_Elo 1320 (Stockfish's minimum). Higher anchors give no information. Absolute Elo reported with one anchor (wide CI); agent comparison from the round-robin. | No (test-month openings used only to choose anchors) |

## 2026-10-19 Tier 2 and exploratory analysis
- Tier 2 round-robin was run as two parallel processes (tour_t2a, tour_t2b; disjoint pairings). Agents use fixed simulation counts, so moves are unaffected; tournament timings are not used in any analysis.
- Elo pooled over tour_1 + tour_t2a + tour_t2b (500 games).
- scripts/explore_study_a.py added AFTER test results were seen: all EXP_* outputs are exploratory/post hoc and are reported as such. The frozen gate and all pre-registered analyses are unchanged.
- 223/300 tier-2 games ended in threefold repetition (deterministic agents, no repetition avoidance); reported as a limitation.

## 2026-10-19 v2 value net acceptance (before any fresh-set run)
- Pre-specified rule (V2_PROTOCOL §3.1): dead-head rate < 1% AND val Spearman > v1. Result: Spearman 0.665 vs 0.619 (met);
  dead-head 1.3% (not met). The dead-head metric was meant to detect constant outputs; with the v2 LeakyReLU head an
  all-negative hidden layer no longer gives a constant output, and the constant-output rate is 0.0% (v1: 8.1%).
  Decision: use the v2 net; the v1 net is evaluated anyway as the pre-planned ablation v2_full_oldnet.
- v2 dev choice (V2_PROTOCOL §3.2), first 300 val positions: v2_full ACPL 75.3 (prior T=0.01) vs 68.8 (T=0.1) -> T=0.1.
  Val (optimistic, also used for the choice): laya_only 116.9 ACPL / 17.7% blunders; v2_safe 77.4 / 11.7%; v2_full 68.8 / 10.0%.
- v2 practice match (dev openings, 10 games vs sf_elo_1320): no errors, 73 s/game, v2_full 6.5/10. Not a result.
- v2 frozen before any fresh-set run (tag v2-frozen).
- v2 final runs: the first launch (3 tournaments in parallel) crashed WSL (out of memory while loading Laya three times);
  no games had been played. Relaunched as 2 staggered processes (v2_tour_a, v2_tour_b); ablation games vs laya_only head-to-head
  included in v2_tour_b. Study A-v2 (v2_bench) was unaffected.
- POST HOC (after seeing the tournament): v2_full_oldnet added to Study A-v2 (same 2000 fresh positions, resume of v2_bench)
  to test whether the two value nets differ in per-move quality. Reported as exploratory.
