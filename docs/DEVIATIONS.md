# Deviations from the pre-registered protocol

Log every change to `docs/EXPERIMENT_PROTOCOL.md` or `configs/` made after the protocol was committed. Each entry is reported in the paper.

| Date | Who | What changed | Why | Seen test results before deciding? |
|---|---|---|---|---|
| | | | | |

Also record here: pilot timings (seconds per position per agent, seconds per game), the chosen anchor Elos, the final train/val/test months, and the Laya revision hash.
| 2026-10-18 | Rohan | Value net: color_flip_augmentation true, weight_decay 1e-3, 10 epochs | Run 1 (defaults) overfit after epoch 5: best val MSE 0.703, Spearman 0.574. Run 2: best val MSE 0.686 (baseline 0.811), Spearman 0.614. Kept run 2. | No (validation data only) |

Pilot timing (2026-10-18, RTX 4060 Laptop, WSL2, torch 2.14.1+cu130, Stockfish 16, n=20 val positions):
laya_only 392 ms/move (81 move scores/s); always_s2 1343 ms (laya 309 + search 1034); mcts_uniform 1092 ms. 400 simulations kept; full study sizes kept.
