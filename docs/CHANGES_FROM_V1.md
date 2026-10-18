# Changes from the v1 README (and why)

Each row is a problem in v1 and what v2 does instead. Severity: **C** = would silently produce wrong results, **R** = would weaken or invalidate the paper's claims, **M** = minor.

| # | Sev | v1 | Problem | v2 |
|---|---|---|---|---|
| 1 | C | Labels "from the perspective of the side to move", but the CNN "always evaluates from white's perspective" and MCTS negates for black | Contradictory sign conventions; the 12-plane input has no side-to-move information, so identical piece layouts got opposite labels | Labels are **white-perspective**; one sign conversion, only in `SearchTree`; tests guard both |
| 2 | C | 12 planes; design log says castling/en passant are encoded | They are not in the input spec; no side-to-move plane | **18 planes**: 12 pieces + side to move + 4 castling + en passant (`src/model/encoding.py`, implemented and tested) |
| 3 | C | Head: 1×1 conv to **1 channel** → global average pool → FC(256) | The whole position reaches the FC layers as one averaged number | 1×1 conv to 32 channels → flatten (2048) → FC(256) → FC(1) → tanh; ≈ 2.32M params; test checks the output depends on piece location |
| 4 | C | Laya = one forward pass giving a move distribution; confidence = argmax probability | LayaChess (`datafreak/laya-chess`) scores **each legal move** with a win chance; no move distribution exists | `LayaOutput` holds per-move win chances; confidence functions (margin / softmax / entropy) are defined, and one is **selected on validation data** before testing |
| 5 | C | MCTS: "UCB1 = Q + c_puct·sqrt(ln N / n)", negate only at the leaf | No prior (Laya's scores unused), wastes the budget on ~35 root children; sign must flip at every ply; terminal states not specified | **PUCT** with Laya root prior, FPU, exact terminal values, sign flip per ply; mate-in-1 tests for both colors |
| 6 | R | Hypothesis: "comparable strength to a pure-search engine" (Stockfish) | Not achievable with a ~2M-parameter net and 800 simulations; the paper would report a failed headline | Stockfish becomes the **judge** and **rating anchor**; hypotheses test whether the **gate** is useful (H1–H6), each outcome is reportable |
| 7 | R | Agents: Laya, Stockfish, Hybrid at 6 τ values | No control showing the gate beats trivial alternatives; no "always search" upper bound | Adds `always_s2`, `mcts_uniform`, and **random-gate controls at matched S2 rates**, the key comparison for a gating claim |
| 8 | R | τ ∈ {0.5,…,0.9} fixed by hand | Raw τ values are meaningless across confidence functions and uncalibrated models | τ set by **quantile of validation confidences** for target S2 rates (10/25/50/75%), frozen before test |
| 9 | R | "Compute" = nodes searched | Ignores the 421M-param Laya inference per legal move; biased toward the hybrid | Logs wall time (cuda-synchronized) **and** Laya calls, moves scored, CNN evals, sims |
| 10 | R | Elo from 100-game round-robin vs full-strength Stockfish | Full Stockfish scores ~100%, so the Elo scale is undefined; 2,800+ games, deterministic agents replay identical games | **Study A** position benchmark (main results, thousands of judged positions, cheap); **Study B** small tournament with 25 fixed openings × 2 colors vs `UCI_Elo` anchors, Bradley–Terry Elo with cluster-bootstrap CIs |
| 11 | R | Blunder = "eval drops > 200 cp vs Stockfish" | Undefined perspective, mate handling, judge depth | CPL against a fixed-node Stockfish judge with clipping and mate mapping (protocol §2.1) |
| 12 | R | Random 90/10 split of positions | Positions from one game in both train and val; leaks into results | **Month-based** train / val / test split |
| 13 | R | Val accuracy = sign match | Draws have no sign | Sign accuracy excluding draws + constant-predictor baseline + Spearman vs Stockfish |
| 14 | R | No statistics | No CIs, no tests | Paired bootstrap CIs, Holm correction, pre-registered hypotheses, deviations log |
| 15 | R | No reproducibility plan | | `configs/default.yaml`, config + env snapshot per run, committed benchmark positions and result CSVs, pinned Laya revision |
| 16 | M | "No time cutoffs in MCTS" but "5 s max per move"; tree reuse with a stateless `search(board)` | Contradictions | Budget = simulations only; no wall-clock cap for our agents; tree reuse off (possible later ablation) |
| 17 | M | `stockfish` pip package | Fragile; python-chess already has `chess.engine` | `chess.engine` |
| 18 | M | `.gitignore` ignores `*.csv` | Result CSVs could not be shared through git | Only raw/processed data ignored; benchmark + results committed |
| 19 | M | `requirements.txt` has torchvision; no scipy / sklearn / pytest | Unused / missing | Fixed; PyPI name `chess`; LayaChess install documented |
| 20 | M | `plot_metrics.ipynb` contained a single space | Not valid JSON; Jupyter cannot open it | Valid notebook skeleton with a section per figure |
| 21 | M | Plot plan: "Elo vs τ", stacked area, violin | Elo per τ needs a tournament per τ | Figures F1–F6 mostly from Study A; Elo table from Study B |

## Added while building (v2.1)

| Change | Why |
|---|---|
| LayaChess API confirmed from its source: `evaluate(boards)` batches all legal moves; win chances lie in [0.05, 0.95]; position cache cleared before timed calls | Spec previously had to guess |
| Protocol §8: System 1 cost is measured, not assumed; LayaChess's own Laya-MCTS cited as closest baseline; run sizes from `pilot_timing.py` | LayaChess runs one 421M-param pass per legal move |
| Stockfish judge starts a new game per analysis and scores non-best moves via `root_moves` | Deterministic, order-independent CPL |
| Stand-in scorer (`stand-in:material`) | Smoke tests and CI without GPU or model download |
| `src/eval/compose.py`, `src/eval/figures.py`, `scripts/make_figures.py`, `scripts/pilot_timing.py`, `docs/RUNBOOK.md` | Analysis code shared by the notebook and CLI; run sizing; ordered run procedure |
