# Build Specification (v2)

This is the contract for building the code. **Read `docs/EXPERIMENT_PROTOCOL.md` first**: it defines what each module must produce. Where this file and `docs/CHANGES_FROM_V1.md` disagree with the v1 README, this file wins.

## Hard rules

1. **`python-chess` is the single source of truth** for boards, legal moves, FEN parsing, game termination. No custom move generation. Use `chess.engine` (built into python-chess) to talk to Stockfish; do not use the `stockfish` pip package.
2. **Raw PyTorch only** for the value net (`torch.nn`, `torch.optim`, `torch.utils.data`). No Lightning/HF Trainer.
3. **No GUI / web frontend.** CLI scripts and one notebook.
4. **Static plots only** (`matplotlib`, `seaborn`). Save PNG (300 dpi) and PDF to `paper/figures/`.
5. **No opening books or tablebases** in any agent. The experiment isolates the gate.
6. **Type hints everywhere**, with `from __future__ import annotations`. Python ≥ 3.10 (3.11 recommended).
7. **All results as CSV with headers** under `data/results/<run_id>/`, plus the config snapshot and `env.json`.
8. **One perspective convention (non-negotiable):** the value network and its training labels are **white-perspective** (+1 white wins, 0 draw, −1 black wins). Conversion to side-to-move happens in exactly one place: `SearchTree`.
9. No wall-clock time limits inside `SearchTree` (budget = `simulations`).

## Repository layout

```
configs/default.yaml           every hyperparameter (see file)
configs/gate_frozen.yaml       written by calibrate_gate.py, then committed
data/raw/                      Lichess Elite PGN months (*.pgn.zst), not committed
data/processed/                train/val arrays, not committed
data/benchmark/                committed: positions_val.csv, positions_test.csv, openings.csv, puzzles.csv
data/results/<run_id>/         committed CSVs + config.yaml + env.json
docs/                          this spec, protocol, paper outline, changes
src/utils.py                   config loading, seeding, run dirs, env.json snapshots
src/model/encoding.py          board_to_planes, flip/pack helpers (pure numpy)
src/model/architecture.py      ValueCNN
src/model/dataset.py           PGN -> positions -> torch Dataset
src/model/train.py             training CLI
src/engine/gating.py           confidence functions, quantile τ, Gate classes
src/engine/LayaAgent.py        System 1 wrapper
src/engine/Stockfish.py        StockfishJudge + StockfishPlayer
src/engine/HybridAgent.py      the router
src/engine/factory.py          builds every named agent from config
src/mcts/SearchTree.py         PUCT search
src/eval/metrics.py            CPL, AUROC, ECE, bootstrap
src/eval/elo.py                anchored Elo + cluster bootstrap
src/eval/compose.py            offline hybrid / random-gate composition + paired tests (protocol §2.3)
src/eval/figures.py            F1–F6 + LaTeX tables (used by the notebook and make_figures)
scripts/prepare_data.py        PGN months -> splits, benchmark positions, openings, puzzles
scripts/pilot_timing.py        per-move costs on this machine + run-time estimates (run first)
scripts/calibrate_gate.py      §2.4 of the protocol
scripts/run_position_benchmark.py   Study A
scripts/run_tournament.py      Study B
scripts/run_puzzles.py         Study C
scripts/fit_elo.py             games.csv -> ratings.csv
scripts/make_figures.py        all figures/tables into paper/figures/
notebooks/plot_metrics.ipynb   figures F1–F6
tests/                         invariants (run with `pytest`)
third_party/LayaChess/         git clone of the LayaChess repo, not committed
```

---

## `src/model/encoding.py`

```python
def board_to_planes(board: chess.Board) -> np.ndarray:   # float32, shape (18, 8, 8)
```

| Plane | Content |
|---|---|
| 0–11 | pieces in order `P N B R Q K p n b r q k` (white first). Square (rank r, file f) → `[plane, r, f]` with r = 0 for rank 1, so a1 → `[:, 0, 0]` |
| 12 | side to move: all ones if white to move, else all zeros |
| 13–16 | castling rights: white kingside, white queenside, black kingside, black queenside (constant planes) |
| 17 | en-passant target square: a single 1 on the target square, else zeros |

Pure function of the `chess.Board` state; move history, halfmove clock and repetition do not enter. `flip_color(board)` returns `board.mirror()` and callers negate the label (optional color-flip augmentation).

## `src/model/architecture.py`

```python
class ValueCNN(nn.Module):
    """(batch, 18, 8, 8) -> (batch, 1) in [-1, 1], white perspective."""
```

"ValueNet-6×128":

- Stem: `Conv3×3(18→128, bias=False) → BN → ReLU`
- 6 residual blocks, 128 channels: `conv3×3 → BN → ReLU → conv3×3 → BN`, add skip, ReLU
- Head: `Conv1×1(128→32) → BN → ReLU → Flatten (2048) → Linear(2048→256) → ReLU → Linear(256→1) → Tanh`
- He init (`kaiming_normal_`) for convs. 14 conv layers + 2 FC. **≈ 2.32M parameters.** Do not pool the head down to a scalar (the v1 design did and discarded all spatial information).
- `make_value_fn(model, device) -> Callable[[chess.Board], float]`: encodes with `board_to_planes`, runs `model` in `eval()`/`no_grad`, returns a white-perspective float. Counts evaluations in `value_fn.calls` (for `cnn_evals` logging).

## `src/model/dataset.py` and `train.py`

- Read `.pgn.zst` via `zstandard` stream + `io.TextIOWrapper` + `chess.pgn.read_game`. Keep games with `Termination == "Normal"`, a decisive-or-draw `Result`, and ≥ 20 plies.
- Per game sample ≤ 8 positions, ply ≥ 8, de-duplicate by the first four FEN fields. Label `z` = game result, **white perspective**.
- `label.mode` (config): `outcome` → y = z; `blend` → y = λ·z + (1−λ)·tanh(cp/400), cp from Stockfish at `label.sf_nodes`, white perspective. Switch to `blend` if the §5 Spearman check fails.
- Split **by month**, never by position (see protocol §2).
- Loss `MSELoss`; `AdamW(lr=1e-3, weight_decay=1e-4)`; `CosineAnnealingLR`; batch 256; epochs 20 (CLI). Save `src/model/checkpoints/best_model.pt` on best validation loss. CLI: `--config`, `--epochs`, `--batch-size`, `--lr`, `--device`.
- Per epoch print `train_loss, val_loss, val_sign_acc_excl_draws`; also print the constant-predictor baseline MSE once.

## `src/engine/gating.py`

```python
def margin(win_chances: Sequence[float]) -> float                           # top1 - top2 (1.0 if only one move)
def top1_softmax(win_chances: Sequence[float], T: float) -> float           # softmax over logit(w)/T, top prob
def norm_entropy_conf(win_chances: Sequence[float], T: float) -> float      # 1 - H/log(n); 1.0 if n == 1
def quantile_threshold(confidences: Sequence[float], s2_rate: float) -> float   # τ with ≈ s2_rate of values <= τ
class Gate(Protocol): def use_system2(self, confidence: float) -> bool
class ConfidenceGate(Gate)   # S2 iff confidence <= tau
class RandomGate(Gate)       # S2 with prob = rate, seeded
class AlwaysGate(Gate)       # always S2
class NeverGate(Gate)        # never S2
```

A position with a single legal move has confidence 1.0 (it is forced). `n == 0` is never passed.

## `src/engine/LayaAgent.py`

LayaChess is a **separate fine-tuned checkpoint** (`datafreak/laya-chess`, Apache-2.0, ≈ 0.4B parameters) of the Laya base model, not the base model itself. It **scores each legal move**, returning the side to move's win chance (10-level scale) per move. Its reference implementation is the GitHub repo `devroopsaha744/LayaChess`; clone it to `third_party/LayaChess` and put `third_party/LayaChess/engine` on `PYTHONPATH`. The documented usage is:

```python
from laya_chess import LayaChessModel
model = LayaChessModel("datafreak/laya-chess")
model.score_moves(chess.Board())      # -> list of (move, win) pairs
```

**Confirmed from the LayaChess source** (`engine/laya_chess/model.py`): `LayaChessModel(checkpoint, revision=None, device="auto", dtype=None, batch_size=96, cache_size=200_000)`; `evaluate(boards) -> [(legal_moves, q)]` batches every legal move of every board (one Laya forward pass per move-question, `batch_size` per pass, fp16 on CUDA); `q` is the expectation over 10 equal-width win-chance levels, so it lies in **[0.05, 0.95]**; `score_moves` sorts best-first. It keeps a position cache: our wrapper clears it before every timed call (`laya.use_cache: false`). It needs the `laya` package (`github.com/NandhaKishorM/laya`) listed in its own `requirements.txt`.

**Cost:** one position costs one forward pass of a 421M-parameter model *per legal move* (~35). LayaChess's README reports ~14 move scores/s on an Apple M4. System 1 is therefore **not** cheap relative to our System 2; every cost claim must be measured (`scripts/pilot_timing.py`).

`model_name: "stand-in:material"` loads `MaterialStandIn`, a fake scorer with the same interface for smoke tests and CI. Never report results from it.

```python
@dataclass(frozen=True)
class LayaOutput:
    scores: dict[chess.Move, float]    # win chance in [0,1], one entry per legal move
    best_move: chess.Move
    confidence: float
    n_scored: int
    elapsed_ms: float

class LayaAgent:
    def __init__(self, model_name: str = "datafreak/laya-chess", revision: str | None = None,
                 device: str = "cuda", confidence: str = "margin", temperature: float = 0.1,
                 scorer=None, batch_size: int = 96, use_cache: bool = False) -> None: ...
    def evaluate(self, board: chess.Board) -> LayaOutput: ...            # timed, cache cleared
    def score_many(self, boards) -> list[dict[chess.Move, float]]: ...  # untimed batch (calibration)
```

`evaluate` must not call Stockfish or any search; it falls back to CPU if CUDA is unavailable; it returns only legal moves.

## `src/engine/Stockfish.py`

Two classes, both on `chess.engine.SimpleEngine.popen_uci`, restart on `EngineTerminatedError`:

- `StockfishJudge(path, nodes=200_000, threads=1, hash_mb=128)`: `analyse(board) -> JudgeResult(best_move, score_cp, second_move, second_score_cp)` (+ `gap_cp`) with MultiPV=2, scores from the side to move's view, mate → ±1000, clipped to ±1000. `cpl(board, move)` scores a non-best move by searching it as the only root move (`root_moves=[move]`) at the same node budget. Every analysis starts a new `game` (ucinewgame clears the hash), so results are deterministic and order-independent. Results are cached by FEN. `eval_white_cp(board)` gives white-perspective scores for labels. Engines are closed automatically at interpreter exit (an open python-chess engine otherwise hangs the exit).
- `StockfishPlayer(path, elo, movetime_ms=100, threads=1, hash_mb=64)`: sets `UCI_LimitStrength=true`, `UCI_Elo=elo` (valid range 1320–3190), `play(board) -> chess.Move`.

## `src/mcts/SearchTree.py`

```python
@dataclass
class SearchResult:
    move: chess.Move
    visits: dict[chess.Move, int]
    root_q: float                 # mean value at the root, side-to-move perspective
    sims: int
    cnn_evals: int
    max_depth: int
    mean_leaf_depth: float

class SearchTree:
    def __init__(self, value_fn: Callable[[chess.Board], float], simulations: int = 400,
                 c_puct: float = 1.5, fpu_reduction: float = 0.2, prior_temperature: float = 0.3) -> None: ...
    def search(self, board: chess.Board, root_scores: dict[chess.Move, float] | None = None) -> SearchResult: ...
```

- `value_fn` returns a **white-perspective** value in [−1, 1]. A node's value from the player to move there is `v if white to move else −v`. This is the only place sign conversion happens.
- **PUCT:** `score(child) = Q(child) + c_puct · P(child) · sqrt(N_parent) / (1 + N_child)`. Q is the child's mean value *from the parent's player's view*, i.e. the negation of the child's own mean. Unvisited children use `Q = Q_parent − fpu_reduction`.
- **Priors:** at the root, `P = softmax(logit(w)/prior_temperature)` over `root_scores` (Laya win chances); if `root_scores is None` → uniform. **Interior nodes use uniform priors** (Laya is too expensive to call there). `mcts_uniform` passes `None`.
- **Terminal nodes** are evaluated exactly, with no network call: checkmate → −1 for the side to move at that node; stalemate / insufficient material / 75-move / fivefold → 0 (`board.outcome(claim_draw=False)`; tree boards have no move stack, so threefold and 50-move *claims* are not seen inside the tree).
- **Backpropagation flips sign at every ply.**
- Final move: most visits; ties → higher Q → higher prior. No Dirichlet noise, no temperature, no tree reuse (tree reuse is off in all agents so each decision is independent; it may be added later as a separate ablation).
- Expand with `board.copy(stack=False)` for speed (repetition is not detected inside the tree; document this limitation).
- Log `cnn_evals` (non-terminal leaf evaluations), `max_depth`, `mean_leaf_depth` (mean depth of the leaf reached over all simulations).
- If a pilot shows > 1 s per search at 400 simulations, add leaf batching with virtual loss as an optimization (not before).

## `src/engine/HybridAgent.py` and `factory.py`

```python
@dataclass
class MoveInfo:
    system: int                      # 1 or 2
    confidence: float                # NaN for mcts_uniform
    time_elapsed_ms: float
    laya_calls: int
    laya_moves_scored: int
    cnn_evals: int
    mcts_sims: int
    max_depth: int                   # 0 if system == 1
    mean_leaf_depth: float           # 0.0 if system == 1
    laya_ms: float                   # part of time_elapsed_ms spent in Laya
    search_ms: float                 # part spent in the search

class HybridAgent:
    def __init__(self, laya: LayaAgent | None, search: SearchTree | None, gate: Gate,
                 use_laya_prior: bool = True) -> None: ...
    def select_move(self, board: chess.Board) -> tuple[chess.Move, MoveInfo]: ...
    def decide(self, board, laya_out=None): ...   # also returns the LayaOutput / SearchResult; reuses laya_out
```

Logic: if `laya` is not None, call `laya.evaluate`. If `search is None` or `not gate.use_system2(conf)` → play Laya's `best_move` (system 1). Otherwise run `search.search(board, root_scores=out.scores if use_laya_prior else None)` and play its move (system 2). `mcts_uniform` = `HybridAgent(laya=None, search=tree, gate=AlwaysGate(), use_laya_prior=False)`. `tau` and all knobs come from config, never hard-coded. `factory.build_agent(name, Resources(cfg, gate_cfg), seed)` returns any agent named in the protocol table (Resources loads Laya and the value net once). Stockfish anchors are wrapped as `StockfishAnchorAgent` (system 0) so every agent has `select_move` and `new_game`.

## `scripts/prepare_data.py`

From `data/raw/*.{pgn,pgn.zst,zip}` (YYYY-MM in each file name), with month roles from config (or, if unset, newest month = test, the one before = val, the rest = train): write `data/processed/{train,val}.npz` (packed planes, `y`, `fens`, `game_ids`, `plies`, optional `sf_cp_white`), `data/benchmark/positions_{val,test}.csv` (`pos_id, fen, ply, phase, n_legal, game_id`; FENs seen in training are excluded), `data/benchmark/openings.csv` (25 openings, 8 plies, eval within ±100 cp by the judge), and `data/benchmark/puzzles.csv` (stratified sample, 200 per rating bucket, from `lichess_db_puzzle.csv.zst`). Also writes `split_summary.json`. Deterministic given the config seed. `--val-sf-nodes N` adds Stockfish evals to `val.npz` for the Spearman check; `--sf-labels` adds them to train for `label.mode: blend`.

## `scripts/calibrate_gate.py`, `run_position_benchmark.py`

- `calibrate_gate.py`: protocol §2.4 on the `val` positions; writes `configs/gate_frozen.yaml` and a small CSV of candidate AUROCs.
- `run_position_benchmark.py --split test`: runs `laya_only`, `always_s2`, `mcts_uniform` per position; judge (cached); writes `benchmark_moves.csv`:
  `run_id, split, pos_id, fen, ply, phase, n_legal, sf_best_uci, sf_score_cp, sf_gap_cp, agent, move_uci, system, confidence, cpl, is_best, wall_ms, laya_ms, search_ms, laya_calls, laya_moves_scored, cnn_evals, mcts_sims, max_depth, mean_leaf_depth`.
  `always_s2` reuses the same position's `laya_only` Laya call as its prior (its `wall_ms` = `laya_ms + search_ms`). Hybrids and random-gate controls are composed from this file by `src/eval/compose.py` (protocol §2.3). `--resume`, `--shard i/n` (then `--merge`).

## `scripts/run_tournament.py`, `fit_elo.py`

- `run_tournament.py --tier 1|2|3`: plays pairings from the protocol with `openings.csv`; appends to `games.csv` (`run_id, game_id, opening_id, white, black, result, plies, termination, seconds, white_error, black_error`) and `moves.csv` (`game_id, ply, agent, move_uci, fen_before` + all `MoveInfo` fields). Resumable: skip `(opening, white, black)` triples already present.
- `run_tournament.py --pilot` plays 2 openings per pairing and prints the projected hours; `--judge-moves` writes `moves_judged.csv` (CPL and blunder flag for every move). `random_s2_*` agents are re-seeded per game from the game id.
- `fit_elo.py --run-id a,b`: pooled `games.csv` + anchors → `ratings.csv` (agent, elo, ci_low, ci_high, games, score, anchor, off_scale) via `src/eval/elo.py`.

## `scripts/run_puzzles.py`

Study C as in the protocol → `puzzles_results.csv` (`puzzle_id, rating, bucket, agent, move_uci, solved, system, confidence`).

## `src/eval/metrics.py`

`cpl` helpers (score clipping, mate mapping), `auroc`, `ece(conf, correct, bins=10)`, `bootstrap_ci(stat_fn, data, n=2000, seed)` with paired resampling, `holm(pvals)`. Use `scikit-learn` for ROC.

## `notebooks/plot_metrics.ipynb`

Thin wrapper over `src/eval/figures.make_all` (also `python -m scripts.make_figures`). Produces F1–F6, `F3_tests.csv` (H2 paired tests at matched S2 rate, Holm over 25%/50%; H3 recovery R), `table_elo.tex`, `table_puzzles.tex`; each figure as PNG (300 dpi), PDF and CSV. Colors: blue = confidence gate, orange = random control, gray = baselines (validated palette; values also direct-labeled / in CSV).

## Tests (`tests/`, run with `pytest`)

36 tests: encoding (side to move, castling, en passant, purity, flip, packing), gating, ValueCNN shape/size/spatial head, MCTS (mate-in-1 for both colors with a value_fn that always returns 0, white-perspective sign, not hanging the queen, root prior), Stockfish judge/player (mate mapping, perspective, determinism; skipped if no binary), agent routing and logging (with the stand-in scorer), offline composition, Elo recovery of a known difference, and figure generation.

## Build order

All modules are implemented and smoke-tested end to end (synthetic games + stand-in scorer). Run order on the real machine: see `docs/RUNBOOK.md`.
