# Runbook: real runs on one machine (RTX 4060, 8 GB)

Every command runs from the repo root. Each step says what to commit. If anything fails, stop and fix it: do not skip steps, because later steps depend on files written by earlier ones.

## 0. Setup (once)

```bash
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
# Install a CUDA build of PyTorch first: pick your OS + CUDA on https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # must print True + RTX 4060

git clone https://github.com/devroopsaha744/LayaChess third_party/LayaChess
pip install -r third_party/LayaChess/engine/requirements.txt        # pulls transformers and the `laya` package
export PYTHONPATH=$PWD/third_party/LayaChess/engine:$PYTHONPATH      # PowerShell: $env:PYTHONPATH="$PWD\third_party\LayaChess\engine"
python -c "from laya_chess import LayaChessModel; import chess; m = LayaChessModel(); print(m.score_moves(chess.Board())[:3])"
```

- **Stockfish:** install it (Linux: `sudo apt install stockfish`; Windows: download the binary from stockfishchess.org) and set `judge.path` in `configs/default.yaml` to the executable if it is not on PATH.
- **Data** into `data/raw/`: at least **4 monthly files** from the Lichess Elite Database (https://database.nikonoel.fr/), with the `YYYY-MM` part of the name kept as is, plus `lichess_db_puzzle.csv.zst` from https://database.lichess.org/ (puzzles section).
- `pytest`: all tests must pass (Stockfish tests run only if the binary is found).

## 1. Prepare data (≈ minutes to an hour)

Set `data.train_months`, `val_month`, `test_month` in `configs/default.yaml` (newest = test, the one before = val), or leave them empty to get that automatically.

```bash
python -m scripts.prepare_data --config configs/default.yaml --val-sf-nodes 50000
```
**Commit** `data/benchmark/` (positions, openings, puzzles, split_summary.json). Copy the months into `docs/DEVIATIONS.md`.

## 2. Train the value network (start early, it runs in the background)

```bash
python -m src.model.train --config configs/default.yaml --device cuda
```
Check the last lines: val MSE must beat the printed baseline; if `Spearman < 0.3`, set `label.mode: blend`, rerun step 1 with `--sf-labels` (slow) and retrain. Upload `best_model.pt` somewhere shared and put its SHA-256 in `docs/ARTIFACTS.md` (`sha256sum src/model/checkpoints/best_model.pt`).
Don't run Laya steps on the GPU while training if VRAM runs out; Laya in fp16 needs about 1 GB plus activations.

## 3. Pilot timing (5 minutes; decides run sizes)

```bash
python -m scripts.pilot_timing --config configs/default.yaml --n 20
```
Paste the whole output into `docs/DEVIATIONS.md`. If Study A would take more than ~10 h, lower `benchmark.n_test` (not below 1,500) or `mcts.simulations` (not below 200), and record it there.

## 4. Calibrate the gate on val, then freeze it

```bash
python -m scripts.calibrate_gate --config configs/default.yaml
git add configs/gate_frozen.yaml && git commit -m "Freeze gate (val calibration)"
```
**Commit before step 5.** This commit is what makes the test results honest.

## 5. Study A on test (main results; resumable)

```bash
python -m scripts.run_position_benchmark --config configs/default.yaml --split test --run-id bench_test
# interrupted? same command + --resume
```
**Commit** `data/results/bench_test/`.

## 6. Study B tournament

```bash
# pilot: choose anchors so our agents score 25–75% against at least two of them
python -m scripts.run_tournament --config configs/default.yaml --tier 1 --run-id tour_pilot --pilot
# set tournament.anchors.elos, record the pilot estimate in DEVIATIONS.md, then:
python -m scripts.run_tournament --config configs/default.yaml --tier 1 --run-id tour_1
python -m scripts.run_tournament --config configs/default.yaml --run-id tour_1 --judge-moves
python -m scripts.fit_elo --config configs/default.yaml --run-id tour_1
```
Tiers 2 and 3 only if time allows (`--tier 2 --run-id tour_2`, pool with `--run-id tour_1,tour_2`).

## 7. Study C puzzles

```bash
python -m scripts.run_puzzles --config configs/default.yaml --run-id puzzles_1
```

## 8. Figures and tables for the paper

```bash
python -m scripts.make_figures --config configs/default.yaml --bench-run bench_test --tour-run tour_1 --puzzle-run puzzles_1
```
Everything lands in `paper/figures/` (PNG, PDF, CSV, and `.tex` tables). `F3_tests.csv` has the H2/H3 numbers. **Commit** results and figures.
