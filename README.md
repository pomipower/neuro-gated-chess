# Neuro-Gated Chess

**When should a chess agent think?** A fast "System 1" model (LayaChess) proposes a move along with how confident it is. If it isn't confident enough, a slow "System 2" search takes over: PUCT Monte Carlo Tree Search guided by our own value CNN. We test whether that confidence signal is a good guide to *when* search is worth its cost.

> **Building this code (human or AI assistant)?** Read in this order:
> 1. `docs/EXPERIMENT_PROTOCOL.md`: what we measure and why (pre-registered).
> 2. `docs/SPEC.md`: module contracts, file formats, build order. **This is the authoritative spec.**
> 3. `configs/default.yaml`: every hyperparameter. Never hard-code values.
>
> `docs/CHANGES_FROM_V1.md` explains every way this differs from the first README.

## Agents

| Agent | What it does |
|---|---|
| `laya_only` | System 1 only: plays LayaChess's highest-scored move |
| `always_s2` | Always searches (PUCT, 400 simulations, LayaChess scores as root prior) |
| `hybrid_s2_{10,25,50,75}` | **Our method:** searches only when confidence ≤ τ; τ chosen so ≈10/25/50/75% of positions are searched |
| `random_s2_{25,50}` | Control: searches a random 25/50% of positions, to show that *confidence* matters, not just the amount of search |
| `mcts_uniform` | Search with our value net and no LayaChess at all |
| Stockfish | **Judge** (scores every move by centipawn loss) and **rating anchors** (`UCI_Elo` levels). It is a measuring stick, not a competitor to beat |

## Studies

- **A: position benchmark (main results).** 3,000 held-out Lichess Elite positions; every agent's move is judged by Stockfish. Measures quality (ACPL, blunder rate) against cost (wall time, model calls).
- **B: tournament.** 25 fixed openings × both colors against Stockfish anchors → Elo with confidence intervals.
- **C: puzzles.** Lichess tactics by rating band; does the gate fire on tactical positions?

## Status

All modules, scripts, the figure notebook and 36 tests are implemented and smoke-tested end to end in a cloud sandbox, using synthetic games and a stand-in scorer (`laya.model_name: stand-in:material`), because the sandbox could not download the LayaChess weights or Lichess data. **No real results exist yet.** Real runs follow `docs/RUNBOOK.md`.

## Quick start

See **`docs/RUNBOOK.md`** for the full, ordered procedure (setup, data, training, pilot timing, gate calibration, Studies A–C, figures). In short:

```bash
pip install -r requirements.txt          # after installing a CUDA build of PyTorch
git clone https://github.com/devroopsaha744/LayaChess third_party/LayaChess
pip install -r third_party/LayaChess/engine/requirements.txt
export PYTHONPATH=$PWD/third_party/LayaChess/engine:$PYTHONPATH
pytest

python -m scripts.prepare_data           --config configs/default.yaml --val-sf-nodes 50000
python -m src.model.train                --config configs/default.yaml
python -m scripts.pilot_timing           --config configs/default.yaml
python -m scripts.calibrate_gate         --config configs/default.yaml   # commit configs/gate_frozen.yaml
python -m scripts.run_position_benchmark --config configs/default.yaml --split test --run-id bench_test
python -m scripts.run_tournament         --config configs/default.yaml --tier 1 --run-id tour_1
python -m scripts.fit_elo                --config configs/default.yaml --run-id tour_1
python -m scripts.run_puzzles            --config configs/default.yaml --run-id puzzles_1
python -m scripts.make_figures           --config configs/default.yaml --bench-run bench_test --tour-run tour_1 --puzzle-run puzzles_1
```

## Repository layout

```
configs/      default.yaml (all hyperparameters), gate_frozen.yaml (after calibration)
data/         raw/ (not committed), processed/ (not committed), benchmark/ and results/ (committed)
docs/         RUNBOOK, SPEC, EXPERIMENT_PROTOCOL, PAPER_OUTLINE, CHANGES_FROM_V1, DEVIATIONS, ARTIFACTS
src/model/    encoding, ValueCNN, dataset, train
src/engine/   LayaAgent, Stockfish (judge + anchors), gating, HybridAgent, factory
src/mcts/     SearchTree (PUCT)
src/eval/     metrics (CPL, AUROC, ECE, bootstrap), elo, compose (offline hybrids), figures
scripts/      prepare_data, pilot_timing, calibrate_gate, run_position_benchmark, run_tournament, fit_elo,
              run_puzzles, make_figures
notebooks/    plot_metrics.ipynb (figures F1–F6 → paper/figures/)
paper/        main.tex, figures/, references/
tests/        pytest invariants
```

## Hard rules (summary; full list in `docs/SPEC.md`)

- `python-chess` for all board logic, `chess.engine` for Stockfish. Raw PyTorch. No GUI. Static matplotlib/seaborn plots.
- No opening books or tablebases in any agent.
- Value network and labels are **white-perspective**. The only sign conversion is in `SearchTree`.
- Results go to `data/results/<run_id>/` as CSV, with a config and environment snapshot.
- Change the protocol only via `docs/DEVIATIONS.md`.

## License

TBD before publication. Note: LayaChess is Apache-2.0; Lichess database exports are CC0.
