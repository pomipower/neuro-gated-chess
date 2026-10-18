# Neuro-Gated Chess Engine

> **This README is an agent-optimized system instruction manual.** It is designed to be consumed by AI coding assistants (Claude, Cursor, Copilot, Gemini, etc.) as the authoritative reference for understanding, navigating, and contributing to this codebase. Human readers will also find it useful, but the structure, specificity, and directive tone are intentionally tailored for LLM consumption.

---

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- SECTION 1: PROJECT SUMMARY & GOAL                                      -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

## [PROJECT SUMMARY & GOAL]

### Simplified Description

We are building a chess bot that **plays fast when a move is obvious** and **calculates deeply when a position is complex**, mimicking how strong human players allocate thinking time. Simple recaptures and forced moves are executed instantly; murky middlegame positions trigger deep search.

### Technical Description

We are implementing a **Confidence-Gated System 1 / System 2 neuro-symbolic architecture** for chess move selection.

| Concept | Implementation |
|---|---|
| **System 1 (Fast/Intuitive)** | **Laya 421M** — an open-weights chess decision model. It takes a board state and returns a move probability distribution + a scalar confidence score. |
| **System 2 (Slow/Deliberate)** | **Bounded MCTS** — a Monte Carlo Tree Search whose leaf-node evaluations come from a custom **10-layer PyTorch CNN** (the "value network"), trained on Lichess Elite game data. |
| **Gating Mechanism** | A confidence threshold **τ** (tau). If Laya's confidence **> τ**, the System 1 move is played instantly. If confidence **≤ τ**, the system delegates to the System 2 MCTS search. |
| **Baseline Comparator** | **Stockfish** (via subprocess) serves as the ground-truth oracle for evaluation, Elo calibration, and blunder detection in tournament analysis. |

### Core Hypothesis

> A confidence-gated hybrid agent can achieve **comparable playing strength to a pure-search engine** while consuming **significantly less compute** (measured in nodes searched per move), because the majority of chess positions admit an obvious best move that does not require deep search.

### Research Paper Target

The results will be compiled into an **IEEE-format LaTeX paper** (`paper/main.tex`) reporting:
- Elo rating of each agent variant (Laya-only, MCTS-only, Hybrid at various τ values, Stockfish baseline)
- Compute efficiency: nodes/second, time/move distributions
- Blunder rate reduction across game phases (opening, middlegame, endgame)
- Search depth utilization histograms

---

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- SECTION 2: AGENT DIRECTIVES & CODEBASE MAP                            -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

## [AGENT DIRECTIVES & CODEBASE MAP]

### Repository Tree

```
neuro-gated-chess/
├── data/
│   ├── raw/                     # Downloaded Lichess PGN files (*.pgn, *.zst)
│   └── results/                 # Tournament CSV outputs from run_tournament.py
├── paper/
│   ├── main.tex                 # IEEE-format LaTeX paper draft
│   └── figures/                 # Exported PNG/PDF graphs from plot_metrics.ipynb
├── src/
│   ├── engine/
│   │   ├── LayaAgent.py         # Laya local inference logic
│   │   ├── Stockfish.py         # Stockfish subprocess wrapper
│   │   └── HybridAgent.py       # The confidence-gating router
│   ├── mcts/
│   │   └── SearchTree.py        # Bounded MCTS logic
│   └── model/
│       ├── train.py             # PyTorch Value CNN training script
│       └── architecture.py      # 10-layer CNN definition
├── scripts/
│   └── run_tournament.py        # Automates bots playing each other
├── notebooks/
│   └── plot_metrics.ipynb       # Jupyter notebook for paper graphs
├── .gitignore
├── requirements.txt
└── README.md                    # THIS FILE — the agent instruction manual
```

### File-by-File Directives

> **CRITICAL:** Every module below uses `python-chess` (`import chess`) as the single source of truth for board representation, move generation, and legality validation. Do NOT implement custom move generators, FEN parsers, or board representations outside of what `python-chess` provides.

---

#### `src/engine/LayaAgent.py` — System 1 Fast Inference

**Purpose:** Wraps the Laya 421M open-weights model for local inference.

**Required Interface:**

```python
class LayaAgent:
    def __init__(self, model_path: str, device: str = "cuda"):
        """Load Laya model weights from disk."""
        ...

    def evaluate(self, board: chess.Board) -> tuple[chess.Move, float]:
        """
        Given a python-chess Board, return:
          - best_move: chess.Move — the top-ranked legal move
          - confidence: float in [0.0, 1.0] — scalar confidence score

        Implementation notes:
          1. Convert board to the input tensor format expected by Laya.
          2. Run forward pass. Do NOT use batching for single-board inference.
          3. Mask illegal moves from the output distribution.
          4. Return argmax move and its probability as the confidence scalar.
        """
        ...
```

**Constraints:**
- Must handle both white and black perspectives (flip board if needed).
- Must gracefully fall back if CUDA is unavailable (default to CPU).
- Must NOT call Stockfish or any external engine — this is purely neural.

---

#### `src/engine/Stockfish.py` — Baseline Engine Wrapper

**Purpose:** Subprocess wrapper around the Stockfish binary for baseline Elo comparisons and position analysis.

**Required Interface:**

```python
class StockfishAgent:
    def __init__(self, path: str = "stockfish", depth: int = 20, time_limit: float | None = None):
        """
        Initialize Stockfish subprocess via UCI protocol.
        Either depth or time_limit should constrain the search, not both.
        """
        ...

    def evaluate(self, board: chess.Board) -> tuple[chess.Move, float]:
        """
        Return:
          - best_move: chess.Move
          - eval_score: float — centipawn evaluation normalized to [-1, 1]
            (use tanh scaling: tanh(cp / 400))
        """
        ...

    def close(self):
        """Terminate the Stockfish subprocess cleanly."""
        ...
```

**Constraints:**
- Must use the `stockfish` Python package or raw UCI protocol via `subprocess.Popen`.
- Must handle Stockfish crashes gracefully (restart subprocess on failure).
- Expose configurable `Threads`, `Hash`, and `Contempt` UCI options.

---

#### `src/engine/HybridAgent.py` — The Confidence-Gating Router

**Purpose:** This is the **core innovation** of the project. It implements the System 1 / System 2 routing decision.

**Required Interface:**

```python
class HybridAgent:
    def __init__(self, laya: LayaAgent, search_tree: SearchTree, tau: float = 0.75):
        """
        Args:
            laya: The System 1 fast agent.
            search_tree: The System 2 MCTS searcher.
            tau: Confidence threshold. Laya plays if confidence > tau.
        """
        ...

    def select_move(self, board: chess.Board) -> tuple[chess.Move, dict]:
        """
        Gating logic:
          1. Call laya.evaluate(board) → (move, confidence)
          2. If confidence > self.tau:
               return (move, {"system": 1, "confidence": confidence})
          3. Else:
               mcts_move = self.search_tree.search(board)
               return (mcts_move, {"system": 2, "confidence": confidence, "nodes_searched": ...})

        The metadata dict is essential for research logging.
        """
        ...
```

**Constraints:**
- The `tau` threshold must be configurable at runtime (not hardcoded).
- The metadata dictionary returned alongside each move MUST include at minimum: `system` (1 or 2), `confidence`, `time_elapsed_ms`. When System 2 is triggered, also include `nodes_searched` and `search_depth`.
- Do NOT add additional routing logic (e.g., opening book lookups, endgame tablebases) unless explicitly requested. Keep the gating mechanism clean and measurable.

---

#### `src/model/architecture.py` — Value Network CNN

**Purpose:** Defines the 10-layer convolutional neural network used as the value/evaluation function.

**Required Interface:**

```python
class ValueCNN(nn.Module):
    """
    Input:  Tensor of shape (batch, 12, 8, 8) — 12-plane bitboard encoding
            Plane order: [P, N, B, R, Q, K, p, n, b, r, q, k]
            where uppercase = white pieces, lowercase = black pieces.
            Each plane is an 8x8 binary matrix (1 = piece present, 0 = empty).

    Output: Tensor of shape (batch, 1) — continuous value in [-1, 1]
            -1 = black winning, 0 = draw, +1 = white winning.
    """
    def __init__(self):
        super().__init__()
        # 10 convolutional layers with residual connections
        # Final layers: GlobalAvgPool → FC(256) → FC(1) → Tanh
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

**Architecture Specification:**
| Layer | Type | Filters | Kernel | Activation | Notes |
|-------|------|---------|--------|------------|-------|
| 1 | Conv2d | 64 | 3×3 | ReLU | Input projection, padding=1 |
| 2–3 | ResBlock | 64 | 3×3 | ReLU | Residual skip connection |
| 4 | Conv2d | 128 | 3×3 | ReLU | Channel expansion, padding=1 |
| 5–6 | ResBlock | 128 | 3×3 | ReLU | Residual skip connection |
| 7 | Conv2d | 256 | 3×3 | ReLU | Channel expansion, padding=1 |
| 8–9 | ResBlock | 256 | 3×3 | ReLU | Residual skip connection |
| 10 | Conv2d | 1 | 1×1 | — | Reduce to 1 channel |
| — | GlobalAvgPool | — | — | — | Spatial dims → scalar |
| — | FC | 256 | — | ReLU | Hidden dense layer |
| — | FC | 1 | — | Tanh | Output value head |

**Constraints:**
- Use `nn.BatchNorm2d` after every Conv2d (before activation).
- Use He initialization (`kaiming_normal_`) for all convolutional weights.
- Total parameter count should be approximately 1–3M parameters. Do NOT over-engineer.
- Input tensor must follow the 12-plane bitboard encoding described above. Provide a utility function `board_to_tensor(board: chess.Board) -> torch.Tensor` in this file.

---

#### `src/model/train.py` — Training Pipeline

**Purpose:** Trains the `ValueCNN` on labeled chess positions extracted from Lichess Elite PGN data.

**Required Functionality:**
1. **Data Loading:** Parse PGN files from `data/raw/`, extract positions at random ply depths, label each position with the game outcome from the perspective of the side to move: `+1` (win), `0` (draw), `-1` (loss).
2. **Bitboard Conversion:** Use the `board_to_tensor()` utility from `architecture.py`.
3. **Training Loop:** Standard PyTorch training with:
   - Loss: `nn.MSELoss` (regression to game outcome)
   - Optimizer: `AdamW`, lr=1e-3, weight_decay=1e-4
   - Scheduler: `CosineAnnealingLR`
   - Batch size: 256
   - Epochs: configurable via CLI argument (default: 20)
4. **Checkpointing:** Save model weights to `src/model/checkpoints/best_model.pt` whenever validation loss improves.
5. **Logging:** Print epoch-level metrics (train_loss, val_loss, val_accuracy where accuracy = percentage of positions where predicted sign matches outcome sign).

**Constraints:**
- Must accept CLI arguments: `--data-dir`, `--epochs`, `--batch-size`, `--lr`, `--device`.
- Train/validation split: 90/10 random split.
- Do NOT use any external training frameworks (no PyTorch Lightning, no HuggingFace Trainer). Use raw PyTorch.

---

#### `src/mcts/SearchTree.py` — Bounded Monte Carlo Tree Search

**Purpose:** Implements MCTS with neural network leaf evaluation (no random rollouts).

**Required Interface:**

```python
class SearchTree:
    def __init__(self, value_network: ValueCNN, simulations: int = 800, c_puct: float = 1.4):
        """
        Args:
            value_network: The trained ValueCNN for leaf evaluation.
            simulations: Max number of MCTS iterations per search call.
            c_puct: Exploration constant for UCB1 formula.
        """
        ...

    def search(self, board: chess.Board) -> chess.Move:
        """
        Run bounded MCTS and return the most-visited root child's move.

        MCTS Phases (per simulation):
          1. SELECTION: Traverse from root to a leaf using UCB1:
             UCB1(child) = Q(child) + c_puct * sqrt(ln(N_parent) / N_child)
          2. EXPANSION: Expand the leaf node by generating all legal moves.
          3. EVALUATION: Run value_network on the leaf board state.
             Do NOT perform random rollouts. The CNN IS the evaluation.
          4. BACKPROPAGATION: Update Q-values and visit counts up the tree.

        Return the move corresponding to the root child with the highest visit count.
        """
        ...
```

**Constraints:**
- Tree nodes should store: `board` (or a compact hash), `move` (the move that led here), `parent`, `children`, `visit_count (N)`, `total_value (W)`, `mean_value (Q = W/N)`.
- The search MUST be bounded by the `simulations` parameter. Do NOT add time-based cutoffs unless explicitly requested later.
- Use `board.copy()` when expanding to avoid mutating parent state.
- The value network evaluation must respect the side to move (negate value if it's black's turn, since the CNN outputs from white's perspective).
- Implement tree reuse: if the opponent's response is in the existing tree, re-root instead of discarding.

---

#### `scripts/run_tournament.py` — Tournament Automation

**Purpose:** Orchestrates round-robin matches between agent variants and writes structured results.

**Required Functionality:**
1. Instantiate all agents: `LayaAgent`, `StockfishAgent`, `HybridAgent` (at multiple τ values: 0.5, 0.6, 0.7, 0.75, 0.8, 0.9).
2. Play a configurable number of games (default: 100) per pairing, alternating colors.
3. Per game, log: `white_agent`, `black_agent`, `result` (1-0, 0-1, 1/2-1/2), `total_moves`, `avg_time_per_move_ms`, `avg_nodes_per_move`, `system1_percentage` (for Hybrid agents), `blunders` (moves where eval drops > 200 centipawns vs Stockfish).
4. Output: Write all game records to `data/results/tournament_YYYYMMDD_HHMMSS.csv`.
5. Summary: Print Elo estimates (using Bayesian Elo or simple win-rate Elo) to stdout after completion.

**Constraints:**
- Must use `argparse` for CLI configuration: `--games`, `--tau-values`, `--stockfish-path`, `--stockfish-depth`, `--output-dir`.
- Must support resumption: if the CSV already exists, append rather than overwrite.
- Enforce time controls per move (e.g., 5 seconds max per move for all agents).

---

#### `notebooks/plot_metrics.ipynb` — Visualization

**Purpose:** Reads tournament CSVs and generates publication-quality figures.

**Required Plots:**
1. **Elo vs. τ curve:** X-axis = τ threshold, Y-axis = estimated Elo. Show Stockfish and Laya-only as horizontal baselines.
2. **Compute efficiency:** Scatter plot of Elo vs. average nodes searched per move.
3. **Blunder rate by game phase:** Grouped bar chart (opening/middle/endgame) × agent.
4. **System 1 vs System 2 utilization:** Stacked area chart showing % of moves handled by each system across τ values.
5. **Search depth distribution:** Violin plot of MCTS search depths when System 2 is triggered.

**Constraints:**
- Use `matplotlib` and `seaborn`. Do NOT use `plotly` or interactive plotting libraries.
- All figures must be saved to `paper/figures/` in both PNG (300 DPI) and PDF formats.
- Use a consistent, publication-appropriate style (`seaborn-v0_8-whitegrid` or similar).

---

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- SECTION 3: DEVELOPMENT ROADMAP (1-Week Sprint)                        -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

## [DEVELOPMENT ROADMAP — 1-Week Sprint]

### Phase 1: Data & Baselines (Days 1–2)

| Task | File(s) | Acceptance Criteria |
|------|---------|-------------------|
| Install dependencies | `requirements.txt` | `pip install -r requirements.txt` succeeds |
| Download Lichess Elite DB | `data/raw/` | At least 100,000 games in PGN format |
| Implement Stockfish wrapper | `src/engine/Stockfish.py` | `StockfishAgent.evaluate(board)` returns valid move + eval |
| Implement Laya inference | `src/engine/LayaAgent.py` | `LayaAgent.evaluate(board)` returns valid move + confidence |
| Verify both agents play legal chess | — | 10-game smoke test with no illegal moves |

### Phase 2: Deep Learning (Days 3–4)

| Task | File(s) | Acceptance Criteria |
|------|---------|-------------------|
| Define ValueCNN architecture | `src/model/architecture.py` | Model forward pass on random (1, 12, 8, 8) tensor succeeds, output shape = (1, 1) |
| Implement `board_to_tensor` | `src/model/architecture.py` | Round-trip test: tensor correctly encodes piece placement |
| Implement PGN data loader | `src/model/train.py` | DataLoader yields (tensor, label) batches from PGN files |
| Train value network | `src/model/train.py` | Validation loss < 0.5 MSE after 20 epochs on 100K positions |
| Save best checkpoint | `src/model/checkpoints/` | `best_model.pt` exists and loads correctly |

### Phase 3: Neuro-Symbolic Integration (Days 5–6)

| Task | File(s) | Acceptance Criteria |
|------|---------|-------------------|
| Implement MCTS with CNN evaluation | `src/mcts/SearchTree.py` | `SearchTree.search(board)` returns a legal move within 800 simulations |
| Implement HybridAgent routing | `src/engine/HybridAgent.py` | At τ=0.5, some moves go to System 1, some to System 2 |
| Integration smoke test | — | HybridAgent plays a complete 40-move game with no crashes |
| Verify metadata logging | — | Every move returns the required metadata dict |

### Phase 4: Evaluation & Publication (Day 7)

| Task | File(s) | Acceptance Criteria |
|------|---------|-------------------|
| Run 100-game round-robin | `scripts/run_tournament.py` | Tournament CSV written to `data/results/` |
| Generate all plots | `notebooks/plot_metrics.ipynb` | 5 figures saved to `paper/figures/` in PNG + PDF |
| Draft IEEE paper | `paper/main.tex` | Compiles with `pdflatex`, includes all figures |
| Final Elo analysis | `data/results/` | Elo estimates printed, hypothesis evaluated |

---

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- SECTION 4: TECHNICAL CONSTRAINTS FOR AGENTS                           -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

## [TECHNICAL CONSTRAINTS FOR AGENTS]

> **READ THIS ENTIRE SECTION BEFORE WRITING ANY CODE.** These are hard constraints. Violating them will produce code that is incompatible with the rest of the project.

### 🚫 Absolute Prohibitions

1. **Do NOT build a GUI or web frontend.** This is a research project. All interaction is via CLI, scripts, and Jupyter notebooks. No Flask, no Django, no Streamlit, no Gradio.

2. **Do NOT implement custom move generation or legality checking.** Use `python-chess` (`import chess`) for ALL board manipulation, move generation, FEN parsing, and game state management. The `chess.Board` class is the single source of truth.

3. **Do NOT use external training frameworks.** No PyTorch Lightning, no HuggingFace Trainer, no Keras. Use raw `torch.nn`, `torch.optim`, and `torch.utils.data`.

4. **Do NOT use interactive/web-based plotting.** Use `matplotlib` and `seaborn` only. No Plotly, no Bokeh, no Altair. Figures must be static and publication-ready.

5. **Do NOT add opening book lookups or endgame tablebases** to the HybridAgent unless explicitly requested. The experiment must isolate the confidence-gating mechanism.

### ✅ Mandatory Conventions

1. **All engine outputs must be CSV-formatted** for the research paper pipeline. The `run_tournament.py` script must write structured CSVs with headers. No JSON, no SQLite, no pickle for tournament results.

2. **Optimize `train.py` for standard GPU execution.** Use `torch.cuda.is_available()` for device detection. Support single-GPU training. Do NOT add multi-GPU / distributed training unless explicitly requested.

3. **Use type hints everywhere.** All function signatures must have complete type annotations. Use `from __future__ import annotations` for forward references.

4. **Follow the exact interface contracts** specified in Section 2. The `evaluate()` and `search()` method signatures are canonical. Downstream code (tournament runner, notebook) depends on them.

5. **Bitboard encoding is 12 planes of 8×8.** The plane order is `[P, N, B, R, Q, K, p, n, b, r, q, k]` (white pieces first, then black, in standard piece-value order). This is non-negotiable — the trained model weights depend on this encoding.

6. **Side-to-move awareness:** The CNN always evaluates from white's perspective. When it's black's turn, the value must be negated before use in MCTS. This inversion happens in `SearchTree.py`, NOT in `architecture.py`.

### 🔧 Environment & Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.11 | Type hint features, performance |
| `python-chess` | ≥ 1.10.0 | Board representation, move generation |
| `torch` | ≥ 2.0.0 | Value CNN training and inference |
| `numpy` | ≥ 1.24.0 | Tensor manipulation |
| `pandas` | ≥ 2.0.0 | CSV I/O for tournament results |
| `matplotlib` | ≥ 3.7.0 | Static plotting |
| `seaborn` | ≥ 0.12.0 | Statistical visualizations |
| `stockfish` | ≥ 3.28.0 | Python wrapper for Stockfish UCI |
| `tqdm` | ≥ 4.65.0 | Progress bars for training and tournaments |
| `zstandard` | ≥ 0.21.0 | Decompressing Lichess `.zst` archives |

### 📁 Data Flow

```
Lichess PGN (data/raw/)
        │
        ▼
   train.py ──────► best_model.pt (src/model/checkpoints/)
        │                     │
        ▼                     ▼
  architecture.py ◄──── SearchTree.py ◄──── HybridAgent.py
                                                   │
                                                   ▼
                                         run_tournament.py
                                                   │
                                                   ▼
                                          tournament.csv (data/results/)
                                                   │
                                                   ▼
                                         plot_metrics.ipynb
                                                   │
                                                   ▼
                                          figures/ (paper/figures/)
                                                   │
                                                   ▼
                                            main.tex (paper/)
```

### 🧪 Testing Strategy

- **Unit tests are not scaffolded** but are encouraged. If you write tests, place them in a `tests/` directory at the project root.
- **Smoke tests:** Every agent must be able to play a complete game (up to 200 moves or adjudication) without crashing.
- **Consistency check:** `board_to_tensor(board)` must produce identical output for identical board states regardless of move history (i.e., it's a pure function of the position, not the game history — except for castling rights and en passant, which are encoded in the `chess.Board` state).

---

## Quick Start (for Human Developers)

```bash
# Clone and install
git clone <repo-url> && cd neuro-gated-chess
pip install -r requirements.txt

# Download Stockfish binary and place it in PATH
# Download Lichess Elite DB to data/raw/

# Train the value network
python src/model/train.py --data-dir data/raw/ --epochs 20 --device cuda

# Run a tournament
python scripts/run_tournament.py --games 100 --stockfish-path stockfish

# Generate plots
jupyter notebook notebooks/plot_metrics.ipynb
```

---

## License

TBD — To be determined before publication.

---

<details>
<summary><strong>Agent Meta-Notes (collapse for human readers)</strong></summary>

### Inter-Module Dependency Graph

```
LayaAgent ─────────┐
                    ├──► HybridAgent ──► run_tournament.py ──► CSV ──► plot_metrics.ipynb
SearchTree ────────┘         ▲
    │                        │
    └── ValueCNN ◄── train.py
                        ▲
                        │
              Lichess PGN data

StockfishAgent ──────────────► run_tournament.py (as baseline opponent)
```

### Key Design Decision Log

1. **Why no random rollouts in MCTS?** Random rollouts are computationally expensive and poorly calibrated in chess. Neural evaluation at leaf nodes (as in AlphaZero) is strictly superior. The CNN replaces the simulation phase entirely.

2. **Why MSE loss instead of cross-entropy?** We treat game outcome prediction as regression to a continuous value in [-1, 1], not classification into {win, draw, loss}. This allows the model to express uncertainty through intermediate values. MSE is standard for value networks in this paradigm (see Silver et al., 2017).

3. **Why 12 planes, not 119 planes (like AlphaZero)?** We deliberately use a simpler encoding (just piece placement + castling/en passant from `chess.Board`) to keep the model small (~1-3M params) and trainable on commodity hardware. The 119-plane encoding includes move history, repetitions, and other features that require additional bookkeeping.

4. **Why τ as a single scalar, not position-dependent?** Initial experiments use a fixed threshold. Position-dependent τ (e.g., higher τ in endgames) is a future extension and should NOT be implemented in Phase 1-4 unless explicitly requested.

</details>
