#!/usr/bin/env bash
# Build / refresh the organised project workspace on the Windows D: drive (D:\ChessBot-MDM).
# Safe to re-run any time: it only copies (never deletes anything in the repo) and overwrites older copies.
#
# Run from the repo root inside WSL:   bash scripts/sync_workspace.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WS="${1:-/mnt/d/ChessBot-MDM}"
cd "$REPO"

if [ ! -d "$(dirname "$WS")" ]; then echo "$(dirname "$WS") not visible (is the D: drive mounted in WSL?)"; exit 1; fi
if [ -n "$(git status --porcelain -- src scripts configs docs demo paper 2>/dev/null)" ]; then
  echo "NOTE: you have uncommitted changes in the repo; the code snapshot uses the last commit."
  echo "      (commit first with: git add -A src scripts configs docs demo && git commit -m 'update')"
fi

mkdir -p "$WS"/{01_Code,02_Results,03_Figures,04_Demo,05_Paper/drafts,06_Docs,07_Presentation,08_Logs}

# 01 Code: full git history (bundle), a plain source zip, the trained value net and the frozen configs
git bundle create "$WS/01_Code/neuro-gated-chess.bundle" --all >/dev/null 2>&1
git archive --format=zip -o "$WS/01_Code/neuro-gated-chess-source.zip" HEAD
mkdir -p "$WS/01_Code/model_and_configs"
cp -u src/model/checkpoints/best_model.pt "$WS/01_Code/model_and_configs/" 2>/dev/null || true
cp -u configs/*.yaml "$WS/01_Code/model_and_configs/"
cp -u requirements.txt "$WS/01_Code/"

# 02 Results: one folder per study, raw CSVs exactly as produced
copy_run() { [ -d "data/results/$1" ] && mkdir -p "$WS/02_Results/$2" && cp -ru "data/results/$1" "$WS/02_Results/$2/"; return 0; }
copy_run bench_test   "StudyA_position_benchmark"
copy_run tour_1       "StudyB_tournament"
copy_run tour_t2a     "StudyB_tournament"
copy_run tour_t2b     "StudyB_tournament"
copy_run puzzles_1    "StudyC_puzzles"
for d in data/results/calib_* ; do [ -d "$d" ] && copy_run "$(basename "$d")" "gate_calibration_val"; done
for d in data/results/sweep_* ; do [ -d "$d" ] && copy_run "$(basename "$d")" "prior_temperature_sweep_val"; done
copy_run tour_pilot   "pilots"

# 03 Figures (PNG for slides, PDF for the paper, CSV of every plotted number, LaTeX tables)
cp -ru paper/figures/. "$WS/03_Figures/"

# 04 Demo
cp -u demo/game_viewer.html "$WS/04_Demo/" 2>/dev/null || true

# 05 Paper + 06 Docs + 08 Logs
cp -u paper/main.tex "$WS/05_Paper/" 2>/dev/null || true
cp -u docs/PAPER_OUTLINE.md "$WS/05_Paper/" 2>/dev/null || true
cp -ru docs/. "$WS/06_Docs/"
cp -u README.md "$WS/06_Docs/REPO_README.md"
cp -ru logs/. "$WS/08_Logs/" 2>/dev/null || true

cat > "$WS/README.md" <<'EOF'
# ChessBot-MDM — Neuro-gated chess (System 1 / System 2)

AIML minor-degree mini project. Team: Om Kulkarni (repo owner), Rohan Bhide (build & experiments),
Shaunak & Parth (paper). Repo: github.com/pomipower/neuro-gated-chess

## Folders
| Folder | What is in it | Who needs it |
|---|---|---|
| 01_Code | `neuro-gated-chess.bundle` (full git history), source zip, trained value net (`best_model.pt`), frozen configs | Om (to update GitHub), anyone re-running |
| 02_Results | raw result CSVs per study (A = 3000 test positions, B = tournament, C = 1000 puzzles, + val calibration/sweeps) | paper writers (numbers), reviewers |
| 03_Figures | F1–F6 + EXP_* figures: `.png` for slides, `.pdf` for the paper, `.csv` with the plotted numbers, `.tex` tables | Shaunak & Parth, slides |
| 04_Demo | `game_viewer.html`: replay viewer with the bot's thought process (★ games). Double-click to open, works offline | presentation |
| 05_Paper | LaTeX skeleton, paper outline, `drafts/` for your drafts | Shaunak & Parth |
| 06_Docs | pre-registered protocol, spec, DEVIATIONS log, runbook | everyone (method section) |
| 07_Presentation | slides go here | presenters |
| 08_Logs | console logs of every run | debugging / proof of runs |

## Headline results (test set, frozen before testing)
- **H1 supported:** always searching improves move quality: ACPL 107.9 → 103.1 (−4.8 cp, 95% CI −7.4 to −2.5, p = 0.001), at 3.2× compute (468 → 1499 ms/move).
- **H2 not supported:** the confidence gate is no better than a random gate at any search rate (all CIs include 0).
- **Why (exploratory):** search gains concentrate in sharp positions (+20 cp when one move is >300 cp better) and endgames (+11 cp), where Laya is *confident*; confidence vs gain Spearman ρ = 0.01. "Uncertainty is not criticality."
- **Headroom (exploratory):** an oracle gate searching 10% of positions reaches ACPL 100.5, better than always searching, at ~1/3 of the cost.
- **Games:** Laya 904, always_s2 887, hybrid_s2_50 879 Elo (tied, overlapping CIs); uniform MCTS 734 (clearly weaker). Relative ratings anchored to Stockfish UCI_Elo 1320.
- **Puzzles:** search adds nothing (solve rates within ±2.5 pts across rating buckets).
- **Limitation:** deterministic agents; 74% of round-robin games ended in threefold repetition.

## Updating this folder
Inside WSL, from the repo:  `bash scripts/sync_workspace.sh`  (re-copies everything; safe to re-run).

## For Om: putting the exact code that produced these results on GitHub
The bundle is the full local repo (its history starts from the project zip, so it is not a descendant of `main`).
Push it as its own branch and make it the default / merge later:

    git clone D:/ChessBot-MDM/01_Code/neuro-gated-chess.bundle ngc && cd ngc
    git remote add github https://github.com/pomipower/neuro-gated-chess.git
    git push github HEAD:experiments
EOF

echo "Workspace ready: D:\\ChessBot-MDM"
du -sh "$WS"/* 2>/dev/null | sed "s#$WS/##"
