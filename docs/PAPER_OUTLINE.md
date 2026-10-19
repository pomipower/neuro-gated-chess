# Paper outline (IEEE conference format, `paper/main.tex`)

Working title: **"When Should a Chess Agent Think? Confidence-Gated System 1 / System 2 Move Selection"**

Write every section now; leave numbers as `\todo{}`. The protocol (`docs/EXPERIMENT_PROTOCOL.md`) is the source for Methods and Experimental Setup, so most of those sections can be finished before any result exists.

## Sections

1. **Abstract** (~150 words). Problem (uniform compute per move is wasteful), method (Laya's per-move win chances → confidence → gate to PUCT search with a value CNN), evaluation (judged positions, random-gate controls, anchored tournament), result placeholders.
2. **Introduction.** Dual-process framing (Kahneman); adaptive computation; contributions as a bullet list: (i) a gated agent reusing System 1 scores as the search prior, (ii) a protocol that separates *gate quality* from *search quality* using matched random-gate controls, (iii) analysis of where the System 1 model fails.
3. **Related work.** (a) Search + learned evaluation in games; (b) search-free / policy-only chess; (c) dual-process and "thinking fast and slow" AI; (d) adaptive computation and early exit; (e) calibration of neural networks; (f) human-like chess and time management.
4. **Method.** Encoding (18 planes, table), value network (diagram, 2.32M params), LayaChess System 1, confidence functions, PUCT with Laya root prior (equation), gate rule, cost accounting.
5. **Experimental setup.** Data and month splits; judge and CPL definition; agents table; Studies A/B/C; statistics; hardware.
6. **Results.** H1 (F1, F2), H2/H3 (F3, main figure), H4 (F4), H5 (F5), Study B Elo table (H6), puzzles table.
7. **Discussion.** What confidence captures (forced positions? quiet positions?), when search hurts, cost breakdown. Report honestly whether System 1 (one Laya pass per legal move) was actually cheaper than System 2 on our hardware (protocol §8).
8. **Limitations / threats to validity.** Copy from protocol §7.
9. **Conclusion and future work.** Position-dependent τ, interior-node priors, tree reuse, a learned gate.

## Reference starting list (verify every entry's details before adding to the .bib)

Search with learned evaluation
1. Silver et al., "Mastering the game of Go without human knowledge," *Nature*, 2017.
2. Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play," *Science*, 2018.
3. Silver et al., "Mastering the game of Go with deep neural networks and tree search," *Nature*, 2016.
4. Schrittwieser et al., "Mastering Atari, Go, chess and shogi by planning with a learned model," *Nature*, 2020.
5. Kocsis and Szepesvári, "Bandit based Monte-Carlo planning," ECML, 2006.
6. Browne et al., "A survey of Monte Carlo tree search methods," *IEEE Trans. Comput. Intell. AI Games*, 2012.
7. Rosin, "Multi-armed bandits with episode context," *Annals of Mathematics and Artificial Intelligence*, 2011 (PUCT).
8. Lai, "Giraffe: Using deep reinforcement learning to play chess," arXiv, 2015.

Search-free chess / System 1 models
9. Ruoss et al., "Grandmaster-level chess without search," arXiv 2402.04494, 2024 (introduces ChessBench, which LayaChess was fine-tuned on).
10. McIlroy-Young et al., "Aligning superhuman AI with human behavior: Chess as a model system," KDD, 2020 (Maia).
11. LayaChess model card, `datafreak/laya-chess`, Hugging Face (cite as software, with revision hash).
    LayaChess engine repo (`github.com/devroopsaha744/LayaChess`): already includes MCTS that uses Laya for both priors and leaf values. **Discuss explicitly**: our System 2 is a separate cheap searcher, and our question is *when* to search (protocol §8).

Dual process
12. Kahneman, *Thinking, Fast and Slow*, 2011.
13. Anthony, Tian, Barber, "Thinking fast and slow with deep learning and tree search," NeurIPS, 2017 (**closest prior work; discuss explicitly**).
14. Booch et al., "Thinking fast and slow in AI," AAAI, 2021.

Adaptive computation / early exit
15. Graves, "Adaptive computation time for recurrent neural networks," arXiv, 2016.
16. Teerapittayanon, McDanel, Kung, "BranchyNet: Fast inference via early exiting from deep neural networks," ICPR, 2016.
17. Huang, Coulom, Lin, "Time management for Monte-Carlo tree search applied to the game of Go," TAAI, 2010.

Calibration and evaluation
18. Guo, Pleiss, Sun, Weinberger, "On calibration of modern neural networks," ICML, 2017.
19. Elo, *The Rating of Chessplayers, Past and Present*, 1978.
20. Hunter, "MM algorithms for generalized Bradley–Terry models," *Annals of Statistics*, 2004.
21. Efron and Tibshirani, *An Introduction to the Bootstrap*, 1993.

Software / data: Stockfish (cite version), python-chess, PyTorch, Lichess open database (CC0), Lichess Elite Database (nikonoel).
