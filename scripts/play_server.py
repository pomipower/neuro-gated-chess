"""Play against the bot in your browser and see its thought process for every move.

Starts a small local web server (standard library only) that serves demo/play.html. You move by clicking; the
bot answers with the real agent from src/engine/factory.py (default: v3_learned_25, the V3 agent) and the page
shows what it "thought": Laya's shortlist and confidence, the gate's decision (for the V3 learned gate: score vs
threshold and which features pushed it), how the search split its simulations and the line it expects,
repetition contempt, and whether the safety net vetoed a move.

The explanation is recomputed from the same objects the agent used (Laya's scores, the search result) with the
same functions, and each move is checked: `consistent` is false if the recomputed final move differs from the
move the agent actually played (it should never happen).

Run inside WSL from the repo root, with the project's venv:
    python -m scripts.play_server                       # then open http://localhost:8765 in your browser
    python -m scripts.play_server --agent v2_full --port 8765
Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import chess
import numpy as np

from src.engine.AgentV2 import V2Agent, repeats
from src.engine.factory import Resources, build_agent
from src.engine.tactics import criticality, material, safe_choice
from src.utils import load_config, load_gate_config, resolve

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "demo" / "play.html"
PIECES = ROOT / "demo" / "pieces"
AGENTS = ["v3_learned_25", "v2_full", "v2_safe", "classical_full", "laya_only"]
FEATURE_NAMES = {
    "w_best": "Laya's best win chance", "margin12": "gap between Laya's top 2 moves",
    "margin13": "gap between Laya's top 1 and 3", "laya_entropy": "how spread out Laya's choices are",
    "confidence": "Laya's confidence", "n_legal": "number of legal moves", "n_captures": "number of captures",
    "in_check": "in check", "hanging": "a piece is hanging", "opportunity": "a profitable capture exists",
    "few_moves": "few legal moves", "endgame": "endgame", "npm_total": "pieces left (non-pawn material)",
    "material_balance": "material balance", "ply": "move number", "safety_override": "safety net would veto Laya",
    "top_tactical_vs_now": "material outcome of Laya's move", "best_tactical_vs_top": "best material outcome vs Laya's",
}


def san_line(board: chess.Board, moves: list[chess.Move]) -> list[str]:
    b, out = board.copy(stack=False), []
    for m in moves:
        if m not in b.legal_moves:
            break
        out.append(b.san(m))
        b.push(m)
    return out


def explain(agent: Any, board: chess.Board, move: chess.Move, info: Any, out: Any, res: Any) -> dict[str, Any]:
    """Recompute, step by step, why `agent` played `move` in `board` (board = position before the move)."""
    d = info.as_dict()
    th: dict[str, Any] = {"move": move.uci(), "san": board.san(move), "system": d["system"],
                          "special": d.get("special", ""), "ms": round(d["time_elapsed_ms"]),
                          "laya_ms": round(d.get("laya_ms", 0)), "search_ms": round(d.get("search_ms", 0))}
    if out is None:                                   # tablebase or mate in one: decided before Laya
        th["consistent"] = True
        return th
    ranking = sorted(out.scores, key=lambda m: (-out.scores[m], m.uci()))
    th["laya"] = {"top": [[m.uci(), board.san(m), round(out.scores[m], 4)] for m in ranking[:6]],
                  "best": ranking[0].uci(), "best_san": board.san(ranking[0]), "confidence": round(out.confidence, 4),
                  "n": out.n_scored, "ms": round(out.elapsed_ms), "played_winchance": round(out.scores.get(move, 0.0), 4)}
    preference = ranking
    if res is not None:
        tot = max(1, sum(res.visits.values()))
        top = sorted(res.visits.items(), key=lambda kv: (-kv[1], -res.q.get(kv[0], -9.0)))[:6]
        th["search"] = {"sims": res.sims, "depth": res.max_depth, "cnn_evals": res.cnn_evals,
                        "visits": [[m.uci(), board.san(m), n, round(n / tot, 4), round(res.q.get(m, float("nan")), 4)]
                                   for m, n in top if n > 0],
                        "pv": san_line(board, res.pv), "best": res.move.uci()}
        preference = sorted(board.legal_moves, key=lambda m: (-res.visits.get(m, 0), -res.q.get(m, -9.0),
                                                               -out.scores.get(m, 0.0), m.uci()))
    if not isinstance(agent, V2Agent):                # V1-style agent (laya_only): no gate, no safety net
        th["gate"] = {"kind": "none", "searched": res is not None}
        th["consistent"] = (preference[0] == move)
        return th

    o = agent.o
    crit = criticality(board, o.get("few_legal_moves", 12), o.get("endgame_npm", 13))
    flags = {k: bool(getattr(crit, k)) for k in ("in_check", "hanging", "opportunity", "few_moves", "endgame")}
    gate = {"kind": o.get("gate", "critical"), "searched": res is not None, "flags": flags, "critical": crit.critical,
            "sharp": crit.sharp, "sims": res.sims if res is not None else 0}
    if gate["kind"] == "learned":
        from src.engine.gate_features import gate_features

        lg = o["learned_gate"]
        f = gate_features(board, out.scores, out.confidence, o.get("safety_opts", {}),
                          o.get("few_legal_moves", 12), o.get("endgame_npm", 13))
        x = (np.array([f[k] for k in lg.features]) - lg.mean) / lg.std
        contrib = x * lg.coef
        order = np.argsort(-np.abs(contrib))[:6]
        gate.update({"score": round(lg.score(f), 4), "threshold": round(lg.threshold, 4),
                     "contrib": [[FEATURE_NAMES.get(lg.features[i], lg.features[i]), round(float(f[lg.features[i]]), 3),
                                  round(float(contrib[i]), 3)] for i in order]})
    th["gate"] = gate

    w_best = out.scores[ranking[0]]
    contempt = o.get("contempt", 0.0) if w_best >= o.get("contempt_min_winchance", 0.55) else 0.0
    rep = {m for m in board.legal_moves if repeats(board.copy(), m)} if board.move_stack else set()
    if contempt > 0 and rep and preference[0] in rep:
        tol = o.get("contempt_tolerance", 0.06)
        alt = next((m for m in preference[1:] if m not in rep and out.scores.get(m, 0.0) >= w_best - tol), None)
        if alt is not None:
            th["contempt"] = {"avoided": board.san(preference[0]), "chosen": board.san(alt)}
            preference = [alt] + [m for m in preference if m != alt]
    if int(o.get("laya_lookahead", 0) or 0) > 1 and res is None:
        th["note"] = "Laya look-ahead re-ranked the top moves (not shown)."

    final = preference[0]
    if o.get("safety", True):
        s = o.get("safety_opts", {})
        r = safe_choice(board, preference, s.get("candidates", 6), s.get("full_plies", 2), s.get("margin_cp", 150),
                        s.get("max_nodes", 6000), s.get("check_plies", 0))
        top = preference[0]
        best = max(r.scores.values())
        th["safety"] = {"top": top.uci(), "top_san": board.san(top), "top_score": r.scores[top], "best_score": best,
                        "now": material(board, board.turn), "override": r.overridden, "final_san": board.san(r.move),
                        "margin": s.get("margin_cp", 150),
                        "scores": [[board.san(m), v] for m, v in sorted(r.scores.items(), key=lambda kv: -kv[1])[:6]]}
        final = r.move
    th["consistent"] = (final == move)
    return th


class Session:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.res = Resources(cfg, load_gate_config())
        self.agents: dict[str, Any] = {}
        self.lock = threading.Lock()
        self.reset("white", AGENTS[0])

    def agent(self, name: str) -> Any:
        if name not in self.agents:
            self.agents[name] = build_agent(name, self.res)
        return self.agents[name]

    def reset(self, color: str, agent: str) -> None:
        self.board = chess.Board()
        self.human = chess.WHITE if color == "white" else chess.BLACK
        self.agent_name = agent if agent in AGENTS else AGENTS[0]
        self.thoughts: dict[int, dict[str, Any]] = {}
        self.sans: list[str] = []

    def bot_move(self) -> None:
        if self.board.is_game_over(claim_draw=True) or self.board.turn == self.human:
            return
        agent = self.agent(self.agent_name)
        before = self.board.copy()
        move, info, out, res = agent.decide(before.copy())
        th = explain(agent, before, move, info, out, res)
        th["ply"] = self.board.ply()
        self.thoughts[self.board.ply()] = th
        self.sans.append(self.board.san(move))
        self.board.push(move)

    def human_move(self, uci: str) -> str | None:
        if self.board.turn != self.human or self.board.is_game_over(claim_draw=True):
            return "not your turn"
        try:
            m = chess.Move.from_uci(uci)
        except ValueError:
            return "bad move"
        if m not in self.board.legal_moves:
            return "illegal move"
        self.sans.append(self.board.san(m))
        self.board.push(m)
        return None

    def undo(self) -> None:
        while self.board.move_stack:
            ply = self.board.ply() - 1
            self.board.pop()
            self.sans.pop()
            self.thoughts.pop(ply, None)
            if self.board.turn == self.human:
                break

    def state(self) -> dict[str, Any]:
        b = self.board
        outcome = b.outcome(claim_draw=True)
        last = b.peek().uci() if b.move_stack else None
        return {"fen": b.fen(), "turn": "white" if b.turn else "black", "human": "white" if self.human else "black",
                "agent": self.agent_name, "agents": AGENTS, "sans": self.sans, "last": last,
                "legal": [m.uci() for m in b.legal_moves] if b.turn == self.human and outcome is None else [],
                "check": b.is_check(), "over": outcome is not None,
                "result": outcome.result() if outcome else None,
                "termination": outcome.termination.name.lower().replace("_", " ") if outcome else None,
                "thoughts": {str(k): v for k, v in self.thoughts.items()}}


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:       # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path.startswith("/pieces/") and self.path.endswith(".svg"):
                name = Path(self.path).name
                f = PIECES / name
                if f.parent == PIECES and f.exists():
                    self._send(200, f.read_bytes(), "image/svg+xml")
                else:
                    self._send(404, b"not found", "text/plain")
            elif self.path == "/api/state":
                with session.lock:
                    self._json(session.state())
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            with session.lock:
                try:
                    if self.path == "/api/new":
                        session.reset(data.get("color", "white"), data.get("agent", AGENTS[0]))
                        session.bot_move()
                    elif self.path == "/api/move":
                        err = session.human_move(str(data.get("uci", "")))
                        if err:
                            return self._json({"error": err, **session.state()}, 400)
                        session.bot_move()
                    elif self.path == "/api/undo":
                        session.undo()
                    else:
                        return self._json({"error": "unknown endpoint"}, 404)
                except Exception as e:                       # show the error in the page instead of hanging it
                    import traceback

                    traceback.print_exc()
                    return self._json({"error": f"{type(e).__name__}: {e}", **session.state()}, 500)
                self._json(session.state())

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/v3.yaml")
    p.add_argument("--agent", default=AGENTS[0], choices=AGENTS)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--standin", action="store_true", help="material stand-in instead of LayaChess (testing only)")
    args = p.parse_args()
    cfg = load_config(args.config)
    if cfg["v2"].get("prior_temperature") != 0.1:
        print(f"WARNING: v2.prior_temperature = {cfg['v2'].get('prior_temperature')} (the frozen value is 0.1)")
    if args.standin:
        cfg["laya"]["model_name"], cfg["laya"]["device"] = "stand-in:material", "cpu"
    print("loading models (Laya, value net, tablebases)...")
    t0 = time.time()
    session = Session(cfg)
    session.agent_name = args.agent
    session.agent(args.agent)
    _ = session.res.laya
    print(f"ready in {time.time() - t0:.0f}s -> open http://localhost:{args.port}   (Ctrl+C to stop)")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.res.close()


if __name__ == "__main__":
    main()
