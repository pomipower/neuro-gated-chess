"""Confidence functions and gates for the System 1 / System 2 router.

Inputs are Laya's per-legal-move win chances in [0, 1] (side to move's perspective). A position with a single
legal move is forced and gets confidence 1.0 from every function. Higher confidence = more certain.

Hybrid rule (see docs/EXPERIMENT_PROTOCOL.md): play the System 1 move if confidence > tau, else run System 2.
"""

from __future__ import annotations

import math
import random
from typing import Protocol, Sequence

import numpy as np

_EPS = 1e-6


def _logit(w: float) -> float:
    w = min(max(float(w), _EPS), 1.0 - _EPS)
    return math.log(w / (1.0 - w))


def _softmax_of_logits(win_chances: Sequence[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    z = [_logit(w) / temperature for w in win_chances]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    total = sum(e)
    return [v / total for v in e]


def margin(win_chances: Sequence[float]) -> float:
    """Top-1 minus top-2 win chance. 1.0 if there is only one legal move."""
    if len(win_chances) == 0:
        raise ValueError("win_chances must be non-empty")
    if len(win_chances) == 1:
        return 1.0
    top = sorted(win_chances, reverse=True)
    return float(top[0] - top[1])


def top1_softmax(win_chances: Sequence[float], T: float) -> float:
    """Probability of the best move under softmax(logit(w) / T). 1.0 if only one legal move."""
    if len(win_chances) == 0:
        raise ValueError("win_chances must be non-empty")
    if len(win_chances) == 1:
        return 1.0
    return max(_softmax_of_logits(win_chances, T))


def norm_entropy_conf(win_chances: Sequence[float], T: float) -> float:
    """1 - H(p)/log(n) with p = softmax(logit(w)/T). 1.0 if only one legal move."""
    n = len(win_chances)
    if n == 0:
        raise ValueError("win_chances must be non-empty")
    if n == 1:
        return 1.0
    p = _softmax_of_logits(win_chances, T)
    entropy = -sum(q * math.log(q) for q in p if q > 0.0)
    return 1.0 - entropy / math.log(n)


def quantile_threshold(confidences: Sequence[float], s2_rate: float) -> float:
    """Threshold tau such that about `s2_rate` of the given confidences satisfy confidence <= tau.

    s2_rate <= 0 -> -inf (never System 2); s2_rate >= 1 -> +inf (always System 2). With ties at the
    threshold the achieved rate can exceed `s2_rate`; always report the rate achieved on the test split.
    """
    values = np.sort(np.asarray(list(confidences), dtype=float))
    n = len(values)
    if n == 0:
        raise ValueError("confidences must be non-empty")
    k = int(round(s2_rate * n))
    if k <= 0:
        return float("-inf")
    if k >= n:
        return float("inf")
    return float(values[k - 1])


class Gate(Protocol):
    def use_system2(self, confidence: float) -> bool: ...


class ConfidenceGate:
    """System 2 iff confidence <= tau (NaN confidence is treated as 'unsure' -> System 2)."""

    def __init__(self, tau: float) -> None:
        self.tau = float(tau)

    def use_system2(self, confidence: float) -> bool:
        if math.isnan(confidence):
            return True
        return confidence <= self.tau


class RandomGate:
    """Control: System 2 with probability `rate`, independent of confidence. Seeded for reproducibility."""

    def __init__(self, rate: float, seed: int) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be in [0, 1]")
        self.rate = float(rate)
        self._rng = random.Random(seed)

    def use_system2(self, confidence: float) -> bool:
        return self._rng.random() < self.rate


class AlwaysGate:
    def use_system2(self, confidence: float) -> bool:
        return True


class NeverGate:
    def use_system2(self, confidence: float) -> bool:
        return False
