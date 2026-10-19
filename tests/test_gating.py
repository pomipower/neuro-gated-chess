from __future__ import annotations

import math

import pytest

from src.engine.gating import (
    AlwaysGate,
    ConfidenceGate,
    NeverGate,
    RandomGate,
    margin,
    norm_entropy_conf,
    quantile_threshold,
    top1_softmax,
)


def test_margin() -> None:
    assert margin([0.9, 0.5, 0.1]) == pytest.approx(0.4)
    assert margin([0.5]) == 1.0                      # forced move


def test_top1_softmax_uniform_and_monotone() -> None:
    assert top1_softmax([0.5, 0.5, 0.5, 0.5], T=0.2) == pytest.approx(0.25)
    assert top1_softmax([0.5], T=0.2) == 1.0
    small_gap = top1_softmax([0.55, 0.50, 0.30], T=0.2)
    big_gap = top1_softmax([0.90, 0.50, 0.30], T=0.2)
    assert big_gap > small_gap
    with pytest.raises(ValueError):
        top1_softmax([0.5, 0.4], T=0.0)


def test_norm_entropy_conf_extremes() -> None:
    assert norm_entropy_conf([0.5, 0.5, 0.5], T=0.2) == pytest.approx(0.0, abs=1e-9)
    assert norm_entropy_conf([0.999, 0.001, 0.001], T=0.05) > 0.9
    assert norm_entropy_conf([0.3], T=0.2) == 1.0


def test_quantile_threshold() -> None:
    values = [float(v) for v in range(10)]
    tau = quantile_threshold(values, 0.3)
    assert tau == 2.0
    assert sum(v <= tau for v in values) / len(values) == pytest.approx(0.3)
    assert quantile_threshold(values, 0.0) == -math.inf
    assert quantile_threshold(values, 1.0) == math.inf


def test_confidence_gate_boundary_and_nan() -> None:
    gate = ConfidenceGate(0.5)
    assert gate.use_system2(0.5) is True             # confidence <= tau -> System 2
    assert gate.use_system2(0.51) is False
    assert gate.use_system2(float("nan")) is True


def test_random_gate_rate_and_reproducibility() -> None:
    a, b = RandomGate(0.25, seed=7), RandomGate(0.25, seed=7)
    draws_a = [a.use_system2(0.9) for _ in range(20000)]
    draws_b = [b.use_system2(0.9) for _ in range(20000)]
    assert draws_a == draws_b
    assert sum(draws_a) / len(draws_a) == pytest.approx(0.25, abs=0.02)


def test_always_and_never() -> None:
    assert AlwaysGate().use_system2(0.99) is True
    assert NeverGate().use_system2(0.0) is False
