from __future__ import annotations

import pytest

from tests.helpers import skip_if_not_implemented

torch = pytest.importorskip("torch")


@skip_if_not_implemented
def test_value_cnn_shape_range_and_size() -> None:
    from src.model.architecture import ValueCNN

    model = ValueCNN()
    out = model(torch.randn(4, 18, 8, 8))
    assert out.shape == (4, 1)
    assert float(out.abs().max()) <= 1.0
    n_params = sum(p.numel() for p in model.parameters())
    assert 2_000_000 <= n_params <= 2_700_000        # spec: about 2.32M


@skip_if_not_implemented
def test_head_keeps_spatial_information() -> None:
    """Guards the v1 bug: pooling the head to a scalar made the output depend only on a mean."""
    from src.model.architecture import ValueCNN

    model = ValueCNN().eval()
    x = torch.zeros(2, 18, 8, 8)
    x[1, 0, 0, 0] = 1.0
    x[0, 0, 7, 7] = 1.0                              # same number of pieces, different squares
    with torch.no_grad():
        out = model(x)
    assert not torch.allclose(out[0], out[1])
