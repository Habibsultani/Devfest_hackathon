import numpy as np
import pytest
import torch
from PIL import Image

from src.dinomaly_model import LoRALinear, StableLinearAttention
from src.tiling import extract_tiles, instrument_score, stitch_tile_maps, tile_tensor


def test_native_tile_stitching_preserves_edges_and_mask():
    image = Image.new("RGB", (300, 280), "white")
    mask = np.ones((280, 300), dtype=bool)
    mask[:20, :20] = False
    tiles = extract_tiles(image, mask)
    assert len(tiles) == 4
    assert tile_tensor(tiles[-1][1]).shape == (3, 252, 252)
    maps = [np.full((252, 252), 2, dtype=np.float32) for _ in tiles]
    reconstructed = stitch_tile_maps(tiles, maps, mask.shape, mask)
    np.testing.assert_allclose(reconstructed[mask], 2, atol=1e-5)
    assert not reconstructed[~mask].any()


def test_score_uses_instrument_only_and_not_background():
    values = np.ones((100, 100), dtype=np.float32)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    values[~mask] = 100
    assert instrument_score(values, mask) == 1
    with pytest.raises(ValueError):
        instrument_score(values, np.zeros_like(mask))


def test_lora_starts_identical_but_adapters_receive_gradients():
    base = torch.nn.Linear(8, 16)
    module = LoRALinear(base, rank=2)
    inputs = torch.randn(3, 8)
    torch.testing.assert_close(module(inputs), base(inputs))
    module(inputs).sum().backward()
    assert module.b.grad.abs().sum() > 0
    assert module.base.weight.grad is None


def test_stable_linear_attention_produces_finite_gradients():
    attention = StableLinearAttention(32, num_heads=4)
    inputs = (torch.randn(2, 36, 32) * 100).requires_grad_()
    output, _ = attention(inputs)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(inputs.grad).all()
