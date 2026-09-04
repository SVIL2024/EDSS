"""TDD suite for the auxiliary temporal regularisers (`src/utils/regularizers.py`).

These are the classic Sultani-style smoothness / sparsity priors, absent from
VadCLIP.  They are *not* the novelty -- they are cheap, well-understood
supporting terms whose weights default to 0 so the baseline is recoverable.

Run:  cd src && python -m pytest ../tests/test_regularizers.py -q
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.regularizers import smoothness_loss, sparsity_loss  # noqa: E402


# ---------------------------------------------------------------- smoothness
def test_smoothness_is_zero_for_constant_scores():
    s = torch.full((3, 64), 0.4)
    assert smoothness_loss(s, torch.tensor([64, 64, 64])).item() < 1e-9


def test_smoothness_penalises_jitter():
    lengths = torch.tensor([64])
    smooth = torch.linspace(0, 1, 64).unsqueeze(0)
    jitter = torch.tensor([0.0, 1.0] * 32).unsqueeze(0)
    assert smoothness_loss(jitter, lengths).item() > smoothness_loss(smooth, lengths).item()


def test_smoothness_ignores_padding():
    core = torch.rand(1, 10)
    lengths = torch.tensor([10])
    a = smoothness_loss(torch.cat([core, torch.zeros(1, 54)], 1), lengths)
    b = smoothness_loss(torch.cat([core, torch.rand(1, 54)], 1), lengths)
    assert torch.allclose(a, b, atol=1e-6)


def test_smoothness_handles_length_one():
    assert torch.isfinite(smoothness_loss(torch.rand(1, 64), torch.tensor([1])))


def test_smoothness_accepts_trailing_channel():
    out = smoothness_loss(torch.rand(2, 32, 1), torch.tensor([32, 16]))
    assert out.dim() == 0


# ------------------------------------------------------------------ sparsity
def test_sparsity_is_zero_for_all_zero_scores():
    assert sparsity_loss(torch.zeros(2, 64), torch.tensor([64, 64])).item() < 1e-9


def test_sparsity_grows_with_mean_score():
    lengths = torch.tensor([64])
    low = sparsity_loss(torch.full((1, 64), 0.1), lengths).item()
    high = sparsity_loss(torch.full((1, 64), 0.8), lengths).item()
    assert high > low


def test_sparsity_ignores_padding():
    core = torch.rand(1, 10)
    lengths = torch.tensor([10])
    a = sparsity_loss(torch.cat([core, torch.zeros(1, 54)], 1), lengths)
    b = sparsity_loss(torch.cat([core, torch.ones(1, 54)], 1), lengths)
    assert torch.allclose(a, b, atol=1e-6)


# ----------------------------------------------------------------- gradients
def test_both_losses_are_differentiable():
    s = torch.rand(2, 32, requires_grad=True)
    (smoothness_loss(s, torch.tensor([32, 32])) + sparsity_loss(s, torch.tensor([32, 32]))).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
