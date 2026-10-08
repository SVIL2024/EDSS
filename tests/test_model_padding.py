"""Padded snippets cannot change valid detector responses or gradients."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import model as model_module
from utils.tools import get_batch_mask


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(49408, 8)

    def encode_token(self, tokens):
        return self.token_embedding(tokens.long())

    def encode_text(self, embeddings, tokens):
        return embeddings.sum(dim=1)


@pytest.fixture
def detector(monkeypatch):
    torch.manual_seed(234)
    monkeypatch.setattr(model_module.clip, "load", lambda *args: (TinyCLIP(), None))

    def cpu_distance(self, batch_size, length):
        positions = torch.arange(length, dtype=torch.float32)
        distance = (positions[:, None] - positions[None, :]).abs()
        return torch.exp(-distance / torch.exp(torch.tensor(1.0))).expand(
            batch_size, length, length)

    monkeypatch.setattr(model_module.DistanceAdj, "forward", cpu_distance)
    return model_module.CLIPVAD(2, 8, 8, 8, 1, 1, 8, 1, 1, "cpu").eval()


def test_padding_cannot_change_valid_features_or_either_branch(detector):
    lengths = torch.tensor([5, 8])
    padding = get_batch_mask(lengths, 8)
    visual = torch.randn(2, 8, 8)
    changed = visual.clone()
    changed[padding] = 10000

    original = detector(visual, padding, ["normal", "anomaly"], lengths)
    perturbed = detector(changed, None, ["normal", "anomaly"], lengths)
    for before, after in zip(original[1:], perturbed[1:]):
        assert torch.isfinite(before).all()
        torch.testing.assert_close(before[~padding], after[~padding])
        assert torch.count_nonzero(before[padding]) == 0
        assert torch.count_nonzero(after[padding]) == 0


def test_padding_has_no_gradient(detector):
    lengths = torch.tensor([5, 8])
    padding = get_batch_mask(lengths, 8)
    visual = torch.randn(2, 8, 8, requires_grad=True)
    _, c, a = detector(visual, padding, ["normal", "anomaly"], lengths)
    (c[~padding].square().sum() + a[~padding].square().sum()).backward()
    assert torch.isfinite(visual.grad).all()
    assert torch.count_nonzero(visual.grad[padding]) == 0
    assert torch.count_nonzero(visual.grad[~padding]) > 0


def test_invalid_padding_shape_is_rejected(detector):
    with pytest.raises(ValueError, match="padding_mask must have shape"):
        detector.encode_video(torch.randn(2, 8, 8), torch.zeros(2, 7), [5, 8])
