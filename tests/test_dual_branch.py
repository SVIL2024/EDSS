"""TDD suite for the dual-branch betting head wired into the training loops.

Contract: given the two branch logits and the multi-hot text labels VadCLIP
already builds, produce the auxiliary betting loss and keep the per-branch null
regimes up to date -- using *only* the normal videos for the null.

Run:  cd src && python -m pytest ../tests/test_dual_branch.py -q
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.betting import DualBranchBetting, abnormal_margin  # noqa: E402


def _labels(flags, n_class=7):
    """flags[i] == 0 -> normal video (index 0 hot), else an anomaly class."""
    y = torch.zeros(len(flags), n_class)
    for i, f in enumerate(flags):
        y[i, 0 if f == 0 else f] = 1.0
    return y


def _batch(B=6, T=64, C=7, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(B, T, 1, requires_grad=True),
        torch.randn(B, T, C, requires_grad=True),
        _labels([0, 0, 0, 1, 2, 3], C),
        torch.full((B,), T),
    )


# ------------------------------------------------------------------ plumbing
def test_disabled_returns_exact_zero_and_no_graph():
    head = DualBranchBetting(weight1=0.0, weight2=0.0)
    l1, l2, y, n = _batch()
    loss, stats = head(l1, l2, y, n)
    assert float(loss) == 0.0
    assert not loss.requires_grad
    assert stats == {}


def test_enabled_returns_scalar_and_stats():
    head = DualBranchBetting(weight1=1.0, weight2=0.5)
    l1, l2, y, n = _batch()
    loss, stats = head(l1, l2, y, n)
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert set(stats) == {"bet1", "bet2"}


def test_weights_scale_the_loss_linearly():
    l1, l2, y, n = _batch()
    a = DualBranchBetting(weight1=1.0, weight2=0.0)
    b = DualBranchBetting(weight1=2.0, weight2=0.0)
    b.agg1.load_state_dict(a.agg1.state_dict())
    la, _ = a(l1, l2, y, n)
    lb, _ = b(l1, l2, y, n)
    assert torch.allclose(lb, 2 * la, atol=1e-5)


def test_only_the_requested_branch_receives_gradient():
    head = DualBranchBetting(weight1=1.0, weight2=0.0)
    l1, l2, y, n = _batch()
    head(l1, l2, y, n)[0].backward()
    assert l1.grad.abs().sum() > 0
    assert l2.grad is None or l2.grad.abs().sum() == 0


def test_branch_two_gradient_reaches_alignment_logits():
    head = DualBranchBetting(weight1=0.0, weight2=1.0)
    l1, l2, y, n = _batch()
    head(l1, l2, y, n)[0].backward()
    assert l2.grad.abs().sum() > 0
    assert l1.grad is None or l1.grad.abs().sum() == 0


# ----------------------------------------------------------- null estimation
def test_null_uses_only_normal_videos():
    """The null regime must be estimated from label-0 videos alone."""
    head = DualBranchBetting(weight1=1.0, weight2=0.0, null_momentum=0.0)
    T, C = 32, 7
    l1 = torch.zeros(4, T, 1)
    l1[0] = 1.0   # normal
    l1[1] = 1.0   # normal
    l1[2] = 90.0  # anomalous - must NOT contaminate the null
    l1[3] = 90.0
    head(l1, torch.randn(4, T, C), _labels([0, 0, 1, 2], C), torch.full((4,), T))
    assert abs(head.agg1.mu.item() - 1.0) < 1e-4


def test_all_anomalous_batch_leaves_null_untouched():
    head = DualBranchBetting(weight1=1.0, weight2=0.0)
    T, C = 32, 7
    head.agg1.update_null(torch.full((2, T), 5.0), torch.full((2,), T))
    before = head.agg1.mu.clone()
    head(torch.randn(3, T, 1), torch.randn(3, T, C), _labels([1, 2, 3], C), torch.full((3,), T))
    assert torch.allclose(head.agg1.mu, before)


def test_null_update_is_detached_from_the_graph():
    head = DualBranchBetting(weight1=1.0, weight2=1.0)
    l1, l2, y, n = _batch()
    head(l1, l2, y, n)[0].backward()  # must not raise "backward through the graph twice"
    assert not head.agg1.mu.requires_grad and not head.agg2.mu.requires_grad


# --------------------------------------------------------------- correctness
def test_branch_two_uses_the_evaluated_score():
    """Branch 2 must bet on `1 - softmax(logits2)[..., 0]`, the XD metric."""
    head = DualBranchBetting(weight1=0.0, weight2=1.0)
    l1, l2, y, n = _batch()
    loss, _ = head(l1, l2, y, n)
    from utils.betting import betting_mil_loss

    expected = betting_mil_loss(abnormal_margin(l2), 1 - y[:, 0], n, head.agg2)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_binary_target_is_derived_from_the_normal_column():
    head = DualBranchBetting(weight1=1.0, weight2=0.0, null_momentum=0.0)
    T, C = 32, 7
    y = _labels([0, 1, 0, 2], C)
    # a detector that fires on exactly the anomalous videos should beat one
    # that fires on the normal ones
    good = torch.zeros(4, T, 1)
    good[1] = 6.0
    good[3] = 6.0
    bad = torch.zeros(4, T, 1)
    bad[0] = 6.0
    bad[2] = 6.0
    l2 = torch.randn(4, T, C)
    n = torch.full((4,), T)
    lo, _ = DualBranchBetting(weight1=1.0, weight2=0.0, null_momentum=0.0)(good, l2, y, n)
    hi, _ = DualBranchBetting(weight1=1.0, weight2=0.0, null_momentum=0.0)(bad, l2, y, n)
    assert lo.item() < hi.item()


def test_multi_hot_xd_labels_are_handled():
    """XD videos can carry several anomaly classes at once."""
    head = DualBranchBetting(weight1=1.0, weight2=1.0)
    T, C = 32, 7
    y = torch.zeros(3, C)
    y[0, 0] = 1
    y[1, 1] = y[1, 3] = 1
    y[2, 2] = y[2, 5] = y[2, 6] = 1
    loss, _ = head(torch.randn(3, T, 1), torch.randn(3, T, C), y, torch.full((3,), T))
    assert torch.isfinite(loss)


def test_variable_lengths_are_respected():
    head = DualBranchBetting(weight1=1.0, weight2=1.0)
    T, C = 64, 7
    l1 = torch.randn(3, T, 1)
    l2 = torch.randn(3, T, C)
    y = _labels([0, 1, 2], C)
    lengths = torch.tensor([64, 30, 3])
    a, _ = head(l1, l2, y, lengths)

    l1b, l2b = l1.clone(), l2.clone()
    l1b[1, 30:] = 99.0
    l1b[2, 3:] = -99.0
    l2b[1, 30:] = 99.0
    l2b[2, 3:] = -99.0
    b, _ = head(l1b, l2b, y, lengths)
    assert torch.allclose(a, b, atol=1e-5)


def test_parameters_are_exposed_for_the_optimizer():
    head = DualBranchBetting(weight1=1.0, weight2=1.0)
    names = {n for n, _ in head.named_parameters()}
    assert any(n.startswith("agg1.") for n in names)
    assert any(n.startswith("agg2.") for n in names)
    assert len(list(head.parameters())) == 8  # 4 scalars per branch


@pytest.mark.parametrize("device", ["cpu"])
def test_runs_on_device(device):
    head = DualBranchBetting(weight1=1.0, weight2=1.0).to(device)
    T, C = 32, 7
    loss, _ = head(
        torch.randn(4, T, 1, device=device),
        torch.randn(4, T, C, device=device),
        _labels([0, 1, 0, 2], C).to(device),
        torch.full((4,), T, device=device),
    )
    assert loss.device.type == device
