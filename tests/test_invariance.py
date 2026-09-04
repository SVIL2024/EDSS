"""TDD suite for the affine-invariance fix (run `ucf_bet1w1` post-mortem).

The first betting run standardised with a *detached* EMA null, so the loss
could be reduced by translating every score downwards.  The C-branch logits
drifted to mean -41, `sigmoid(logits1)` collapsed to 0, `CLAS2` exploded and
the AUC fell to chance.

Contract now: when the null location/scale are estimated from the batch's own
normal videos *with gradient*, the wealth statistic depends only on the SHAPE
of the score distribution -- both degenerate directions have zero gradient.

Run:  cd src && python -m pytest ../tests/test_invariance.py -q
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.betting import BettingAggregator, DualBranchBetting  # noqa: E402


def _agg(**kw):
    torch.manual_seed(0)
    kw.setdefault("learn_eta", False)
    kw.setdefault("learn_lambda", False)
    kw.setdefault("init_eta", 1.0)
    kw.setdefault("init_lambda", 0.3)
    return BettingAggregator(**kw)


def _batch_stats(agg, s, lengths):
    return agg.batch_null(s, lengths)


def _labels(flags, n_class=7):
    y = torch.zeros(len(flags), n_class)
    for i, f in enumerate(flags):
        y[i, 0 if f == 0 else f] = 1.0
    return y


# ---------------------------------------------------------------- invariance
def test_wealth_is_invariant_to_a_global_shift():
    agg = _agg()
    torch.manual_seed(1)
    s = torch.randn(4, 32)
    lengths = torch.full((4,), 32)
    c, sc = _batch_stats(agg, s, lengths)
    a = agg.log_wealth(s, lengths, center=c, scale=sc)

    shifted = s + 7.5
    c2, sc2 = _batch_stats(agg, shifted, lengths)
    b = agg.log_wealth(shifted, lengths, center=c2, scale=sc2)
    assert torch.allclose(a, b, atol=1e-4)


def test_wealth_is_invariant_to_a_global_rescale():
    agg = _agg()
    torch.manual_seed(2)
    s = torch.randn(4, 32) * 3.0
    lengths = torch.full((4,), 32)
    c, sc = _batch_stats(agg, s, lengths)
    a = agg.log_wealth(s, lengths, center=c, scale=sc)

    scaled = s * 4.0
    c2, sc2 = _batch_stats(agg, scaled, lengths)
    b = agg.log_wealth(scaled, lengths, center=c2, scale=sc2)
    assert torch.allclose(a, b, atol=1e-4)


def test_shift_direction_has_zero_gradient():
    """The exact direction that broke run `ucf_bet1w1` must now be flat."""
    agg = _agg()
    torch.manual_seed(3)
    s = torch.randn(4, 32)
    lengths = torch.full((4,), 32)
    shift = torch.zeros(1, requires_grad=True)
    c, sc = _batch_stats(agg, s + shift, lengths)
    agg.log_wealth(s + shift, lengths, center=c, scale=sc).sum().backward()
    assert shift.grad.abs().item() < 1e-4


def test_scale_direction_has_zero_gradient():
    agg = _agg()
    torch.manual_seed(4)
    s = torch.randn(4, 32)
    lengths = torch.full((4,), 32)
    log_a = torch.zeros(1, requires_grad=True)
    scaled = s * torch.exp(log_a)
    c, sc = _batch_stats(agg, scaled, lengths)
    agg.log_wealth(scaled, lengths, center=c, scale=sc).sum().backward()
    assert log_a.grad.abs().item() < 1e-4


def test_relative_separation_still_produces_gradient():
    """Invariance must not make the loss blind -- moving one video's scores
    away from the normal population must still change the capital."""
    agg = _agg()
    torch.manual_seed(5)
    s = torch.randn(4, 32)
    lengths = torch.full((4,), 32)
    c, sc = _batch_stats(agg, s[:2], lengths[:2])  # null from the first two only
    base = agg.log_wealth(s, lengths, center=c, scale=sc)

    s2 = s.clone()
    s2[3, 10:14] += 6.0
    c2, sc2 = _batch_stats(agg, s2[:2], lengths[:2])
    bumped = agg.log_wealth(s2, lengths, center=c2, scale=sc2)
    assert bumped[3].item() > base[3].item() + 0.05


# ------------------------------------------------------------- batch_null API
def test_batch_null_matches_masked_moments():
    agg = _agg()
    torch.manual_seed(6)
    s = torch.randn(2, 16)
    lengths = torch.tensor([16, 6])
    c, sc = _batch_stats(agg, s, lengths)
    flat = torch.cat([s[0, :16], s[1, :6]])
    assert abs(c.item() - flat.mean().item()) < 1e-5
    assert abs(sc.item() - (flat.var(unbiased=False) + agg.eps).sqrt().item()) < 1e-3


def test_batch_null_is_differentiable():
    agg = _agg()
    s = torch.randn(2, 16, requires_grad=True)
    c, sc = _batch_stats(agg, s, torch.tensor([16, 16]))
    (c + sc).backward()
    assert s.grad is not None and s.grad.abs().sum() > 0


def test_batch_null_floors_the_variance():
    agg = _agg(var_min=1e-2)
    _, sc = _batch_stats(agg, torch.full((2, 16), 4.0), torch.tensor([16, 16]))
    assert sc.item() >= 1e-1 - 1e-6


def test_ema_is_used_when_center_is_not_supplied():
    agg = _agg()
    agg.update_null(torch.full((2, 16), 5.0) + torch.randn(2, 16), torch.tensor([16, 16]))
    s = torch.randn(2, 16)
    lengths = torch.tensor([16, 16])
    manual = agg.log_wealth(s, lengths, center=agg.mu, scale=agg.var.sqrt())
    default = agg.log_wealth(s, lengths)
    assert torch.allclose(manual, default, atol=1e-4)


# ------------------------------------------------------- dual-branch wiring
def test_dual_branch_is_shift_invariant_end_to_end():
    head = DualBranchBetting(weight1=1.0, weight2=0.0)
    torch.manual_seed(7)
    T, C = 32, 7
    l1 = torch.randn(4, T, 1)
    l2 = torch.randn(4, T, C)
    y = _labels([0, 0, 1, 2], C)
    n = torch.full((4,), T)
    a, _ = head(l1, l2, y, n)
    b, _ = head(l1 + 12.0, l2, y, n)
    assert torch.allclose(a, b, atol=1e-4)


def test_dual_branch_shift_gradient_is_zero():
    head = DualBranchBetting(weight1=1.0, weight2=1.0)
    torch.manual_seed(8)
    T, C = 32, 7
    shift = torch.zeros(1, requires_grad=True)
    head(
        torch.randn(4, T, 1) + shift,
        torch.randn(4, T, C),
        _labels([0, 0, 1, 2], C),
        torch.full((4,), T),
    )[0].backward()
    assert shift.grad.abs().item() < 1e-4


def test_dual_branch_falls_back_to_ema_without_normal_videos():
    head = DualBranchBetting(weight1=1.0, weight2=0.0)
    torch.manual_seed(9)
    T, C = 32, 7
    head.agg1.update_null(torch.randn(2, T), torch.full((2,), T))
    loss, _ = head(
        torch.randn(3, T, 1), torch.randn(3, T, C), _labels([1, 2, 3], C), torch.full((3,), T)
    )
    assert torch.isfinite(loss)


def test_dual_branch_still_separates_after_the_fix():
    """The whole point: a detector that spikes on anomalies beats a flat one,
    even though absolute level no longer matters."""
    T, C = 64, 7
    y = _labels([0, 0, 1, 2], C)
    n = torch.full((4,), T)
    l2 = torch.randn(4, T, C)

    good = torch.randn(4, T, 1) * 0.1
    good[2, 20:26] += 5.0
    good[3, 40:46] += 5.0
    flat = torch.randn(4, T, 1) * 0.1

    lo, _ = DualBranchBetting(weight1=1.0, weight2=0.0)(good, l2, y, n)
    hi, _ = DualBranchBetting(weight1=1.0, weight2=0.0)(flat, l2, y, n)
    assert lo.item() < hi.item()


def test_moving_only_the_normal_population_is_meaningful():
    """Only a *global* translation is inert.  Pushing the normal population down
    while the anomalous videos stay put genuinely raises their surprise, so the
    capital of the anomalous videos must go up.

    (Probed on the wealth statistic rather than the loss: with a well-separated
    batch the BCE saturates and hides the effect.)
    """
    head = DualBranchBetting(weight1=1.0, weight2=0.0)
    torch.manual_seed(10)
    T, C = 32, 7
    l1 = torch.randn(4, T, 1) * 0.5
    l1[2:] += 1.0  # anomalous videos mildly separated
    y = _labels([0, 0, 1, 2], C)
    n = torch.full((4,), T)

    def anomalous_capital(x):
        s = x.squeeze(-1)
        c, sc = head.agg1.batch_null(s[:2], n[:2])
        return head.agg1.log_wealth(s, n, center=c, scale=sc)[2:].mean().item()

    moved = l1.clone()
    moved[:2] -= 3.0
    assert anomalous_capital(moved) > anomalous_capital(l1) + 1e-3
