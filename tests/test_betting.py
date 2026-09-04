"""TDD suite for the testing-by-betting MIL aggregator.

Written against the contract in `src/utils/betting.py`:

    u_t  = (s_t - mu) / sigma
    e_t  = exp(eta*u_t - eta^2/2)          E_H0[e_t] = 1
    logK = sum_t log(1 + lam*(e_t - 1))    Kelly wealth / test martingale

Run:  cd src && python -m pytest ../tests/test_betting.py -q
"""
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.betting import (  # noqa: E402
    BettingAggregator,
    abnormal_margin,
    betting_mil_loss,
)


def _agg(**kw):
    torch.manual_seed(0)
    return BettingAggregator(**kw)


def _std_agg(**kw):
    """Aggregator whose null is already (0, 1) -- i.e. inputs are pre-standardised."""
    kw.setdefault("init_eta", 1.0)
    kw.setdefault("learn_eta", False)
    kw.setdefault("init_lambda", 0.5)
    kw.setdefault("learn_lambda", False)
    return _agg(**kw)


# --------------------------------------------------------------------------
# shapes / plumbing
# --------------------------------------------------------------------------
def test_output_shape_is_one_scalar_per_video():
    out = _agg()(torch.randn(4, 256), torch.tensor([256, 200, 17, 1]))
    assert out.shape == (4,)


def test_accepts_trailing_singleton_channel():
    """model.py emits logits1 of shape [B, T, 1]; the aggregator must cope."""
    out = _agg()(torch.randn(3, 256, 1), torch.tensor([256, 128, 64]))
    assert out.shape == (3,)


def test_rejects_bad_rank():
    with pytest.raises(ValueError):
        _agg()(torch.randn(2, 8, 5), torch.tensor([8, 8]))


def test_log_wealth_shape():
    lw = _agg().log_wealth(torch.randn(5, 32), torch.tensor([32, 31, 16, 8, 1]))
    assert lw.shape == (5,)


# --------------------------------------------------------------------------
# masking: padded snippets must be inert
# --------------------------------------------------------------------------
def test_padding_is_ignored():
    agg = _agg()
    core = torch.randn(1, 10)
    lengths = torch.tensor([10])
    a = agg.log_wealth(torch.cat([core, torch.zeros(1, 22)], 1), lengths)
    b = agg.log_wealth(torch.cat([core, torch.full((1, 22), 9.0)], 1), lengths)
    c = agg.log_wealth(torch.cat([core, torch.randn(1, 22) * 50], 1), lengths)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(a, c, atol=1e-6)


def test_batch_rows_are_independent():
    agg = _agg()
    s = torch.randn(3, 32)
    lengths = torch.tensor([32, 32, 32])
    joint = agg.log_wealth(s, lengths)
    for i in range(3):
        single = agg.log_wealth(s[i: i + 1], lengths[i: i + 1])
        assert torch.allclose(joint[i], single[0], atol=1e-6)


# --------------------------------------------------------------------------
# e-value semantics (the statistical contract)
# --------------------------------------------------------------------------
def test_e_value_is_fair_under_the_null():
    """E_H0[e] = 1 for standard-normal surprises -- this is what makes the
    wealth process a martingale and Ville's inequality applicable."""
    agg = _std_agg(init_eta=0.8, learn_eta=False)
    torch.manual_seed(7)
    e = agg.e_values(torch.randn(20000, 16))
    assert abs(e.mean().item() - 1.0) < 0.05


def test_e_value_is_one_at_the_typical_surprise():
    """u = eta/2 is the break-even point of the tilt."""
    agg = _std_agg(init_eta=1.0, learn_eta=False)
    e = agg.e_values(torch.full((1, 4), 0.5))
    assert torch.allclose(e, torch.ones_like(e), atol=1e-5)


def test_e_value_is_unbounded_above():
    """Contrast with a probability-ratio e-value, which saturates at 1/p0."""
    agg = _std_agg(init_eta=1.0, learn_eta=False, exp_clamp=30.0)
    assert agg.e_values(torch.tensor([[10.0]])).item() > 1e3


def test_lambda_zero_gives_zero_log_wealth():
    """No bet placed -> capital never changes, whatever the evidence."""
    agg = _agg(init_lambda=0.0, learn_lambda=False)
    lw = agg.log_wealth(torch.randn(2, 64) * 5, torch.tensor([64, 64]))
    assert torch.allclose(lw, torch.zeros(2), atol=1e-6)


def test_null_martingale_expectation_is_at_most_one():
    """Under H0 the capital is a martingale: E[K] <= 1, so E[logK] <= 0."""
    agg = _std_agg(init_eta=0.5, learn_eta=False, init_lambda=0.3, normalize="none")
    torch.manual_seed(1234)
    logK = agg.log_wealth(torch.randn(2000, 64), torch.full((2000,), 64))
    assert logK.mean().item() <= 0.0
    assert torch.exp(logK).mean().item() < 3.0


def test_capital_rises_with_a_single_strong_snippet():
    """One highly-surprising snippet must move the capital on its own.

    (The absolute level is negative: betting at tilt `eta` loses a little on
    every typical outcome -- that *is* the martingale property in log space.
    The learnable bias absorbs that constant offset; only the gap matters.)
    """
    agg = _std_agg(init_eta=1.0, learn_eta=False, init_lambda=0.5, learn_lambda=False)
    T = 256
    quiet = torch.zeros(1, T)
    spike = quiet.clone()
    spike[0, 100] = 8.0
    lengths = torch.tensor([T])
    assert agg.log_wealth(spike, lengths).item() > agg.log_wealth(quiet, lengths).item()


def test_evidence_weighting_beats_uniform_topk_weighting():
    """The point of the wealth process: gradient mass follows the *evidence*.

    top-k spreads the video-level gradient uniformly over its k selected
    snippets, so with k = T/16+1 = 17 a 2-snippet anomaly drags 15 normal
    snippets up with it.  The betting head weights each snippet by its own
    e-value, so the true anomalies dominate.
    """
    T = 256
    truth = slice(100, 102)  # a 2-snippet anomaly, far shorter than k = 17
    base = torch.zeros(1, T)
    base[0, truth] = 6.0
    base[0, 40:70] = 1.0  # mildly noisy normal snippets, the top-k distractors

    # --- betting head -----------------------------------------------------
    agg = _std_agg(init_eta=1.0, learn_eta=False, init_lambda=0.2, learn_lambda=False)
    s = base.clone().requires_grad_(True)
    betting_mil_loss(s, torch.tensor([1.0]), torch.tensor([T]), agg).backward()
    g_bet = s.grad[0].abs()

    # --- classic top-k MIL (VadCLIP's CLAS2) ------------------------------
    s2 = base.clone().requires_grad_(True)
    p = torch.sigmoid(s2)
    tmp, _ = torch.topk(p[0], k=T // 16 + 1)
    F.binary_cross_entropy(tmp.mean().view(1), torch.tensor([1.0])).backward()
    g_topk = s2.grad[0].abs()

    def focus(g):
        anom = g[truth].mean()
        norm = g[40:70].mean()
        return (anom / norm.clamp_min(1e-12)).item()

    assert focus(g_bet) > 3.0 * focus(g_topk)


def test_monotone_in_evidence():
    agg = _std_agg()
    lengths = torch.tensor([32])
    base = torch.zeros(1, 32)
    prev = agg.log_wealth(base, lengths).item()
    for v in [0.5, 1.0, 2.0, 4.0, 8.0]:
        cur = base.clone()
        cur[0, 5] = v
        val = agg.log_wealth(cur, lengths).item()
        assert val > prev
        prev = val


def test_lower_bound_is_log_one_minus_lambda():
    """log(1 + lam(e-1)) >= log(1-lam) since e >= 0."""
    lam = 0.7
    agg = _std_agg(init_lambda=lam, learn_lambda=False, normalize="mean")
    lw = agg.log_wealth(torch.full((1, 64), -50.0), torch.tensor([64]))
    assert lw.item() >= math.log(1 - lam) - 1e-4
    assert math.isfinite(lw.item())


# --------------------------------------------------------------------------
# numerical robustness
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fill", [-1e4, -60.0, 0.0, 60.0, 1e4])
def test_no_nan_for_extreme_scores(fill):
    agg = _agg()
    lw = agg.log_wealth(torch.full((2, 64), float(fill)), torch.tensor([64, 64]))
    assert torch.isfinite(lw).all()


def test_no_nan_when_null_variance_collapses():
    agg = _agg()
    agg.update_null(torch.full((4, 64), 3.0), torch.full((4,), 64))  # zero variance
    out = agg(torch.randn(4, 64), torch.full((4,), 64))
    assert torch.isfinite(out).all()


def test_length_one_video():
    out = _agg()(torch.randn(1, 256), torch.tensor([1]))
    assert torch.isfinite(out).all()


def test_float16_inputs_stay_finite():
    agg = _agg()
    out = agg(torch.randn(2, 64).half(), torch.tensor([64, 64]))
    assert torch.isfinite(out.float()).all()


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------
def test_gradient_reaches_every_valid_snippet():
    """Contrast with top-k, which zeroes the gradient of all but k snippets."""
    agg = _std_agg()
    s = torch.randn(1, 64, requires_grad=True)
    agg.log_wealth(s, torch.tensor([64])).sum().backward()
    assert (s.grad[0].abs() > 0).all()


def test_no_gradient_leaks_into_padding():
    agg = _std_agg()
    s = torch.randn(1, 64, requires_grad=True)
    agg.log_wealth(s, torch.tensor([10])).sum().backward()
    assert torch.allclose(s.grad[0, 10:], torch.zeros(54), atol=1e-9)


def test_eta_and_lambda_are_learnable_and_bounded():
    agg = _agg(learn_eta=True, learn_lambda=True, eta_max=4.0, lambda_max=0.95)
    names = {n for n, p in agg.named_parameters() if p.requires_grad}
    assert {"eta_raw", "lambda_raw", "tau_raw", "bias"} <= names
    agg(torch.randn(4, 64), torch.tensor([64] * 4)).sum().backward()
    assert 0.0 < agg.eta.item() < 4.0
    assert 0.0 < agg.lam.item() < 0.95
    assert agg.eta_raw.grad is not None and agg.lambda_raw.grad is not None


def test_tau_stays_positive():
    agg = _agg(init_tau=20.0)
    assert abs(agg.tau.item() - 20.0) < 1e-3
    with torch.no_grad():
        agg.tau_raw.fill_(-50.0)
    assert agg.tau.item() >= 0.0


def test_null_update_does_not_create_graph():
    agg = _agg()
    s = torch.randn(4, 64, requires_grad=True)
    agg.update_null(s, torch.full((4,), 64))
    assert not agg.mu.requires_grad and not agg.var.requires_grad


# --------------------------------------------------------------------------
# EMA null tracking
# --------------------------------------------------------------------------
def test_first_null_update_initialises_exactly():
    agg = _agg(null_momentum=0.99)
    torch.manual_seed(3)
    s = torch.randn(8, 64) * 2.0 + 5.0
    agg.update_null(s, torch.full((8,), 64))
    assert abs(agg.mu.item() - s.mean().item()) < 1e-4
    assert abs(agg.var.item() - s.var(unbiased=False).item()) < 1e-3


def test_null_tracks_a_shifting_regime():
    agg = _agg(null_momentum=0.5)
    s = torch.full((8, 64), 3.0)
    for _ in range(60):
        agg.update_null(s, torch.full((8,), 64))
    assert abs(agg.mu.item() - 3.0) < 1e-3


def test_null_update_respects_lengths():
    agg = _agg()
    s = torch.cat([torch.full((1, 8), 2.0), torch.full((1, 56), 99.0)], dim=1)
    agg.update_null(s, torch.tensor([8]))
    assert abs(agg.mu.item() - 2.0) < 1e-5


def test_variance_never_degenerates():
    agg = _agg(var_min=1e-3)
    agg.update_null(torch.zeros(4, 64), torch.full((4,), 64))
    assert agg.var.item() >= 1e-3


# --------------------------------------------------------------------------
# alignment-branch margin
# --------------------------------------------------------------------------
def test_abnormal_margin_is_logit_of_the_eval_score():
    """The XD metric uses 1 - softmax(logits2)[..., 0]; the margin is its logit."""
    torch.manual_seed(11)
    logits2 = torch.randn(3, 16, 7)
    p = 1.0 - logits2.softmax(dim=-1)[..., 0]
    assert torch.allclose(torch.sigmoid(abnormal_margin(logits2)), p, atol=1e-5)


def test_abnormal_margin_is_stable_for_large_logits():
    logits2 = torch.full((1, 4, 7), 200.0)
    logits2[..., 0] = -200.0
    assert torch.isfinite(abnormal_margin(logits2)).all()


def test_abnormal_margin_gradient_flows():
    logits2 = torch.randn(2, 8, 7, requires_grad=True)
    abnormal_margin(logits2).sum().backward()
    assert torch.isfinite(logits2.grad).all() and (logits2.grad.abs().sum() > 0)


# --------------------------------------------------------------------------
# loss wrapper
# --------------------------------------------------------------------------
def test_bag_loss_separates_normal_from_anomalous():
    """A detector that spikes on the anomaly must beat a flat one."""
    agg = _std_agg(init_lambda=0.3, learn_lambda=False, init_tau=20.0)
    T = 128
    lengths = torch.tensor([T, T])
    labels = torch.tensor([0.0, 1.0])

    good = torch.zeros(2, T)
    good[1, 40:50] = 4.0
    flat = torch.zeros(2, T)

    assert betting_mil_loss(good, labels, lengths, agg).item() < betting_mil_loss(
        flat, labels, lengths, agg
    ).item()


def test_bag_loss_punishes_false_positives_in_normal_videos():
    agg = _std_agg(init_lambda=0.3, learn_lambda=False, init_tau=20.0)
    T = 128
    lengths = torch.tensor([T])
    labels = torch.tensor([0.0])

    clean = torch.zeros(1, T)
    spiky = torch.zeros(1, T)
    spiky[0, 7] = 5.0
    assert betting_mil_loss(spiky, labels, lengths, agg).item() > betting_mil_loss(
        clean, labels, lengths, agg
    ).item()


def test_bag_loss_is_scalar_and_finite():
    loss = betting_mil_loss(
        torch.randn(6, 256, 1), torch.tensor([0.0, 1, 0, 1, 1, 0]), torch.full((6,), 256), _agg()
    )
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_bag_loss_backprops_through_alignment_branch():
    agg = _agg()
    logits2 = torch.randn(3, 64, 7, requires_grad=True)
    loss = betting_mil_loss(
        abnormal_margin(logits2), torch.tensor([0.0, 1, 1]), torch.tensor([64, 40, 12]), agg
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits2.grad).all()


# --------------------------------------------------------------------------
# baseline recoverability
# --------------------------------------------------------------------------
def test_zero_weight_recovers_baseline_exactly():
    """With weight 0 the module must not perturb the baseline gradient at all."""
    agg = _agg()
    s = torch.randn(4, 64, requires_grad=True)
    base = (s ** 2).sum()
    total = base + 0.0 * betting_mil_loss(s, torch.tensor([0.0, 1, 0, 1]), torch.full((4,), 64), agg)
    total.backward()
    assert torch.allclose(s.grad, 2 * s.detach(), atol=1e-6)
