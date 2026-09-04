"""TDD suite for null-alignment: pull the non-selected bulk of a video toward
the normal-video null.

Motivation (docs/LIMITATIONS.md): the selector's ideal FDR contract
fails because the *true* null -- normal snippets inside anomalous videos -- sits
**+2.00 sigma** above the assumed null (snippets of fully-normal videos).
Weakly-supervised MIL lifts whole anomalous videos, so the e-values of genuinely
normal snippets are inflated ~7.6x and e-BH over-rejects by construction.

This loss enforces the null contract *during training*: the snippets e-BH does
**not** select must behave like the normal regime -- their standardised scores
``u = (s - mu)/sigma`` under the EMA normal-video null should be standard
normal.  Selection and null enforcement are computed from the same e-values, so
the two co-calibrate: select the true anomalies, and everything left over must
look null.

Run:  cd src && python -m pytest ../tests/test_null_alignment.py -q
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.ebh import ebh_reject, null_alignment_loss  # noqa: E402

NEG = -1e30


# ------------------------------------------------------------------- helpers
def _log(e):
    return torch.log(torch.as_tensor(e, dtype=torch.float32))


# ------------------------------------------------------------ exact semantics
def test_zero_for_exactly_standard_normal_null_set():
    """u = [-1,-1,1,1] has mean 0 and variance 1 -> both terms vanish."""
    s = torch.tensor([[-1.0, -1.0, 1.0, 1.0]])
    loss, stats = null_alignment_loss(
        s, torch.zeros(1, 4), torch.tensor([4]), 0.0, 1.0
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_positive_when_null_set_is_shifted_by_two_sigma():
    """The +2.00 sigma diagnosis: u = [2,2,2,2] must cost exactly loc=2.0."""
    s = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    loss, _ = null_alignment_loss(
        s, torch.zeros(1, 4), torch.tensor([4]), 0.0, 1.0
    )
    # loc = mean(u)^2 / 2 = 2.0 ; scale = (var-1)^2/2 = 0.5
    assert loss.item() == pytest.approx(2.5, abs=1e-6)


def test_scale_term_catches_variance_mismatch():
    """u = [-2,-1,1,2]: mean 0, var 2.5 -> scale = (2.5-1)^2/2 = 1.125."""
    s = torch.tensor([[-2.0, -1.0, 1.0, 2.0]])
    loss, _ = null_alignment_loss(
        s, torch.zeros(1, 4), torch.tensor([4]), 0.0, 1.0
    )
    assert loss.item() == pytest.approx(1.125, abs=1e-6)


def test_both_terms_combine():
    """u = [0,0,2,2]: mean 1, var 1 -> loc 0.5, scale 0."""
    s = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    loss, _ = null_alignment_loss(
        s, torch.zeros(1, 4), torch.tensor([4]), 0.0, 1.0
    )
    assert loss.item() == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------- selection
def test_selected_snippets_are_excluded():
    """sel=True on position 0 (-1) -> null is [1,1,1] -> loss 1.0, not 0."""
    s = torch.tensor([[-1.0, 1.0, 1.0, 1.0]])
    sel = torch.tensor([[True, False, False, False]])
    loss, _ = null_alignment_loss(
        s, torch.zeros(1, 4), torch.tensor([4]), 0.0, 1.0, sel=sel
    )
    assert loss.item() == pytest.approx(1.0, abs=1e-6)


def test_sel_none_means_all_valid_snippets_are_null():
    s = torch.tensor([[-1.0, -1.0, 1.0, 1.0]])
    a, _ = null_alignment_loss(s, torch.zeros(1, 4), torch.tensor([4]), 0.0, 1.0)
    b, _ = null_alignment_loss(
        s, torch.zeros(1, 4), torch.tensor([4]), 0.0, 1.0,
        sel=torch.zeros(1, 4, dtype=torch.bool),
    )
    assert a.item() == b.item()


def test_composes_with_ebh_reject_inside_anomalous_video():
    """The real wiring: sel = ebh_reject(log_e); loss on the complement."""
    torch.manual_seed(0)
    s = torch.randn(2, 32)
    log_e = _log(torch.exp(torch.randn(2, 32) * 1.5 + 1.0))
    lengths = torch.tensor([32, 16])
    sel = ebh_reject(log_e, lengths, alpha=0.3, min_reject=1, max_frac=0.25)
    loss, stats = null_alignment_loss(
        s, log_e, lengths, 0.0, 1.0, sel=sel
    )
    assert loss.item() == loss.item()  # finite
    assert "u_mean" in stats and "u_std" in stats
    assert sel.sum(1)[0].item() >= 1  # min_reject honoured


# ----------------------------------------------------------- masking / shape
def test_padding_never_contributes():
    """T=5, length=3: padding scores of 1e9 must not touch the loss."""
    s = torch.tensor([[-1.0, -1.0, 1.0, 1e9, 1e9]])
    loss, _ = null_alignment_loss(
        s, torch.zeros(1, 5), torch.tensor([3]), 0.0, 1.0
    )
    # valid u = [-1,-1,1] -> mean -1/3, var 8/9
    #   loc = (1/3)^2/2 = 1/18 ; scale = (8/9-1)^2/2 = 1/162
    assert loss.item() == pytest.approx(10.0 / 162.0, abs=1e-6)


def test_accepts_trailing_singleton_channel():
    s = torch.randn(2, 16, 1)
    loss, _ = null_alignment_loss(
        s, torch.zeros(2, 16), torch.tensor([16, 8]), 0.0, 1.0
    )
    assert torch.isfinite(loss)


def test_single_snippet_video_no_nan():
    s = torch.tensor([[5.0, 1e9, 1e9, 1e9]])
    loss, _ = null_alignment_loss(
        s, torch.zeros(1, 4), torch.tensor([1]), 0.0, 1.0
    )
    # u=[5]: loc 12.5, scale (0-1)^2/2 = 0.5
    assert loss.item() == pytest.approx(13.0, abs=1e-5)


# --------------------------------------------------------------- gradients
def test_gradients_flow_to_scores_not_null():
    """mu/sigma are the detached EMA null: scores get grad, null gets none."""
    s = torch.tensor([[2.0, 2.0, 2.0, 2.0]], requires_grad=True)
    mu = torch.tensor(0.0)  # a plain tensor: never requires grad
    sigma = torch.tensor(1.0)
    loss, _ = null_alignment_loss(s, torch.zeros(1, 4), torch.tensor([4]), mu, sigma)
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert mu.grad is None and sigma.grad is None


def test_gradients_do_not_flow_through_selection():
    """log_e chooses the null set; its values must not receive gradient."""
    s = torch.tensor([[-1.0, -1.0, 1.0, 1.0]], requires_grad=True)
    log_e = torch.zeros(1, 4, requires_grad=True)
    sel = torch.tensor([[True, False, False, False]])
    loss, _ = null_alignment_loss(
        s, log_e, torch.tensor([4]), 0.0, 1.0, sel=sel
    )
    loss.backward()
    assert log_e.grad is None


def test_empty_null_set_returns_zero():
    """All valid snippets selected -> nothing to align -> 0, no NaN."""
    s = torch.tensor([[2.0, 2.0, 2.0, 2.0]], requires_grad=True)
    sel = torch.tensor([[True, True, True, True]])
    loss, stats = null_alignment_loss(
        s, torch.zeros(1, 4), torch.tensor([4]), 0.0, 1.0, sel=sel
    )
    assert loss.item() == 0.0
    loss.backward()
    assert s.grad is not None  # zero tensor still attached to the graph


# ------------------------------------------------------------- integration
def test_composes_with_ebh_pseudo_loss_in_one_graph():
    """Both losses in one graph: finite, backward runs, no NaN grads."""
    torch.manual_seed(0)
    logits = torch.randn(2, 32, requires_grad=True)
    log_e = torch.randn(2, 32).detach()
    lengths = torch.tensor([32, 16])
    labels = torch.tensor([1.0, 0.0])
    from utils.ebh import ebh_pseudo_loss

    sel = ebh_reject(log_e, lengths, alpha=0.3, min_reject=1, max_frac=0.25)
    l1 = ebh_pseudo_loss(
        logits, log_e, lengths, labels, alpha=0.3, min_reject=1,
        normal_weight=1.0, max_frac=0.25, neg_frac=0.1,
    )
    l2, stats = null_alignment_loss(
        logits, log_e, lengths, 0.0, 1.0, sel=sel
    )
    total = l1 + 0.1 * l2
    total.backward()
    assert torch.isfinite(total).item()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
