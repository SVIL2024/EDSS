"""TDD suite for e-BH-inspired snippet selection inside a video.

VadCLIP supervises a fixed `k = T/16 + 1` snippets per video. That is an
approximately 6.25% selection budget regardless of the unknown event duration;
when an event is shorter, normal snippets can receive upward pressure.

The mathematical e-Benjamini-Hochberg procedure (Wang & Ramdas, JRSS-B 2022)
picks the number of rejections adaptively and controls FDR under arbitrary
dependence when its inputs are valid e-values.  The learned, estimated-null
inputs in this project do not satisfy that calibration empirically, so these
tests verify the procedure and selector mechanics, not deployed FDR control.

Procedure on e-values e_1..e_T:
    sort descending -> e_(1) >= ... >= e_(T)
    k* = max { k : e_(k) >= T / (alpha * k) }        (0 if the set is empty)
    reject the k* largest
For valid e-values, FDR of the rejected set is <= alpha.

Run:  cd src && python -m pytest ../tests/test_ebh.py -q
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.ebh import (  # noqa: E402
    ebh_pseudo_loss,
    ebh_reject,
    fixed_k_mask,
    original_topk_mask,
    pseudo_label_loss,
    soft_evidence_weights,
    strict_ebh_reject,
)

NEG = -1e30


def _log(e):
    return torch.log(torch.as_tensor(e, dtype=torch.float32))


# --------------------------------------------------------------------- shape
def test_returns_bool_mask_of_input_shape():
    m = ebh_reject(torch.randn(3, 32), torch.tensor([32, 16, 4]), alpha=0.5)
    assert m.shape == (3, 32) and m.dtype == torch.bool


def test_accepts_trailing_singleton_channel():
    m = ebh_reject(torch.randn(2, 16, 1), torch.tensor([16, 8]), alpha=0.5)
    assert m.shape == (2, 16)


# ----------------------------------------------------------- exact procedure
def test_hand_computed_example():
    """T=4, alpha=1 -> threshold for rank k is 4/k = [4, 2, 1.33, 1].

    e = [10, 3, 0.5, 0.1]:  rank1 10>=4 ok, rank2 3>=2 ok, rank3 0.5>=1.33 no,
    rank4 0.1>=1 no  ->  k* = 2 (the largest k that passes).
    """
    e = torch.tensor([[10.0, 3.0, 0.5, 0.1]])
    m = ebh_reject(_log(e), torch.tensor([4]), alpha=1.0, min_reject=0)
    assert m.tolist() == [[True, True, False, False]]


def test_kstar_is_the_largest_passing_rank_not_the_first_failure():
    """e-BH takes the MAX passing k, so a gap in the middle must not stop it.

    T=4, alpha=1, thresholds [4, 2, 1.333, 1];  e = [10, 1.5, 1.4, 1.2]
    rank1 ok, rank2 1.5>=2 fails, rank3 1.4>=1.333 ok, rank4 1.2>=1 ok -> k*=4.
    """
    e = torch.tensor([[10.0, 1.5, 1.4, 1.2]])
    m = ebh_reject(_log(e), torch.tensor([4]), alpha=1.0, min_reject=0)
    assert m.sum().item() == 4


def test_no_rejection_when_evidence_is_weak():
    e = torch.full((1, 64), 0.5)
    m = ebh_reject(_log(e), torch.tensor([64]), alpha=0.05, min_reject=0)
    assert m.sum().item() == 0


def test_selects_the_largest_e_values():
    torch.manual_seed(0)
    log_e = torch.randn(2, 32) * 3
    m = ebh_reject(log_e, torch.tensor([32, 32]), alpha=1.0, min_reject=3)
    for b in range(2):
        chosen = log_e[b][m[b]]
        rejected = log_e[b][~m[b]]
        assert chosen.min() >= rejected.max()


def test_ebh_mask_equals_top_ranked_mask_at_its_own_budget():
    """e-BH chooses a budget, then selects the top-ranked set at that budget.

    Evidence is monotone in score, so a top-k-at-k* baseline is identical to
    e-BH and must not be presented as an independent experimental method.
    """
    torch.manual_seed(41)
    log_e = torch.randn(3, 32) * 3
    lengths = torch.tensor([32, 21, 7])
    selected = ebh_reject(log_e, lengths, alpha=0.6, min_reject=1, max_frac=0.5)
    recovered = torch.zeros_like(selected)
    for b in range(log_e.shape[0]):
        k = int(selected[b].sum())
        recovered[b:b + 1] = fixed_k_mask(
            log_e[b:b + 1], lengths[b:b + 1], fixed_k=k, min_reject=0)
    assert torch.equal(selected, recovered)


def test_larger_alpha_rejects_at_least_as_much():
    torch.manual_seed(1)
    log_e = torch.randn(4, 64) * 2
    lengths = torch.full((4,), 64)
    prev = -1
    for a in [0.01, 0.05, 0.2, 0.5, 1.0]:
        cur = ebh_reject(log_e, lengths, alpha=a, min_reject=0).sum().item()
        assert cur >= prev
        prev = cur


# ------------------------------------------- isolated strict audit selector
def test_strict_selector_permits_empty_set_on_weak_evidence():
    e = torch.full((2, 64), 0.5)
    selected = strict_ebh_reject(
        _log(e), torch.tensor([64, 19]), alpha=0.05, max_frac=0.25)
    assert selected.sum(1).tolist() == [0, 0]


def test_strict_selector_rechecks_threshold_inside_cap():
    """A strongest-evidence subset of legacy e-BH need not self-consist.

    At alpha=1, the ranks [1, 3, 4] pass before capping, so legacy k*=4 is
    truncated to two.  The second selected e-value is 1.5, below the threshold
    4/2=2 at that final size.  The strict audit rule searches only k<=2 and
    therefore returns the one self-consistent rejection.
    """
    log_e = _log([[4.1, 1.5, 1.4, 1.2]])
    lengths = torch.tensor([4])
    legacy = ebh_reject(
        log_e, lengths, alpha=1.0, min_reject=0, max_frac=0.5)
    strict = strict_ebh_reject(
        log_e, lengths, alpha=1.0, max_frac=0.5)
    assert legacy.tolist() == [[True, True, False, False]]
    assert strict.tolist() == [[True, False, False, False]]


def test_strict_selector_zero_cap_is_an_empty_set():
    selected = strict_ebh_reject(
        torch.full((1, 8), 30.0), torch.tensor([8]),
        alpha=1.0, max_frac=0.0)
    assert not selected.any()


def test_strict_uncapped_matches_mathematical_legacy_rule_without_floor():
    torch.manual_seed(42)
    log_e = torch.randn(4, 41) * 5
    lengths = torch.tensor([41, 29, 11, 1])
    legacy = ebh_reject(
        log_e, lengths, alpha=0.3, min_reject=0, max_frac=1.0)
    strict = strict_ebh_reject(
        log_e, lengths, alpha=0.3, max_frac=1.0)
    assert torch.equal(strict, legacy)


@pytest.mark.parametrize("max_frac", [0.05, 0.10, 0.25, 1.0])
def test_every_strict_rejection_set_is_self_consistent(max_frac):
    torch.manual_seed(84)
    log_e = torch.randn(5, 47) * 8
    lengths = torch.tensor([47, 38, 23, 9, 3])
    alpha = 0.2
    selected = strict_ebh_reject(
        log_e, lengths, alpha=alpha, max_frac=max_frac)
    for index, length in enumerate(lengths.tolist()):
        count = int(selected[index].sum())
        assert count <= math.floor(length * max_frac)
        if count:
            threshold = math.log(length / (alpha * count))
            assert float(log_e[index][selected[index]].min()) >= threshold - 1e-6


@pytest.mark.parametrize(
    ("alpha", "max_frac"),
    [(0.0, 0.2), (1.1, 0.2), (0.2, -0.1), (0.2, 1.1)],
)
def test_strict_selector_rejects_invalid_levels(alpha, max_frac):
    with pytest.raises(ValueError):
        strict_ebh_reject(
            torch.zeros(1, 4), torch.tensor([4]),
            alpha=alpha, max_frac=max_frac)


def test_strict_selector_rejects_nonfinite_valid_evidence():
    with pytest.raises(ValueError, match="finite"):
        strict_ebh_reject(
            torch.tensor([[1.0, float("nan"), 3.0]]), torch.tensor([2]),
            alpha=0.2)


# ------------------------- procedure-level FDR with synthetic valid e-values
def test_rejects_almost_nothing_under_the_null():
    """Chernoff e-values from standard-normal surprises: e = exp(u - 1/2).

    Under H0 this is a valid e-value, so e-BH must almost never fire.
    """
    torch.manual_seed(7)
    u = torch.randn(200, 256)
    log_e = u - 0.5
    m = ebh_reject(log_e, torch.full((200,), 256), alpha=0.05, min_reject=0)
    assert m.float().sum(1).mean().item() < 0.5


def test_finds_a_planted_anomaly():
    torch.manual_seed(8)
    u = torch.randn(1, 256)
    u[0, 100:110] = 12.0  # a genuine, very surprising burst
    log_e = u - 0.5
    m = ebh_reject(log_e, torch.tensor([256]), alpha=0.05, min_reject=0)
    assert m[0, 100:110].all()
    assert m.sum().item() <= 20


def test_adapts_k_to_the_anomaly_length():
    """The whole point: k is read off the evidence, not fixed at T/16+1."""
    torch.manual_seed(9)
    lengths = torch.tensor([256, 256])
    u = torch.randn(2, 256)
    u[0, 50:53] = 12.0    # 3-snippet anomaly
    u[1, 50:130] = 12.0   # 80-snippet anomaly
    m = ebh_reject(u - 0.5, lengths, alpha=0.05, min_reject=0)
    k_short, k_long = m[0].sum().item(), m[1].sum().item()
    assert k_short < 10 < k_long
    assert k_short != 17 and k_long != 17  # neither equals the hard-coded k


# ------------------------------------------------------------------- masking
def test_padding_is_never_selected():
    torch.manual_seed(2)
    log_e = torch.randn(2, 64) * 2
    log_e[:, 10:] = 50.0  # huge values parked in the padding
    m = ebh_reject(log_e, torch.tensor([10, 5]), alpha=1.0, min_reject=1)
    assert not m[0, 10:].any()
    assert not m[1, 5:].any()


def test_threshold_uses_the_valid_length_not_the_padded_one():
    """T in `T/(alpha k)` must be the video's own length."""
    e = torch.tensor([[10.0, 3.0, 0.5, 0.1] + [0.0] * 12])
    short = ebh_reject(_log(e.clamp_min(1e-12)), torch.tensor([4]), alpha=1.0, min_reject=0)
    assert short.sum().item() == 2  # same as the T=4 hand-computed case


def test_length_one_video():
    m = ebh_reject(torch.tensor([[5.0, 9.0, 9.0]]), torch.tensor([1]), alpha=1.0, min_reject=1)
    assert m.tolist() == [[True, False, False]]


# ---------------------------------------------------------------- min_reject
def test_min_reject_floor_is_honoured():
    """MIL needs at least one positive per anomalous bag."""
    m = ebh_reject(torch.full((2, 32), -20.0), torch.tensor([32, 32]), alpha=0.05, min_reject=1)
    assert m.sum(1).tolist() == [1, 1]


def test_min_reject_never_exceeds_the_video_length():
    m = ebh_reject(torch.randn(1, 32), torch.tensor([3]), alpha=0.05, min_reject=10)
    assert m.sum().item() == 3


def test_min_reject_picks_the_argmax():
    log_e = torch.full((1, 16), -20.0)
    log_e[0, 6] = -19.0
    m = ebh_reject(log_e, torch.tensor([16]), alpha=0.05, min_reject=1)
    assert m[0, 6].item() and m.sum().item() == 1


# --------------------------------------------------------------- selection op
def test_selection_is_not_differentiable_but_carries_no_grad():
    log_e = torch.randn(2, 32, requires_grad=True)
    m = ebh_reject(log_e, torch.tensor([32, 32]), alpha=0.5)
    assert not m.requires_grad


# ---------------------------------------------------------------------- loss
def _labels(flags, n_class=7):
    y = torch.zeros(len(flags), n_class)
    for i, f in enumerate(flags):
        y[i, 0 if f == 0 else f] = 1.0
    return y


def test_loss_is_scalar_and_finite():
    torch.manual_seed(3)
    loss = ebh_pseudo_loss(
        torch.randn(4, 64), torch.randn(4, 64), torch.full((4,), 64),
        1 - _labels([0, 0, 1, 2])[:, 0], alpha=0.5,
    )
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_loss_rewards_high_scores_on_selected_snippets():
    T = 64
    log_e = torch.full((1, T), -20.0)
    log_e[0, 10:13] = 20.0
    lengths = torch.tensor([T])
    y = torch.tensor([1.0])

    good = torch.full((1, T), -4.0)
    good[0, 10:13] = 4.0
    bad = torch.full((1, T), -4.0)
    assert ebh_pseudo_loss(good, log_e, lengths, y, alpha=0.5).item() < ebh_pseudo_loss(
        bad, log_e, lengths, y, alpha=0.5
    ).item()


def test_normal_videos_get_dense_negative_supervision():
    """Normal videos in UCF/XD contain no anomalous frame at all, so every
    snippet is a valid negative -- not just the top-k."""
    T = 64
    logits = torch.zeros(1, T, requires_grad=True)
    loss = ebh_pseudo_loss(
        logits, torch.randn(1, T), torch.tensor([T]), torch.tensor([0.0]),
        alpha=0.5, normal_weight=1.0,
    )
    loss.backward()
    assert (logits.grad[0] > 0).all()  # every snippet pushed down


def test_normal_weight_zero_drops_the_normal_term():
    T = 64
    logits = torch.zeros(1, T, requires_grad=True)
    ebh_pseudo_loss(
        logits, torch.randn(1, T), torch.tensor([T]), torch.tensor([0.0]),
        alpha=0.5, normal_weight=0.0,
    ).backward()
    assert logits.grad.abs().sum().item() == 0


def test_loss_ignores_padding():
    T = 64
    torch.manual_seed(4)
    logits = torch.randn(2, T, requires_grad=True)
    log_e = torch.randn(2, T)
    lengths = torch.tensor([20, 8])
    y = torch.tensor([1.0, 0.0])
    ebh_pseudo_loss(logits, log_e, lengths, y, alpha=0.5, normal_weight=1.0).backward()
    assert logits.grad[0, 20:].abs().sum().item() == 0
    assert logits.grad[1, 8:].abs().sum().item() == 0


def test_all_normal_batch_is_finite():
    loss = ebh_pseudo_loss(
        torch.randn(3, 32), torch.randn(3, 32), torch.full((3,), 32),
        torch.zeros(3), alpha=0.5, normal_weight=1.0,
    )
    assert torch.isfinite(loss)


def test_all_anomalous_batch_is_finite():
    loss = ebh_pseudo_loss(
        torch.randn(3, 32), torch.randn(3, 32), torch.full((3,), 32),
        torch.ones(3), alpha=0.5,
    )
    assert torch.isfinite(loss)


def test_evidence_gradient_does_not_flow_through_selection():
    """Pseudo-labels are targets: the selection must be treated as constant."""
    T = 32
    logits = torch.randn(1, T, requires_grad=True)
    log_e = torch.randn(1, T, requires_grad=True)
    ebh_pseudo_loss(logits, log_e, torch.tensor([T]), torch.tensor([1.0]), alpha=0.5).backward()
    assert log_e.grad is None or log_e.grad.abs().sum().item() == 0


@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.5, 1.0])
def test_loss_finite_across_alphas(alpha):
    torch.manual_seed(5)
    loss = ebh_pseudo_loss(
        torch.randn(4, 64), torch.randn(4, 64) * 5, torch.full((4,), 64),
        torch.tensor([0.0, 1, 1, 0]), alpha=alpha, normal_weight=0.5,
    )
    assert torch.isfinite(loss)


# ------------------------------------------------- integration with betting
def test_log_evidence_matches_the_aggregator():
    from utils.betting import DualBranchBetting, abnormal_margin

    head = DualBranchBetting(weight1=1.0, weight2=1.0)
    torch.manual_seed(12)
    T, C = 32, 7
    l1 = torch.randn(4, T, 1)
    l2 = torch.randn(4, T, C)
    y = _labels([0, 0, 1, 2], C)
    n = torch.full((4,), T)

    le1, le2, yb = head.log_evidence(l1, l2, y, n)
    assert le1.shape == (4, T) and le2.shape == (4, T)
    assert yb.tolist() == [0.0, 0.0, 1.0, 1.0]

    c1, s1 = head.agg1.batch_null(l1[:2].squeeze(-1), n[:2])
    assert torch.allclose(le1, head.agg1.log_e_values(l1, c1, s1), atol=1e-5)
    c2, s2 = head.agg2.batch_null(abnormal_margin(l2[:2]), n[:2])
    assert torch.allclose(le2, head.agg2.log_e_values(abnormal_margin(l2), c2, s2), atol=1e-5)


def test_ebh_on_real_evidence_is_finite_and_selective():
    from utils.betting import DualBranchBetting

    head = DualBranchBetting(weight1=1.0, weight2=0.0)
    torch.manual_seed(13)
    T, C = 256, 7
    l1 = torch.randn(4, T, 1) * 0.3
    l1[2, 40:46] += 6.0        # planted anomaly in an anomalous video
    l2 = torch.randn(4, T, C)
    y = _labels([0, 0, 1, 2], C)
    n = torch.full((4,), T)

    le1, _, yb = head.log_evidence(l1, l2, y, n)
    loss = ebh_pseudo_loss(l1, le1, n, yb, alpha=0.5, min_reject=1, normal_weight=1.0)
    assert torch.isfinite(loss)

    sel = ebh_reject(le1[2:3], n[2:3], alpha=0.5, min_reject=1)
    assert sel[0, 40:46].any()          # the planted burst is picked up
    assert sel.sum().item() < 40        # and it stays selective


def test_ebh_gradient_reaches_the_model_logits():
    from utils.betting import DualBranchBetting

    head = DualBranchBetting(weight1=0.0, weight2=1.0)
    torch.manual_seed(14)
    T, C = 64, 7
    l1 = torch.randn(3, T, 1)
    l2 = torch.randn(3, T, C, requires_grad=True)
    y = _labels([0, 1, 2], C)
    n = torch.full((3,), T)
    from utils.betting import abnormal_margin

    _, le2, yb = head.log_evidence(l1, l2, y, n)
    ebh_pseudo_loss(abnormal_margin(l2), le2, n, yb, alpha=0.5).backward()
    assert l2.grad is not None and torch.isfinite(l2.grad).all() and l2.grad.abs().sum() > 0


# ------------------------------------------------------------- max_frac cap
def test_max_frac_caps_the_rejection_set():
    """Self-training feedback: once the trunk separates whole videos, even the
    normal snippets of an anomalous video score high and e-BH rejects almost
    everything.  A hard cap keeps the pseudo-labels localised."""
    log_e = torch.full((2, 256), 30.0)  # overwhelming evidence everywhere
    m = ebh_reject(log_e, torch.full((2,), 256), alpha=0.5, min_reject=1, max_frac=0.2)
    assert m.sum(1).tolist() == [51, 51]  # floor(0.2 * 256) = 51


def test_max_frac_keeps_the_strongest_snippets():
    torch.manual_seed(20)
    log_e = torch.randn(1, 128) * 3 + 30
    m = ebh_reject(log_e, torch.tensor([128]), alpha=1.0, min_reject=1, max_frac=0.1)
    assert m.sum().item() == 12
    assert log_e[0][m[0]].min() >= log_e[0][~m[0]].max()


def test_max_frac_never_drops_below_min_reject():
    m = ebh_reject(torch.full((1, 8), 30.0), torch.tensor([8]), alpha=0.5,
                   min_reject=2, max_frac=0.01)
    assert m.sum().item() == 2


def test_max_frac_disabled_by_default():
    log_e = torch.full((1, 64), 30.0)
    a = ebh_reject(log_e, torch.tensor([64]), alpha=0.5, min_reject=1)
    b = ebh_reject(log_e, torch.tensor([64]), alpha=0.5, min_reject=1, max_frac=1.0)
    assert a.sum().item() == b.sum().item() == 64


def test_max_frac_uses_the_valid_length():
    log_e = torch.full((1, 256), 30.0)
    m = ebh_reject(log_e, torch.tensor([40]), alpha=0.5, min_reject=1, max_frac=0.25)
    assert m.sum().item() == 10  # 0.25 * 40, not 0.25 * 256


# ------------------------------------------------ selector ablation helpers
def test_fixed_k_mask_uses_each_video_valid_length():
    log_e = torch.tensor([[1.0, 4.0, 3.0, 2.0, 100.0],
                          [5.0, 4.0, 3.0, 2.0, 1.0]])
    selected = fixed_k_mask(log_e, torch.tensor([4, 2]), fixed_k=3, min_reject=1)
    assert selected.sum(1).tolist() == [3, 2]
    assert not selected[0, 4]


def test_fixed_frac_mask_has_a_fixed_per_video_fraction():
    selected = fixed_k_mask(torch.randn(2, 20), torch.tensor([20, 11]),
                            fixed_frac=0.25, min_reject=1)
    assert selected.sum(1).tolist() == [5, 2]


def test_original_topk_budget_uses_each_valid_length():
    selected = original_topk_mask(
        torch.randn(3, 256), torch.tensor([256, 31, 15]))
    assert selected.sum(1).tolist() == [17, 2, 1]


def test_original_topk_never_selects_padding():
    log_e = torch.randn(1, 32)
    log_e[0, 9:] = 100.0
    selected = original_topk_mask(log_e, torch.tensor([9]))
    assert selected.sum().item() == 1
    assert not selected[0, 9:].any()


def test_soft_evidence_weights_normalize_and_ignore_padding():
    weights = soft_evidence_weights(torch.tensor([[0.0, 1.0, 20.0, 20.0]]),
                                    torch.tensor([2]), temperature=1.0)
    assert torch.allclose(weights.sum(1), torch.ones(1))
    assert weights[0, 1] > weights[0, 0]
    assert weights[0, 2:].sum().item() == 0


def test_selector_variants_produce_finite_losses_and_gradients():
    for selector, kwargs in [
        ("fixed_k", {"fixed_k": 3}),
        ("fixed_frac", {"fixed_frac": 0.25}),
        ("original_topk", {}),
        ("soft", {"soft_temperature": 1.0}),
        ("normal_only", {}),
    ]:
        logits = torch.randn(2, 16, requires_grad=True)
        loss = pseudo_label_loss(
            logits, torch.randn(2, 16), torch.tensor([16, 10]),
            torch.tensor([1.0, 0.0]), selector=selector, normal_weight=0.5,
            neg_frac=0.1, **kwargs)
        loss.backward()
        assert torch.isfinite(loss) and logits.grad is not None


def test_normal_only_does_not_supervise_anomalous_bag_snippets():
    logits = torch.zeros(2, 8, requires_grad=True)
    pseudo_label_loss(
        logits, torch.randn(2, 8), torch.tensor([8, 8]), torch.tensor([1.0, 0.0]),
        selector="normal_only", normal_weight=1.0).backward()
    assert logits.grad[0].abs().sum().item() == 0
    assert (logits.grad[1] > 0).all()


def test_normal_only_keeps_the_shared_confident_negative_term():
    logits = torch.zeros(2, 8, requires_grad=True)
    log_e = torch.linspace(-4, 4, 8).repeat(2, 1)
    pseudo_label_loss(
        logits, log_e, torch.tensor([8, 8]), torch.tensor([1.0, 0.0]),
        selector="normal_only", normal_weight=1.0, neg_frac=0.25).backward()
    assert (logits.grad[0, :2] > 0).all()
    assert logits.grad[0, 2:].abs().sum().item() == 0


def test_selection_stats_accumulate_training_only_budgets():
    stats = {}
    for length in [8, 5]:
        pseudo_label_loss(
            torch.zeros(1, 8), torch.arange(8).float().unsqueeze(0),
            torch.tensor([length]), torch.tensor([1.0]), selector="fixed_k",
            fixed_k=3, normal_weight=0.0, stats=stats)
    assert stats["selection_video_total"] == 2
    assert stats["selection_selected_total"] == 6
    assert stats["selection_fraction_total"] == pytest.approx(3 / 8 + 3 / 5)


def test_ebh_selector_matches_the_backward_compatible_loss():
    torch.manual_seed(42)
    logits = torch.randn(3, 24)
    log_e = torch.randn(3, 24)
    lengths = torch.tensor([24, 17, 9])
    labels = torch.tensor([1.0, 0.0, 1.0])
    old = ebh_pseudo_loss(
        logits, log_e, lengths, labels, alpha=0.3, min_reject=1,
        normal_weight=0.2, max_frac=0.25, neg_frac=0.1)
    new = pseudo_label_loss(
        logits, log_e, lengths, labels, alpha=0.3, min_reject=1,
        normal_weight=0.2, max_frac=0.25, neg_frac=0.1, selector="ebh")
    assert torch.allclose(old, new)


def test_cap_flows_through_the_loss():
    T = 256
    logits = torch.zeros(1, T, requires_grad=True)
    ebh_pseudo_loss(logits, torch.full((1, T), 30.0), torch.tensor([T]),
                    torch.tensor([1.0]), alpha=0.5, min_reject=1,
                    normal_weight=0.0, max_frac=0.1).backward()
    assert int((logits.grad[0] != 0).sum()) == 25


# ------------------------------------------------- two-sided pseudo-labelling
def test_neg_frac_supervises_the_quietest_snippets_of_anomalous_videos():
    """The dominant AUC error is *normal* frames inside anomalous videos: the
    trunk recognises "this is an anomalous video" and lifts the whole thing.
    Snippets whose e-value sits far below the null are confidently normal even
    inside an anomalous bag, so they are safe dense negatives."""
    T = 64
    log_e = torch.linspace(-10, 10, T).unsqueeze(0)   # ascending evidence
    logits = torch.zeros(1, T, requires_grad=True)
    ebh_pseudo_loss(logits, log_e, torch.tensor([T]), torch.tensor([1.0]),
                    alpha=0.5, min_reject=1, normal_weight=1.0,
                    max_frac=0.1, neg_frac=0.25).backward()
    g = logits.grad[0]
    assert (g[:16] > 0).all()       # quietest quarter pushed down
    assert (g[-1] < 0)              # loudest snippet pushed up


def test_neg_frac_zero_is_a_no_op():
    T = 32
    torch.manual_seed(30)
    log_e = torch.randn(1, T)
    logits = torch.randn(1, T, requires_grad=True)
    a = ebh_pseudo_loss(logits, log_e, torch.tensor([T]), torch.tensor([1.0]),
                        alpha=0.5, neg_frac=0.0)
    b = ebh_pseudo_loss(logits, log_e, torch.tensor([T]), torch.tensor([1.0]),
                        alpha=0.5)
    assert torch.allclose(a, b, atol=1e-7)


def test_positive_and_negative_sets_never_overlap():
    from utils.ebh import bottom_frac_mask, ebh_reject as _rej
    T = 64
    torch.manual_seed(31)
    log_e = torch.randn(1, T) * 3
    lengths = torch.tensor([T])
    pos = _rej(log_e, lengths, alpha=1.0, min_reject=1, max_frac=0.5)
    neg = bottom_frac_mask(log_e, lengths, 0.5, exclude=pos)
    assert not (pos & neg).any()


def test_bottom_frac_picks_the_lowest_evidence():
    from utils.ebh import bottom_frac_mask
    torch.manual_seed(32)
    log_e = torch.randn(2, 64)
    m = bottom_frac_mask(log_e, torch.full((2,), 64), 0.25)
    for b in range(2):
        assert m[b].sum().item() == 16
        assert log_e[b][m[b]].max() <= log_e[b][~m[b]].min()


def test_bottom_frac_respects_padding():
    from utils.ebh import bottom_frac_mask
    log_e = torch.randn(1, 64)
    log_e[0, 10:] = -1e3          # padding would otherwise dominate the bottom
    m = bottom_frac_mask(log_e, torch.tensor([10]), 0.5)
    assert not m[0, 10:].any() and m.sum().item() == 5


def test_bottom_frac_full_is_everything_valid():
    from utils.ebh import bottom_frac_mask
    m = bottom_frac_mask(torch.randn(1, 32), torch.tensor([12]), 1.0)
    assert m.sum().item() == 12


def test_two_sided_loss_is_finite_and_scalar():
    torch.manual_seed(33)
    loss = ebh_pseudo_loss(torch.randn(4, 128), torch.randn(4, 128) * 3,
                           torch.full((4,), 128), torch.tensor([0.0, 1, 1, 0]),
                           alpha=0.05, min_reject=1, normal_weight=1.0,
                           max_frac=0.2, neg_frac=0.3)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_two_sided_ignores_padding():
    T = 64
    torch.manual_seed(34)
    logits = torch.randn(2, T, requires_grad=True)
    ebh_pseudo_loss(logits, torch.randn(2, T), torch.tensor([20, 8]),
                    torch.tensor([1.0, 1.0]), alpha=0.5, normal_weight=1.0,
                    max_frac=0.2, neg_frac=0.4).backward()
    assert logits.grad[0, 20:].abs().sum().item() == 0
    assert logits.grad[1, 8:].abs().sum().item() == 0


def test_two_sided_reports_negative_count():
    st = {}
    T = 128
    ebh_pseudo_loss(torch.randn(1, T), torch.randn(1, T), torch.tensor([T]),
                    torch.tensor([1.0]), alpha=0.5, stats=st, max_frac=0.2,
                    neg_frac=0.25)
    assert st["k"] >= 1 and st["kneg"] == 32
