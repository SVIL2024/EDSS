"""e-BH-inspired adaptive snippet selection for weakly-supervised VAD.

VadCLIP supervises a *fixed* ``k = T/16 + 1`` snippets per video. That constant
encodes an approximately 6.25% selection budget without observing the event's
true duration. When an event is shorter than the budget, normal snippets can
receive upward pressure.

The e-Benjamini-Hochberg procedure (Wang & Ramdas, *False discovery rate
control with e-values*, JRSS-B 2022) reads the number of rejections off the
evidence instead:

    sort descending  e_(1) >= ... >= e_(T)
    k* = max { k : e_(k) >= T / (alpha * k) }
    reject the k* largest

For valid e-values, the mathematical e-BH procedure controls FDR at level
``alpha`` under arbitrary dependence.  The learned scores and estimated null
used here have not satisfied that calibration empirically, so this module uses
e-BH as an adaptive evidence-budget rule and does **not** claim deployed FDR
control.  See ``docs/LIMITATIONS.md``.

The e-values come from :mod:`utils.betting`, so the optional bag process and the
snippet selector share one estimated normal reference. That shared reference
is not empirically calibrated for normal snippets inside anomalous videos.
"""

import math

import torch
import torch.nn.functional as F

from .betting import _to_bt, length_mask

__all__ = [
    "ebh_reject",
    "strict_ebh_reject",
    "bottom_frac_mask",
    "fixed_k_mask",
    "original_topk_mask",
    "soft_evidence_weights",
    "pseudo_label_loss",
    "ebh_pseudo_loss",
    "null_alignment_loss",
]

_NEG = -1e30


@torch.no_grad()
def ebh_reject(log_e: torch.Tensor, lengths: torch.Tensor, alpha: float = 0.5,
               min_reject: int = 1, max_frac: float = 1.0) -> torch.Tensor:
    """Boolean ``[B, T]`` mask of snippets rejected by e-BH.

    Everything is done in log space: ``e_(k) >= T/(alpha k)`` becomes
    ``log e_(k) >= log T - log alpha - log k``, so arbitrarily strong evidence
    never overflows.

    ``min_reject`` floors the selection (MIL needs at least one positive per
    anomalous bag) and is itself capped by the video length.

    ``max_frac`` caps the rejection set at a fraction of the video.  This is a
    legacy training heuristic: simply shrinking ``k*`` can break e-BH's
    self-consistency threshold at the smaller rejection count.  Likewise, a
    positive ``min_reject`` forces discoveries unsupported by e-BH.  They are
    retained here so completed and active training runs keep identical
    semantics.  Use :func:`strict_ebh_reject` for rule-level FDR audits.
    """
    le = _to_bt(log_e).detach().float()
    B, T = le.shape
    device = le.device
    mask = length_mask(lengths, T, device, torch.bool)

    le = le.masked_fill(~mask, _NEG)                    # padding sorts last
    sorted_le, order = torch.sort(le, dim=1, descending=True)

    rank = torch.arange(1, T + 1, device=device, dtype=torch.float32).unsqueeze(0)
    n = mask.sum(1, keepdim=True).clamp_min(1).float()  # the video's own T
    thresh = torch.log(n) - torch.log(torch.tensor(float(alpha), device=device)) - torch.log(rank)

    passes = (sorted_le >= thresh) & (rank <= n)
    # k* = the LARGEST passing rank (not the first failure)
    kstar = (passes.float() * rank).max(dim=1, keepdim=True).values

    if max_frac < 1.0:
        ceiling = torch.floor(n * float(max_frac)).clamp_min(1.0)
        kstar = kstar.minimum(ceiling)
    floor = torch.full_like(kstar, float(min_reject))
    kstar = torch.maximum(kstar, floor).minimum(n)

    keep_sorted = rank <= kstar
    out = torch.zeros_like(keep_sorted)
    out.scatter_(1, order, keep_sorted)
    return out & mask


@torch.no_grad()
def strict_ebh_reject(log_e: torch.Tensor, lengths: torch.Tensor,
                      alpha: float = 0.5,
                      max_frac: float = 1.0) -> torch.Tensor:
    """Self-consistent e-BH mask for an isolated rule-level FDR audit.

    The returned count is

    ``max {k <= floor(n * max_frac): log_e_(k) >= log(n / (alpha * k))}``.

    Unlike the training selector, this function always permits the empty set,
    never imposes a rejection floor, and re-solves the e-BH inequality *inside*
    the requested cap.  Consequently every non-empty returned set satisfies
    the e-BH self-consistency threshold at its final size.  This fixes the rule
    mechanics only: FDR control still additionally requires valid input
    e-values, which the learned VAD evidence has not established.

    This function is deliberately not wired into training.  It exists so
    strict audits cannot silently alter the semantics of registered runs.
    """
    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not math.isfinite(float(max_frac)) or not 0.0 <= float(max_frac) <= 1.0:
        raise ValueError("max_frac must be finite and in [0, 1]")

    le = _to_bt(log_e).detach().float()
    batch, width = le.shape
    lens = torch.as_tensor(lengths, device=le.device).reshape(-1)
    if lens.numel() != batch:
        raise ValueError(
            f"lengths has {lens.numel()} entries for a batch of {batch}")
    if bool(((lens <= 0) | (lens > width)).any()):
        raise ValueError(f"every length must be in [1, {width}]")

    valid = length_mask(lens, width, le.device, torch.bool)
    if not bool(torch.isfinite(le[valid]).all()):
        raise ValueError("valid log_e entries must all be finite")
    sorted_le, order = torch.sort(
        le.masked_fill(~valid, _NEG), dim=1, descending=True)

    rank = torch.arange(
        1, width + 1, device=le.device, dtype=torch.float32).unsqueeze(0)
    n = valid.sum(1, keepdim=True).float()
    threshold = (
        torch.log(n)
        - math.log(float(alpha))
        - torch.log(rank)
    )
    ceiling = torch.floor(n * float(max_frac))
    passes = (
        (sorted_le >= threshold)
        & (rank <= n)
        & (rank <= ceiling)
    )
    kstar = (passes.to(rank.dtype) * rank).max(dim=1, keepdim=True).values

    keep_sorted = rank <= kstar
    selected = torch.zeros_like(keep_sorted)
    selected.scatter_(1, order, keep_sorted)
    return selected & valid


@torch.no_grad()
def bottom_frac_mask(log_e: torch.Tensor, lengths: torch.Tensor, frac: float,
                     exclude: torch.Tensor = None) -> torch.Tensor:
    """``[B, T]`` mask of the ``frac`` lowest-evidence valid snippets per video.

    Used to mine *confident normals inside anomalous videos*: snippets whose
    e-value sits far below the null are normal even though their bag is
    labelled anomalous.  ``exclude`` (typically the e-BH rejection set) is
    pushed to the top of the ordering so the two sets can never overlap.
    """
    le = _to_bt(log_e).detach().float()
    B, T = le.shape
    mask = length_mask(lengths, T, le.device, torch.bool)
    if exclude is not None:
        le = le.masked_fill(exclude, -_NEG)          # never pick these
    le = le.masked_fill(~mask, -_NEG)                # nor the padding
    order = torch.argsort(le, dim=1, descending=False)

    n = mask.sum(1, keepdim=True).float()
    m = torch.floor(n * float(frac))
    if exclude is not None:  # never ask for more than is actually available
        m = m.minimum((mask & ~exclude).sum(1, keepdim=True).float())
    m = m.clamp_min(0.0)

    rank = torch.arange(1, T + 1, device=le.device, dtype=torch.float32).unsqueeze(0)
    keep_sorted = rank <= m
    out = torch.zeros_like(keep_sorted)
    out.scatter_(1, order, keep_sorted)
    return out & mask


@torch.no_grad()
def fixed_k_mask(log_e: torch.Tensor, lengths: torch.Tensor, fixed_k: int = 0,
                 fixed_frac: float = 0.0, min_reject: int = 1,
                 max_frac: float = 1.0) -> torch.Tensor:
    """Top-ranked mask with a fixed count or a fixed fraction per video.

    This is a controlled ablation of e-BH: it uses exactly the same evidence
    ranking, but its budget is independent of the evidence distribution.  Set
    exactly one of ``fixed_k`` or ``fixed_frac``.  A valid-video length is used
    throughout, so padding cannot influence the selection.
    """
    if (fixed_k > 0) == (fixed_frac > 0):
        raise ValueError("set exactly one of fixed_k and fixed_frac")
    if fixed_frac < 0 or fixed_frac > 1:
        raise ValueError("fixed_frac must be in [0, 1]")

    le = _to_bt(log_e).detach().float()
    B, T = le.shape
    mask = length_mask(lengths, T, le.device, torch.bool)
    le = le.masked_fill(~mask, _NEG)
    _, order = torch.sort(le, dim=1, descending=True)

    n = mask.sum(1, keepdim=True).float()
    if fixed_k > 0:
        count = torch.full_like(n, float(fixed_k))
    else:
        count = torch.floor(n * float(fixed_frac))
    if max_frac < 1.0:
        count = count.minimum(torch.floor(n * float(max_frac)).clamp_min(1.0))
    count = count.maximum(torch.full_like(count, float(min_reject))).minimum(n)

    rank = torch.arange(1, T + 1, device=le.device, dtype=torch.float32).unsqueeze(0)
    keep_sorted = rank <= count
    out = torch.zeros_like(keep_sorted)
    out.scatter_(1, order, keep_sorted)
    return out & mask


@torch.no_grad()
def original_topk_mask(log_e: torch.Tensor, lengths: torch.Tensor,
                       min_reject: int = 1,
                       max_frac: float = 1.0) -> torch.Tensor:
    """Top-ranked mask using VadCLIP's original ``floor(T / 16) + 1`` budget.

    The ranking is kept identical to the other selector ablations; only the
    per-video count follows the original fixed-duration prior.  Valid lengths,
    rather than padded tensor width, determine the count.
    """
    le = _to_bt(log_e).detach().float()
    _, width = le.shape
    valid = length_mask(lengths, width, le.device, torch.bool)
    le = le.masked_fill(~valid, _NEG)
    _, order = torch.sort(le, dim=1, descending=True)

    n = valid.sum(1, keepdim=True).float()
    count = torch.floor(n / 16.0) + 1.0
    if max_frac < 1.0:
        ceiling = torch.floor(n * float(max_frac)).clamp_min(1.0)
        count = count.minimum(ceiling)
    count = count.maximum(torch.full_like(count, float(min_reject))).minimum(n)

    rank = torch.arange(1, width + 1, device=le.device,
                        dtype=torch.float32).unsqueeze(0)
    keep_sorted = rank <= count
    out = torch.zeros_like(keep_sorted)
    out.scatter_(1, order, keep_sorted)
    return out & valid


@torch.no_grad()
def soft_evidence_weights(log_e: torch.Tensor, lengths: torch.Tensor,
                          temperature: float = 1.0) -> torch.Tensor:
    """Continuous evidence weights for the soft-selection baseline.

    Weights sum to one in every video and are detached pseudo-target weights.
    Thus this baseline changes only hard closed-form selection to a continuous
    attention-style positive loss; it does not add a learned selector.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    le = _to_bt(log_e).detach().float()
    mask = length_mask(lengths, le.shape[1], le.device, torch.bool)
    scaled = le / float(temperature)
    scaled = scaled.masked_fill(~mask, _NEG)
    weights = torch.softmax(scaled, dim=1)
    return weights * mask.to(weights.dtype)


def pseudo_label_loss(logits, log_e, lengths, labels, alpha: float = 0.5,
                      min_reject: int = 1, normal_weight: float = 1.0, stats=None,
                      max_frac: float = 1.0, neg_frac: float = 0.0,
                      selector: str = "ebh", fixed_k: int = 0,
                      fixed_frac: float = 0.0, soft_temperature: float = 1.0):
    """Dense snippet BCE with an interchangeable positive pseudo-label selector.

    ``selector`` is one of ``ebh`` (the proposed closed-form adaptive rule),
    ``fixed_k``, ``fixed_frac``, ``original_topk``, ``soft`` (continuous
    evidence weighting), or ``normal_only`` (no anomalous-bag pseudo-positive
    term; the shared dense-normal and anomaly-bottom negative terms remain).
    All variants keep the exact normal-video loss and optional confident-negative
    mining, making the selector itself the only experimental difference.

    * anomalous videos -- the e-BH rejection set is labelled 1.  ``k`` adapts to
      the evidence instead of being pinned at ``T/16 + 1``;
    * normal videos -- *every* snippet is labelled 0.  UCF-Crime and
      XD-Violence normal videos contain no anomalous frame at all, so this is
      exact supervision rather than a pseudo-label.

    The selection is a hard, non-differentiable target (``no_grad``); gradients
    flow only through ``logits``.
    """
    s = _to_bt(logits)
    mask = length_mask(lengths, s.shape[1], s.device, s.dtype)
    y = labels.to(device=s.device, dtype=s.dtype).reshape(-1)
    is_anom = y > 0.5

    # start from a zero that is still attached to the graph, so callers can
    # always call .backward() even when no term is active this step
    loss = (s * 0.0).sum()

    if selector not in {
        "ebh", "fixed_k", "fixed_frac", "original_topk", "soft", "normal_only"
    }:
        raise ValueError(f"unknown selector: {selector}")

    if bool(is_anom.any()):
        sa, la = s[is_anom], lengths[is_anom]
        sel = None
        if selector != "normal_only":
            pos = F.binary_cross_entropy_with_logits(
                sa, torch.ones_like(sa), reduction="none"
            )

            if selector == "ebh":
                sel = ebh_reject(log_e[is_anom], la, alpha, min_reject, max_frac)
                pos_weights = sel.to(s.dtype)
            elif selector == "fixed_k":
                sel = fixed_k_mask(log_e[is_anom], la, fixed_k=fixed_k,
                                   min_reject=min_reject, max_frac=max_frac)
                pos_weights = sel.to(s.dtype)
            elif selector == "fixed_frac":
                sel = fixed_k_mask(log_e[is_anom], la, fixed_frac=fixed_frac,
                                   min_reject=min_reject, max_frac=max_frac)
                pos_weights = sel.to(s.dtype)
            elif selector == "original_topk":
                sel = original_topk_mask(
                    log_e[is_anom], la, min_reject=min_reject,
                    max_frac=max_frac)
                pos_weights = sel.to(s.dtype)
            else:
                pos_weights = soft_evidence_weights(
                    log_e[is_anom], la, temperature=soft_temperature).to(s.dtype)

            loss = loss + (pos * pos_weights).sum() / pos_weights.sum().clamp_min(1.0)
            if stats is not None:
                if sel is not None:
                    stats["k"] = float(sel.sum(1).float().mean())
                    fractions = sel.sum(1).float() / la.float().clamp_min(1.0)
                    stats["selection_selected_total"] = (
                        stats.get("selection_selected_total", 0.0)
                        + float(sel.sum())
                    )
                    stats["selection_video_total"] = (
                        stats.get("selection_video_total", 0) + int(sel.shape[0])
                    )
                    stats["selection_fraction_total"] = (
                        stats.get("selection_fraction_total", 0.0)
                        + float(fractions.sum())
                    )
                else:
                    stats["k"] = float((1.0 / pos_weights.square().sum(1).clamp_min(1e-12)).mean())

        # confident normals *inside* anomalous videos -- the dominant AUC error
        # This remains active for ``normal_only``.  The control removes only
        # anomalous-bag pseudo-positives while preserving shared negative mining.
        if neg_frac > 0 and normal_weight > 0:
            quiet = bottom_frac_mask(log_e[is_anom], la, neg_frac, exclude=sel)
            qf = quiet.to(s.dtype)
            negA = F.binary_cross_entropy_with_logits(
                sa, torch.zeros_like(sa), reduction="none"
            )
            loss = loss + normal_weight * (negA * qf).sum() / qf.sum().clamp_min(1.0)
            if stats is not None:
                stats["kneg"] = float(quiet.sum(1).float().mean())

    if normal_weight > 0 and bool((~is_anom).any()):
        m = mask[~is_anom]
        neg = F.binary_cross_entropy_with_logits(
            s[~is_anom], torch.zeros_like(s[~is_anom]), reduction="none"
        )
        loss = loss + normal_weight * (neg * m).sum() / m.sum().clamp_min(1.0)

    return loss


def ebh_pseudo_loss(logits, log_e, lengths, labels, alpha: float = 0.5,
                    min_reject: int = 1, normal_weight: float = 1.0, stats=None,
                    max_frac: float = 1.0, neg_frac: float = 0.0):
    """Backward-compatible e-BH specialization of :func:`pseudo_label_loss`."""
    return pseudo_label_loss(
        logits, log_e, lengths, labels, alpha=alpha, min_reject=min_reject,
        normal_weight=normal_weight, stats=stats, max_frac=max_frac,
        neg_frac=neg_frac, selector="ebh")


def null_alignment_loss(scores, log_e, lengths, mu: float, sigma: float,
                        sel: torch.Tensor = None):
    """Pull the non-selected bulk of a video toward the normal-video null.

    Motivation (``docs/LIMITATIONS.md``): the selector's ideal FDR contract
    fails because the *true* null -- normal snippets inside anomalous videos --
    sits **+2.00 sigma** above the assumed null (snippets of fully-normal
    videos).  Weakly-supervised MIL lifts whole anomalous videos, so the
    e-values of genuinely normal snippets are inflated ~7.6x and e-BH
    over-rejects by construction.

    This loss enforces the null contract *during training*: the snippets e-BH
    does **not** select must behave like the normal regime -- their
    standardised scores ``u = (s - mu)/sigma`` under the EMA normal-video null
    should be standard normal.  Selection and null enforcement are computed
    from the same e-values, so the two co-calibrate: select the true anomalies,
    and everything left over must look null.

    ``mu`` / ``sigma`` are the **detached** EMA null tracked from normal videos
    (see :class:`utils.betting.BettingAggregator`), so no gradient flows to the
    null estimate; only ``scores`` gets gradient.  ``log_e`` is detached
    inside (it only chooses the null set), and ``sel`` (the e-BH rejection set)
    excludes selected snippets from the null population.

    The loss is
    ``E[ (u - 0)^2 ] / 2 + (Var[u] - 1)^2 / 2``
    -- a location term matching the null mean, and a scale term matching the
    null variance -- evaluated only over the null population.
    """
    s = _to_bt(scores).float()
    mask = length_mask(lengths, s.shape[1], s.device, s.dtype)
    null = mask.bool()
    if sel is not None:
        null = null & ~sel.to(device=s.device, dtype=torch.bool)
    le = _to_bt(log_e).detach().float()
    null = null & ~(le <= _NEG / 2).bool()           # padding / exclusions
    cnt = null.sum()
    if cnt == 0:
        return (s * 0.0).sum(), {"u_mean": 0.0, "u_std": 0.0}
    m = cnt.clamp_min(1.0)

    u = (s - mu) / sigma
    um = (u * null).sum() / m
    var = (((u - um) ** 2) * null).sum() / m
    loss = 0.5 * um * um + 0.5 * (var - 1.0) * (var - 1.0)
    loss = (s * 0.0).sum() + loss                     # zero-graph node

    stats = {
        "u_mean": float(um.detach()),
        "u_std": float(torch.sqrt(var).detach()),
    }
    return loss, stats
