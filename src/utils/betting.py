"""Testing-by-betting evidence aggregation for weakly-supervised MIL.

Why
---
Multiple-instance learning in video anomaly detection is, literally, a test of
the global null hypothesis

    H0 : *every* snippet of this video is normal.

The community aggregates snippet scores with a hard ``top-k`` mean
(VadCLIP uses ``k = T/16 + 1``).  That estimator is (i) non-smooth, (ii) blind
to *how* surprising a snippet is relative to the normal regime, and (iii)
pinned to an approximately 6.25% selection budget regardless of the event's
unknown duration. Selecting beyond a short event can push normal snippets in
an anomalous video upward.

Game-theoretic statistics (Ville 1939; Shafer & Vovk, *Game-Theoretic
Foundations for Probability and Finance*; Grunwald/de Heide/Koolen, *Safe
Testing*; Ramdas et al., *e-values and e-processes*) answers exactly this
question with a **wealth process**.  A sceptic starts with capital 1 and
repeatedly bets a fraction ``lam`` of it against H0 at fair odds:

    u_t  = (s_t - mu) / sigma                   standardised surprise
    e_t  = exp(eta * u_t - eta^2 / 2)           Chernoff / exponential-tilt e-value
                                                (E_H0[e_t] = 1 exactly for u ~ N(0,1))
    K_t  = K_{t-1} * (1 + lam * (e_t - 1))      Kelly bet, K_0 = 1
    logK = sum_t log(1 + lam * (e_t - 1))

``(mu, sigma)`` describe the *normal regime* and are EMA-tracked online from
the normal videos that weak supervision hands us for free.  Under the idealised
H0 in which this standardisation really produces the assumed null, capital is a
non-negative martingale and Ville's inequality applies.  The deployed learned
scores do not satisfy that calibration: normal snippets inside anomalous videos
are distribution-shifted.  The project therefore uses this construction as a
training-time evidence score and makes no anytime-valid or deployed-FDR claim.
See ``docs/LIMITATIONS.md``.

What it buys us
---------------
* ``e_t`` is unbounded above, so one strongly surprising snippet already
  multiplies the capital -- no built-in duration prior, short anomalies survive.
* If the optional bag loss is enabled, every snippet moves the capital; the
  registered winning recipes disable that bag loss and use detached evidence
  only for pseudo-label selection.
* ``log(1 + lam(e-1)) >= log(1 - lam)``: bounded below, smooth, stable.
* ``eta`` and ``lam`` can be learned when the optional differentiable bag loss
  is enabled. They are not learned by the detached hard selector alone.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BettingAggregator",
    "DualBranchBetting",
    "betting_mil_loss",
    "length_mask",
    "abnormal_margin",
]


def _to_bt(x: torch.Tensor) -> torch.Tensor:
    """Accept ``[B, T]`` or ``[B, T, 1]`` (the shape ``model.py`` emits)."""
    if x.dim() == 3 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    if x.dim() != 2:
        raise ValueError(f"expected [B, T] or [B, T, 1], got {tuple(x.shape)}")
    return x


def length_mask(lengths: torch.Tensor, max_len: int, device, dtype=torch.float32) -> torch.Tensor:
    """``[B, T]`` float mask, 1 for valid snippets."""
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    lens = lengths.to(device=device).reshape(-1, 1)
    return (idx < lens).to(dtype)


def abnormal_margin(logits2: torch.Tensor) -> torch.Tensor:
    """Log-odds of the alignment-branch anomaly score.

    The evaluation protocol scores a snippet with ``1 - softmax(logits2)[..., 0]``.
    Its logit is ``logsumexp(logits2[..., 1:]) - logits2[..., 0]``, computed here
    in a numerically stable way so the betting head sees an unbounded margin
    that is strictly monotone in the metric being optimised.
    """
    return torch.logsumexp(logits2[..., 1:], dim=-1) - logits2[..., 0]


class BettingAggregator(nn.Module):
    """Kelly wealth process over per-snippet anomaly margins.

    Parameters
    ----------
    init_eta / eta_max / learn_eta
        Exponential tilt of the e-value.  Bounded by ``eta_max * sigmoid(raw)``.
    init_lambda / lambda_max / learn_lambda
        Kelly bet fraction, kept in ``(0, lambda_max)`` so ``log(1-lam)`` is finite.
    null_momentum / var_min
        EMA tracker for the normal-regime mean/variance of the margin.
    normalize
        ``"mean"`` returns the Kelly *growth rate* (log-capital per snippet),
        which is O(1) across variable-length videos; ``"none"`` returns the raw
        log-capital, the quantity Ville's inequality applies to.
    init_tau / init_bias / learn_scale
        Affine read-out mapping growth rate to a bag logit.
    """

    def __init__(
        self,
        init_eta: float = 1.0,
        eta_max: float = 4.0,
        learn_eta: bool = True,
        init_lambda: float = 0.2,
        lambda_max: float = 0.95,
        learn_lambda: bool = True,
        null_momentum: float = 0.95,
        var_min: float = 1e-4,
        exp_clamp: float = 30.0,
        normalize: str = "mean",
        init_tau: float = 20.0,
        init_bias: float = 0.0,
        learn_scale: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        if normalize not in ("mean", "none"):
            raise ValueError("normalize must be 'mean' or 'none'")
        self.normalize = normalize
        self.eta_max = float(eta_max)
        self.lambda_max = float(lambda_max)
        self.learn_eta = bool(learn_eta)
        self.learn_lambda = bool(learn_lambda)
        self.null_momentum = float(null_momentum)
        self.var_min = float(var_min)
        self.exp_clamp = float(exp_clamp)
        self.eps = float(eps)

        self._make_bounded("eta", init_eta, self.eta_max, learn_eta)
        self._make_bounded("lambda", init_lambda, self.lambda_max, learn_lambda)

        tau_raw = math.log(math.expm1(init_tau)) if init_tau > 0 else -10.0
        if learn_scale:
            self.tau_raw = nn.Parameter(torch.tensor(tau_raw))
            self.bias = nn.Parameter(torch.tensor(float(init_bias)))
        else:
            self.register_buffer("tau_raw", torch.tensor(tau_raw))
            self.register_buffer("bias", torch.tensor(float(init_bias)))

        self.register_buffer("mu", torch.tensor(0.0))
        self.register_buffer("var", torch.tensor(1.0))
        self.register_buffer("null_seen", torch.tensor(0.0))

    # ------------------------------------------------------------------ #
    def _make_bounded(self, name, init, upper, learn):
        """Store ``init`` as a sigmoid pre-activation, or as a frozen constant."""
        if learn:
            r = min(max(init / upper, 1e-4), 1 - 1e-4)
            setattr(self, f"{name}_raw", nn.Parameter(torch.tensor(math.log(r / (1 - r)))))
        else:
            self.register_buffer(f"{name}_const", torch.tensor(float(init)))

    @property
    def eta(self) -> torch.Tensor:
        if self.learn_eta:
            return self.eta_max * torch.sigmoid(self.eta_raw)
        return self.eta_const

    @property
    def lam(self) -> torch.Tensor:
        if self.learn_lambda:
            return self.lambda_max * torch.sigmoid(self.lambda_raw)
        return self.lambda_const

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self.tau_raw)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_null(self, scores: torch.Tensor, lengths: torch.Tensor) -> None:
        """EMA-track the normal-regime mean/variance of the snippet margin.

        ``scores`` must come from *normal* videos only.
        """
        s = _to_bt(scores).detach().float()
        if s.numel() == 0:
            return
        mask = length_mask(lengths, s.shape[1], s.device, s.dtype)
        n = mask.sum().clamp_min(1.0)
        mean = (s * mask).sum() / n
        var = ((s - mean) ** 2 * mask).sum() / n

        m = 0.0 if self.null_seen.item() == 0 else self.null_momentum
        self.mu.copy_(m * self.mu + (1.0 - m) * mean)
        self.var.copy_((m * self.var + (1.0 - m) * var).clamp_min(self.var_min))
        self.null_seen.add_(1.0)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def batch_null(self, scores: torch.Tensor, lengths: torch.Tensor):
        """Location/scale of the null regime from *this batch's* normal videos.

        Returned **with gradient** on purpose.  Standardising by statistics that
        move with the scores makes the wealth statistic invariant to any affine
        transform of the whole batch, which removes the two degenerate
        directions the first betting run fell into (translate everything down;
        inflate the spread of the normal videos).  Absolute calibration stays
        the job of the top-k losses; the sceptic only reads distribution shape.
        """
        s = _to_bt(scores).float()
        mask = length_mask(lengths, s.shape[1], s.device, s.dtype)
        n = mask.sum().clamp_min(1.0)
        center = (s * mask).sum() / n
        var = ((s - center) ** 2 * mask).sum() / n
        scale = torch.sqrt(var.clamp_min(self.var_min) + self.eps)
        return center, scale

    def standardize(self, scores, center=None, scale=None) -> torch.Tensor:
        # the wealth process is evaluated in fp32: logaddexp has no half kernel
        # and the log-capital of a long video needs the headroom anyway
        s = _to_bt(scores).float()
        if center is None:
            center = self.mu
        if scale is None:
            scale = torch.sqrt(self.var.clamp_min(self.var_min) + self.eps)
        return (s - center) / scale

    def log_e_values(self, scores, center=None, scale=None) -> torch.Tensor:
        """``log e_t = eta*u_t - eta^2/2``; kept in log space so nothing overflows."""
        u = self.standardize(scores, center, scale)
        eta = self.eta
        expo = eta * u - 0.5 * eta * eta
        return expo.clamp(-self.exp_clamp, self.exp_clamp)

    def e_values(self, scores, center=None, scale=None) -> torch.Tensor:
        """Chernoff e-values ``exp(eta*u - eta^2/2)``; ``E_H0[e] = 1``."""
        return torch.exp(self.log_e_values(scores, center, scale))

    def log_wealth(self, scores, lengths, center=None, scale=None) -> torch.Tensor:
        """``[B]`` log-capital (or growth rate) of the sceptic's betting game.

        ``log(1 + lam(e-1)) = logaddexp(log(1-lam), log(lam) + log e)`` -- exact
        and overflow-free for arbitrarily strong evidence, with a live gradient
        everywhere (a hard clamp on ``e`` would zero it on the very snippets
        that matter most).
        """
        log_e = self.log_e_values(scores, center, scale)
        lam = self.lam
        log_w = torch.logaddexp(
            torch.log1p(-lam).expand_as(log_e),
            torch.log(lam.clamp_min(1e-38)) + log_e,
        )

        mask = length_mask(lengths, log_w.shape[1], log_w.device, log_w.dtype)
        s = (log_w * mask).sum(1)
        if self.normalize == "mean":
            s = s / mask.sum(1).clamp_min(1.0)
        return s

    # ------------------------------------------------------------------ #
    def forward(self, scores, lengths, center=None, scale=None) -> torch.Tensor:
        """Snippet margins -> video-level bag logit ``[B]``."""
        return self.tau * self.log_wealth(scores, lengths, center, scale) + self.bias

    def extra_repr(self) -> str:
        return (
            f"eta={float(self.eta):.3f}, lam={float(self.lam):.3f}, "
            f"mu={float(self.mu):.3f}, sigma={float(self.var).__pow__(0.5):.3f}, "
            f"tau={float(self.tau):.2f}, bias={float(self.bias):.2f}, "
            f"normalize={self.normalize}"
        )


def betting_mil_loss(scores, labels, lengths, aggregator: BettingAggregator,
                     center=None, scale=None):
    """Binary cross-entropy on the game-theoretic bag evidence.

    ``scores``  per-snippet real-valued anomaly margin, ``[B, T]`` or ``[B, T, 1]``
    ``labels``  1 for anomalous videos, 0 for normal ones
    """
    bag_logit = aggregator(scores, lengths, center, scale)
    y = labels.to(device=bag_logit.device, dtype=bag_logit.dtype).reshape(-1)
    return F.binary_cross_entropy_with_logits(bag_logit, y)


class DualBranchBetting(nn.Module):
    """Wealth processes for VadCLIP's two branches.

    VadCLIP scores a snippet twice: ``sigmoid(logits1)`` from the classification
    branch (the UCF-Crime AUC protocol) and ``1 - softmax(logits2)[..., 0]``
    from the alignment branch (the XD-Violence AP protocol).  Each gets its own
    sceptic, because their score scales and null regimes are unrelated.

    The null regime of each branch is EMA-tracked from the *normal* videos of
    every batch -- weak supervision identifies them exactly, so no pseudo-labels
    are needed.
    """

    def __init__(self, weight1: float = 0.0, weight2: float = 0.0, **agg_kwargs):
        super().__init__()
        self.weight1 = float(weight1)
        self.weight2 = float(weight2)
        self.enabled = self.weight1 > 0 or self.weight2 > 0
        self.agg1 = BettingAggregator(**agg_kwargs)
        self.agg2 = BettingAggregator(**agg_kwargs)

    def forward(self, logits1, logits2, text_labels, lengths):
        """Returns ``(loss, stats)``; ``stats`` is empty when disabled."""
        if not self.enabled:
            return torch.zeros((), device=logits1.device), {}

        is_normal = text_labels[:, 0] > 0.5
        y = (1.0 - text_labels[:, 0]).detach()

        s1 = _to_bt(logits1)
        s2 = abnormal_margin(logits2)

        # The null regime is read off *this batch's* normal videos, with
        # gradient, so the statistic is affine-invariant.  This does not repair
        # the cross-bag null shift documented in docs/LIMITATIONS.md.
        if bool(is_normal.any()):
            nl = lengths[is_normal]
            c1, sc1 = self.agg1.batch_null(s1[is_normal], nl)
            c2, sc2 = self.agg2.batch_null(s2[is_normal], nl)
            self.agg1.update_null(s1[is_normal], nl)
            self.agg2.update_null(s2[is_normal], nl)
        else:  # all-anomalous batch: fall back to the EMA estimate
            c1 = sc1 = c2 = sc2 = None

        loss = torch.zeros((), device=logits1.device)
        stats = {}
        if self.weight1 > 0:
            b1 = betting_mil_loss(s1, y, lengths, self.agg1, c1, sc1)
            stats["bet1"] = float(b1.detach())
            loss = loss + self.weight1 * b1
        if self.weight2 > 0:
            b2 = betting_mil_loss(s2, y, lengths, self.agg2, c2, sc2)
            stats["bet2"] = float(b2.detach())
            loss = loss + self.weight2 * b2
        return loss, stats

    def log_evidence(self, logits1, logits2, text_labels, lengths):
        """Per-snippet log e-values for both branches, plus the binary label.

        Shares the batch-null standardisation with :meth:`forward`, so the
        bag-level sceptic and the snippet-level selector are constructed against
        the same estimated normal regime.  Sharing a reference does not imply
        empirical calibration; see ``docs/LIMITATIONS.md``.
        """
        is_normal = text_labels[:, 0] > 0.5
        y = (1.0 - text_labels[:, 0]).detach()
        s1 = _to_bt(logits1)
        s2 = abnormal_margin(logits2)
        if bool(is_normal.any()):
            nl = lengths[is_normal]
            c1, sc1 = self.agg1.batch_null(s1[is_normal], nl)
            c2, sc2 = self.agg2.batch_null(s2[is_normal], nl)
        else:
            c1 = sc1 = c2 = sc2 = None
        return (
            self.agg1.log_e_values(s1, c1, sc1),
            self.agg2.log_e_values(s2, c2, sc2),
            y,
        )

    def describe(self) -> str:
        return f"agg1[{self.agg1.extra_repr()}] agg2[{self.agg2.extra_repr()}]"
