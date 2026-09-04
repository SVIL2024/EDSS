"""Classic auxiliary temporal priors for weakly-supervised VAD.

Not novel -- Sultani et al. (CVPR'18) introduced them -- but absent from
VadCLIP. Kept optional; weight zero removes only these terms.
"""

import torch

from .betting import _to_bt, length_mask

__all__ = ["smoothness_loss", "sparsity_loss"]


def smoothness_loss(scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean squared temporal difference of the snippet scores (total variation).

    Anomaly scores should vary smoothly between adjacent snippets.
    """
    s = _to_bt(scores)
    mask = length_mask(lengths, s.shape[1], s.device, s.dtype)
    pair_mask = mask[:, 1:] * mask[:, :-1]
    diff = (s[:, 1:] - s[:, :-1]) ** 2 * pair_mask
    per_video = diff.sum(1) / pair_mask.sum(1).clamp_min(1.0)
    return per_video.mean()


def sparsity_loss(scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """L1 prior: anomalous snippets are a small fraction of a video."""
    s = _to_bt(scores)
    mask = length_mask(lengths, s.shape[1], s.device, s.dtype)
    per_video = (s.abs() * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    return per_video.mean()
