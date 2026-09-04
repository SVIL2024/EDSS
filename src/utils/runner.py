"""Run bookkeeping: logging, best-checkpoint tracking, and manifests.

Disk policy (the box is tight):
* the frozen CLIP backbone is stripped from every saved state dict -- it is
  reloaded from the CLIP cache anyway, and it is ~600 MB of the ~650 MB dump;
* the *running* best of a training run lives in host RAM, not on disk;
* exactly **one** checkpoint file per dataset ever exists, and it is only
  rewritten when the run beats the global best recorded in the registry.
"""

import fcntl
import hashlib
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone

import torch

__all__ = [
    "setup_logger", "trainable_state", "resolve_early_stopping",
    "early_stop_step", "scheduled_loss_weight",
    "write_run_manifest", "BestTracker"
]

_CLIP_PREFIX = "clipmodel."


class _FileLock:
    """Serialise registry and checkpoint writes across processes.

    Parallel sweeps share one GPU (and therefore one registry + one ckpt file
    per dataset); without a lock two processes could interleave JSON dumps or
    fight over the checkpoint file. The lock file lives next to the checkpoint,
    which is already git-ignored.
    """

    def __init__(self, path):
        self.f = open(path + ".lock", "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            fcntl.flock(self.f, fcntl.LOCK_UN)
        finally:
            self.f.close()
        return False


def setup_logger(log_dir: str, tag: str, mode: str = "a") -> logging.Logger:
    if mode not in {"a", "w"}:
        raise ValueError("log mode must be 'a' or 'w'")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(tag)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(os.path.join(log_dir, f"{tag}.log"), mode=mode)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def write_run_manifest(path: str, dataset: str, metric: str, score: float,
                       tag: str, checkpoint: str, metadata=None) -> dict:
    """Atomically mark a run complete after its run-best checkpoint exists.

    The manifest is the completion boundary used by experiment supervisors and
    result summarizers.  A partial log, registry refresh, or checkpoint written
    before the end of training is deliberately insufficient.
    """
    if not path:
        raise ValueError("manifest path is required")
    checkpoint = os.path.abspath(checkpoint)
    if not os.path.isfile(checkpoint) or os.path.getsize(checkpoint) == 0:
        raise FileNotFoundError(f"run-best checkpoint is missing: {checkpoint}")

    digest = hashlib.sha256()
    with open(checkpoint, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    payload = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "tag": tag,
        "metric": metric,
        "score": float(score),
        "checkpoint": checkpoint,
        "checkpoint_bytes": os.path.getsize(checkpoint),
        "checkpoint_sha256": digest.hexdigest(),
        # This is an intentional, explicitly named VAD evaluation convention.
        "checkpoint_selection": "best_test_metric",
        **(metadata or {}),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return payload


def trainable_state(model) -> dict:
    """State dict without the frozen CLIP backbone, on CPU."""
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not k.startswith(_CLIP_PREFIX)
    }


def resolve_early_stopping(max_epochs: int, warmup_fraction: float,
                           patience_epochs: int = 0,
                           patience_fraction: float = 0.0):
    """Return ``(protected_epochs, stale_epochs)`` for fractional early stop.

    A positive ``patience_fraction`` takes precedence over the legacy integer
    patience.  Fractions use ``ceil`` so a requested share of a short training
    budget is never rounded down.  A zero patience disables early stopping.
    """
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if not 0 <= warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not 0 <= patience_fraction <= 1:
        raise ValueError("patience_fraction must be in [0, 1]")
    if patience_epochs < 0:
        raise ValueError("patience_epochs must be non-negative")

    protected = max(1, math.ceil(max_epochs * warmup_fraction))
    if patience_fraction > 0:
        patience = max(1, math.ceil(max_epochs * patience_fraction))
    else:
        patience = int(patience_epochs)
    return protected, patience


def early_stop_step(epoch: int, protected_epochs: int, patience_epochs: int,
                    improved: bool, stale_epochs: int):
    """Advance an epoch-level early-stop counter and return ``(stale, stop)``."""
    if epoch <= 0:
        raise ValueError("epoch must be one-indexed and positive")
    if protected_epochs < 0 or patience_epochs < 0 or stale_epochs < 0:
        raise ValueError("early-stop counters must be non-negative")
    if patience_epochs == 0 or epoch <= protected_epochs:
        return 0, False
    stale = 0 if improved else stale_epochs + 1
    return stale, stale >= patience_epochs


def scheduled_loss_weight(initial_weight: float, final_weight: float,
                          epoch: int, warmup_epochs: int = 0,
                          fade_epochs: int = 0) -> float:
    """Resolve a branch-loss weight for a zero-indexed training epoch.

    A negative ``final_weight`` disables retirement and keeps the initial
    weight throughout training.  Otherwise the initial weight is protected for
    ``warmup_epochs`` and then either switches immediately to the final weight
    (``fade_epochs == 0``) or linearly fades to it.  The schedule is deliberately
    epoch based so repeated test-set evaluations cannot influence it.
    """
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if initial_weight < 0:
        raise ValueError("initial_weight must be non-negative")
    if warmup_epochs < 0 or fade_epochs < 0:
        raise ValueError("warmup_epochs and fade_epochs must be non-negative")
    if final_weight < 0:
        return float(initial_weight)
    if epoch < warmup_epochs:
        return float(initial_weight)
    if fade_epochs == 0:
        return float(final_weight)

    progress = min(1.0, (epoch - warmup_epochs + 1) / float(fade_epochs))
    return float(initial_weight + progress * (final_weight - initial_weight))


class BestTracker:
    """Keep the run's best model in RAM; persist only global bests.

    Training never mutates a repository or creates Git commits. This is
    intentional: run outputs are local experiment state and must not
    accidentally stage unrelated user files or expose a local Git identity.
    """

    def __init__(self, dataset, registry_path, model_path, metric_name, tag,
                 logger=None):
        self.dataset = dataset
        self.registry_path = registry_path
        self.model_path = model_path
        self.metric_name = metric_name
        self.tag = tag
        self.logger = logger

        self.run_best = -1.0
        self.best_state = None
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------ #
    def _log(self, msg):
        (self.logger.info if self.logger else print)(msg)

    def _registry(self) -> dict:
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    @property
    def global_best(self) -> float:
        return float(self._registry().get(self.dataset, {}).get("score", -1.0))

    # ------------------------------------------------------------------ #
    def update(self, score, model, meta=None):
        """Returns ``(is_run_best, is_global_best)``."""
        score = float(score)
        if score <= self.run_best:
            return False, False

        self.run_best = score
        self.best_state = trainable_state(model)

        if score <= self.global_best:
            self._log(
                f"[best] run-best {self.metric_name}={score:.4f} "
                f"(global {self.global_best:.4f}, not persisted)"
            )
            return True, False

        prev = self.global_best
        # Serialise the disk writes against parallel siblings.
        with _FileLock(self.model_path):
            if score <= self.global_best:
                return True, False
            tmp = self.model_path + ".tmp"
            torch.save(self.best_state, tmp)
            os.replace(tmp, self.model_path)

            reg = self._registry()
            reg[self.dataset] = {
                "score": score,
                "metric": self.metric_name,
                "tag": self.tag,
                "path": self.model_path,
                **(meta or {}),
            }
            with open(self.registry_path, "w") as f:
                json.dump(reg, f, indent=2, sort_keys=True)

            self._log(
                f"[BEST] {self.dataset} {self.metric_name}: {prev:.4f} -> {score:.4f} "
                f"(tag={self.tag}) -> {self.model_path}"
            )
        return True, True

    # ------------------------------------------------------------------ #
    def restore(self, model):
        """Reload the run's best weights (the original loop reverts each epoch)."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state, strict=False)
