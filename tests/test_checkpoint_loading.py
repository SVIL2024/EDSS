"""Checkpoint loading preserves frozen CLIP and validates detector weights."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.runner import load_checkpoint, trainable_state


def detector():
    torch.manual_seed(234)
    model = nn.Module()
    model.clipmodel = nn.Linear(3, 3)
    model.clipmodel.requires_grad_(False)
    model.classifier = nn.Linear(3, 1)
    return model


@pytest.mark.parametrize("compact", [False, True])
def test_saved_weights_load_and_reproduce_detector_output(tmp_path, compact):
    source, target = detector(), detector()
    with torch.no_grad():
        source.classifier.weight.add_(2)
        source.clipmodel.weight.add_(1)
    frozen_before = target.clipmodel.weight.detach().clone()
    state = trainable_state(source) if compact else source.state_dict()
    checkpoint = tmp_path / "weights.pth"
    torch.save(state, checkpoint)

    load_checkpoint(target, str(checkpoint))

    x = torch.tensor([[1.0, -2.0, 3.0]])
    torch.testing.assert_close(target.classifier(x), source.classifier(x))
    expected_clip = frozen_before if compact else source.clipmodel.weight
    torch.testing.assert_close(target.clipmodel.weight, expected_clip)


@pytest.mark.parametrize("invalid", ["missing", "unexpected", "shape"])
def test_incompatible_detector_weights_are_rejected(tmp_path, invalid):
    model = detector()
    state = trainable_state(model)
    if invalid == "missing":
        state.pop("classifier.weight")
    elif invalid == "unexpected":
        state["unknown.weight"] = torch.ones(1)
    else:
        state["classifier.weight"] = torch.ones(2, 3)
    checkpoint = tmp_path / "invalid.pth"
    torch.save(state, checkpoint)

    with pytest.raises(RuntimeError):
        load_checkpoint(model, str(checkpoint))


def test_missing_trainable_clip_parameter_is_rejected(tmp_path):
    model = detector()
    model.clipmodel.weight.requires_grad_(True)
    checkpoint = tmp_path / "invalid.pth"
    torch.save(trainable_state(model), checkpoint)
    with pytest.raises(RuntimeError, match="clipmodel.weight"):
        load_checkpoint(model, str(checkpoint))


@pytest.mark.parametrize("state", [{}, {"epoch": 10}])
def test_non_weight_checkpoint_is_rejected(tmp_path, state):
    checkpoint = tmp_path / "invalid.pth"
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="tensor state dictionary"):
        load_checkpoint(detector(), str(checkpoint))
