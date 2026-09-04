"""Focused regression tests for the original-MIL retirement ablation."""
import os
import sys

import pytest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils.runner import scheduled_loss_weight  # noqa: E402


def test_negative_final_weight_preserves_the_original_objective():
    values = [scheduled_loss_weight(1.0, -1.0, epoch, 2, 3)
              for epoch in range(8)]
    assert values == [1.0] * 8


def test_constant_half_weight_does_not_need_a_schedule():
    values = [scheduled_loss_weight(0.5, -1.0, epoch)
              for epoch in range(4)]
    assert values == [0.5] * 4


def test_immediate_retirement_happens_only_after_warmup():
    values = [scheduled_loss_weight(1.0, 0.0, epoch, 2, 0)
              for epoch in range(5)]
    assert values == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_linear_retirement_reaches_zero_and_stays_there():
    values = [scheduled_loss_weight(1.0, 0.0, epoch, 2, 2)
              for epoch in range(6)]
    assert values == pytest.approx([1.0, 1.0, 0.5, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "args",
    [
        (-0.1, -1.0, 0, 0, 0),
        (1.0, 0.0, -1, 0, 0),
        (1.0, 0.0, 0, -1, 0),
        (1.0, 0.0, 0, 0, -1),
    ],
)
def test_invalid_schedule_inputs_are_rejected(args):
    with pytest.raises(ValueError):
        scheduled_loss_weight(*args)
