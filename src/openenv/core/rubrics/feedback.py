# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Human feedback records.

A `HumanFeedback` record captures a single act of human review over something
the gym produced — a step, a whole trajectory, a generated scenario, or one
rubric dimension. It is deliberately a *data record*, not a scoring mechanism:
this module standardizes how human judgements are represented and carried, so
later work (grader calibration, scenario-quality scoring for adversarial
designers, preference-based optimization) has one canonical shape to consume.

Capturing human feedback is the point of this type. Consuming it (fitting a
calibrated grader, building preference datasets) is intentionally out of scope
here and left to follow-up RFCs.

See RFC 004 (amendment: "Structured Rubric Results & Human Feedback Records").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Union


class FeedbackTarget(str, Enum):
    """What the reviewer was looking at."""

    STEP = "step"
    TRAJECTORY = "trajectory"
    SCENARIO = "scenario"
    DIMENSION = "dimension"


class FeedbackKind(str, Enum):
    """The shape of the judgement the reviewer expressed."""

    # A scalar or categorical label (use `value`).
    LABEL = "label"
    # A preference between two items (use `value` = preferred id, `against` = rejected id).
    PREFERENCE = "preference"
    # A free-text correction or edit (use `comment`).
    CORRECTION = "correction"


@dataclass(frozen=True)
class HumanFeedback:
    """One human judgement over a gym artifact.

    Args:
        kind (`FeedbackKind`):
            Whether this is a label, a preference, or a free-text correction.
        target (`FeedbackTarget`, *optional*, defaults to `TRAJECTORY`):
            What was reviewed (a step, trajectory, scenario, or single dimension).
        value (`float` | `str`, *optional*):
            For ``LABEL``: the label (e.g. ``1.0`` pass / ``0.0`` fail, or a
            category). For ``PREFERENCE``: the id of the preferred item.
        against (`str`, *optional*):
            For ``PREFERENCE``: the id of the rejected item.
        dimension (`str`, *optional*):
            When ``target == DIMENSION``, which rubric dimension this applies to.
        comment (`str`, *optional*):
            Free-text rationale or correction. Required in spirit for
            ``CORRECTION``.
        reviewer_id (`str`, *optional*):
            An opaque, non-PII identifier for the reviewer (for inter-rater
            analysis). Do not put names or emails here.
        target_id (`str`, *optional*):
            An id linking back to the reviewed artifact (e.g. a trajectory or
            step id) so feedback can be joined to what was scored.
        metadata (`Mapping[str, Any]`, *optional*):
            Free-form extension data (review source, timestamp, task type, ...).
    """

    kind: FeedbackKind
    target: FeedbackTarget = FeedbackTarget.TRAJECTORY
    value: Optional[Union[float, str]] = None
    against: Optional[str] = None
    dimension: Optional[str] = None
    comment: Optional[str] = None
    reviewer_id: Optional[str] = None
    target_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
