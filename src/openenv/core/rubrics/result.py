# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Structured rubric results.

`Rubric.forward()` historically returns a bare ``float`` reward. That is enough
to *train* against, but it discards everything an evaluator actually knows: a
written rationale, per-dimension scores, a confidence, or provenance. Those
signals are what human review, dimensioned/model-based scorers, and downstream
optimizers need.

`RubricResult` is a backward-compatible richer return type: a rubric may return
either a plain ``float`` (unchanged) or a `RubricResult` that carries the reward
plus optional feedback/dimensions/confidence/metadata. The framework always
normalizes to the scalar ``reward`` for reward computation, so existing float
rubrics keep working untouched.

See RFC 004 (amendment: "Structured Rubric Results & Human Feedback Records").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Union


@dataclass(frozen=True)
class RubricResult:
    """A reward plus the optional context an evaluator produced alongside it.

    Args:
        reward (`float`):
            The scalar reward, as today (typically 0.0-1.0). This is the only
            field training/reward computation requires.
        feedback (`str`, *optional*):
            Natural-language rationale or correction (e.g. an LLM judge's
            written explanation). Optimizers that consume textual feedback read
            this; it is ignored by scalar-only reward computation.
        dimensions (`Mapping[str, float]`, *optional*):
            Per-criterion sub-scores keyed by dimension name, for multi-criteria
            (dimensioned) graders.
        confidence (`float`, *optional*):
            The grader's confidence in this result (typically 0.0-1.0).
        metadata (`Mapping[str, Any]`, *optional*):
            Free-form provenance/extension data (grader id, model name,
            calibration artifact id, etc.). Kept open so the result shape does
            not need to change as new evaluators appear.
    """

    reward: float
    feedback: Optional[str] = None
    dimensions: Optional[Mapping[str, float]] = None
    confidence: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        """Coerce to the scalar reward so the result is drop-in where a float is expected."""
        return float(self.reward)


# A rubric may return a plain float (legacy) or a structured result (new).
RubricOutput = Union[float, RubricResult]


def reward_value(result: RubricOutput) -> float:
    """Extract the scalar reward from a rubric output (``float`` or `RubricResult`).

    This is the single place the framework collapses a (possibly structured)
    rubric output back to the number reward computation needs, so callers never
    have to branch on the return type.
    """
    if isinstance(result, RubricResult):
        return float(result.reward)
    return float(result)
