# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Structured rubric results + human feedback — a tiny, dependency-free demo.

Runs the RFC 004 amendment end to end without any model or network:

1. A dimensioned grader returns a ``RubricResult`` (reward + per-dimension
   scores + a written rationale), instead of a bare float.
2. The scalar reward is still trivially recoverable for training.
3. A human reviews the same output and attaches ``HumanFeedback`` records
   (a label, a correction). These are *captured as data* here; fitting a
   calibrated grader from them is intentionally a follow-up (out of scope).

Run:  python examples/rubric_feedback_demo.py
"""

from __future__ import annotations

from typing import Any

from openenv.core.rubrics import (
    FeedbackKind,
    FeedbackTarget,
    HumanFeedback,
    Rubric,
    RubricResult,
    reward_value,
)


class AnswerQualityRubric(Rubric):
    """A deterministic, dimensioned grader for a short-answer task.

    Two criteria — did the answer cite a source, and is it concise — combine
    into one reward, and the rubric also emits per-dimension scores and a
    plain-language rationale a human (or an optimizer) can read.
    """

    def forward(self, action: Any, observation: Any) -> RubricResult:
        answer = str(action)
        cites_source = "[source]" in answer
        is_concise = len(answer.split()) <= 40

        dims = {
            "cites_source": 1.0 if cites_source else 0.0,
            "concise": 1.0 if is_concise else 0.0,
        }
        reward = sum(dims.values()) / len(dims)

        notes = []
        notes.append(
            "cites a source" if cites_source else "missing a [source] citation"
        )
        notes.append("concise" if is_concise else "too long (>40 words)")

        return RubricResult(
            reward=reward,
            feedback="; ".join(notes),
            dimensions=dims,
            confidence=1.0,  # deterministic grader
            metadata={"grader": "answer_quality", "version": 1},
        )


def main() -> None:
    rubric = AnswerQualityRubric()

    answer = "Paris is the capital of France."  # no [source], but concise
    result = rubric(action=answer, observation=None)

    print("=== Grader output (RubricResult) ===")
    print(f"answer       : {answer!r}")
    print(f"reward       : {reward_value(result):.2f}")  # scalar for training
    print(f"dimensions   : {result.dimensions}")
    print(f"feedback     : {result.feedback}")  # textual signal
    print(f"confidence   : {result.confidence}")
    print(f"metadata     : {result.metadata}")

    # A legacy consumer that only wants the number still works unchanged:
    scalar_reward = reward_value(result)
    assert 0.0 <= scalar_reward <= 1.0

    print("\n=== Human review (captured as HumanFeedback records) ===")
    review = [
        # A reviewer disagrees with full credit: the answer is fine but unsourced.
        HumanFeedback(
            kind=FeedbackKind.LABEL,
            target=FeedbackTarget.TRAJECTORY,
            value=0.5,
            reviewer_id="rater-7",
            target_id="demo-traj-1",
            comment="acceptable but should cite a source",
        ),
        # A targeted correction on a single dimension.
        HumanFeedback(
            kind=FeedbackKind.CORRECTION,
            target=FeedbackTarget.DIMENSION,
            dimension="cites_source",
            comment="treat an inline URL as a valid citation too",
            reviewer_id="rater-7",
            target_id="demo-traj-1",
        ),
    ]
    for fb in review:
        where = fb.dimension or fb.target.value
        print(f"- [{fb.kind.value}] on {where}: {fb.comment} (value={fb.value})")

    print(
        "\nThe grader's reward (0.5) and the human label (0.5) now sit on the same "
        "artifact. A later calibration step (separate PR) would fit the grader's "
        "thresholds/weights from records like these — without changing this "
        "interface."
    )


if __name__ == "__main__":
    main()
