# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for structured rubric results and human feedback records.

Covers the RFC 004 amendment: ``RubricResult`` / ``reward_value``, the
``HumanFeedback`` record, backward-compatible handling of ``float`` vs
``RubricResult`` in the base class (sync and async), and ``LLMJudge``'s opt-in
feedback mode.
"""

from typing import Any

import pytest
from openenv.core.rubrics import (
    FeedbackKind,
    FeedbackTarget,
    Gate,
    HumanFeedback,
    LLMJudge,
    Rubric,
    RubricResult,
    Sequential,
    WeightedSum,
    reward_value,
)


class FloatRubric(Rubric):
    """Legacy rubric returning a bare float (must keep working unchanged)."""

    def forward(self, action: Any, observation: Any) -> float:
        return 0.8


class StructuredRubric(Rubric):
    """Rubric returning a structured result with feedback and dimensions."""

    def forward(self, action: Any, observation: Any) -> RubricResult:
        return RubricResult(
            reward=0.75,
            feedback="solid, but missed an edge case",
            dimensions={"correctness": 1.0, "style": 0.5},
            confidence=0.9,
            metadata={"grader": "demo"},
        )


class AsyncStructuredRubric(Rubric):
    """Async rubric returning a structured result."""

    async def forward(self, action: Any, observation: Any) -> RubricResult:
        return RubricResult(reward=0.6, feedback="ok")


class ConstStructured(Rubric):
    """Structured rubric returning a configurable reward (for container tests)."""

    def __init__(self, reward: float):
        super().__init__()
        self._reward = reward

    def forward(self, action: Any, observation: Any) -> RubricResult:
        return RubricResult(reward=self._reward, feedback=f"r={self._reward}")


class AsyncConstStructured(Rubric):
    """Async structured rubric returning a configurable reward."""

    def __init__(self, reward: float):
        super().__init__()
        self._reward = reward

    async def forward(self, action: Any, observation: Any) -> RubricResult:
        return RubricResult(reward=self._reward)


class TestRubricResult:
    def test_float_coercion_returns_reward(self):
        result = RubricResult(reward=0.42)
        assert float(result) == 0.42

    def test_defaults_are_empty(self):
        result = RubricResult(reward=1.0)
        assert result.feedback is None
        assert result.dimensions is None
        assert result.confidence is None
        assert result.metadata == {}

    def test_carries_optional_context(self):
        result = RubricResult(
            reward=0.5,
            feedback="why",
            dimensions={"a": 0.1},
            confidence=0.8,
            metadata={"grader": "x"},
        )
        assert result.feedback == "why"
        assert result.dimensions == {"a": 0.1}
        assert result.confidence == 0.8
        assert result.metadata == {"grader": "x"}

    def test_is_frozen(self):
        result = RubricResult(reward=0.5)
        with pytest.raises(Exception):
            result.reward = 0.9  # type: ignore[misc]


class TestRewardValue:
    def test_passthrough_float(self):
        assert reward_value(0.3) == 0.3

    def test_extracts_from_result(self):
        assert reward_value(RubricResult(reward=0.7, feedback="x")) == 0.7

    def test_coerces_int(self):
        assert reward_value(1) == 1.0


class TestBackwardCompatibility:
    def test_float_rubric_returns_float(self):
        rubric = FloatRubric()
        out = rubric("a", "o")
        assert out == 0.8
        assert isinstance(out, float)
        assert rubric.last_score == 0.8
        # last_result holds the same float for legacy rubrics.
        assert rubric.last_result == 0.8

    def test_structured_rubric_returns_result(self):
        rubric = StructuredRubric()
        out = rubric("a", "o")
        assert isinstance(out, RubricResult)
        assert out.feedback == "solid, but missed an edge case"
        # last_score is normalized to the scalar reward (backward compatible).
        assert rubric.last_score == 0.75
        # last_result preserves the full structured output.
        assert rubric.last_result is out
        assert rubric.last_result.dimensions == {"correctness": 1.0, "style": 0.5}

    @pytest.mark.asyncio
    async def test_async_structured_rubric(self):
        rubric = AsyncStructuredRubric()
        out = await rubric("a", "o")
        assert isinstance(out, RubricResult)
        assert out.reward == 0.6
        assert rubric.last_score == 0.6
        assert rubric.last_result is out

    def test_post_hook_sees_full_result(self):
        rubric = StructuredRubric()
        seen = {}

        def hook(r, action, observation, result):
            seen["result"] = result

        rubric.register_forward_hook(hook)
        rubric("a", "o")
        assert isinstance(seen["result"], RubricResult)
        assert seen["result"].feedback == "solid, but missed an edge case"


class TestContainerCoercion:
    """Containers aggregate the scalar reward of structured children (RFC 004
    amendment): a `RubricResult` child is coerced via ``reward_value`` so the
    aggregators never crash on the structured return type."""

    def test_weighted_sum_coerces_structured_child(self):
        rubric = WeightedSum([ConstStructured(0.6), FloatRubric()], [0.5, 0.5])
        assert rubric("a", "o") == pytest.approx(0.5 * 0.6 + 0.5 * 0.8)

    def test_gate_passes_structured_child_above_threshold(self):
        assert Gate(ConstStructured(0.75), threshold=0.5)("a", "o") == 0.75

    def test_gate_blocks_structured_child_below_threshold(self):
        assert Gate(ConstStructured(0.4), threshold=0.5)("a", "o") == 0.0

    def test_sequential_uses_structured_scalar(self):
        rubric = Sequential(FloatRubric(), ConstStructured(0.3))
        assert rubric("a", "o") == 0.3

    def test_sequential_short_circuits_on_zero_structured(self):
        rubric = Sequential(ConstStructured(0.0), FloatRubric())
        assert rubric("a", "o") == 0.0

    @pytest.mark.asyncio
    async def test_weighted_sum_async_structured_children(self):
        rubric = WeightedSum(
            [AsyncConstStructured(0.6), AsyncConstStructured(0.4)], [0.5, 0.5]
        )
        out = await rubric("a", "o")
        assert out == pytest.approx(0.5)


class TestHumanFeedback:
    def test_label_record(self):
        fb = HumanFeedback(
            kind=FeedbackKind.LABEL,
            target=FeedbackTarget.TRAJECTORY,
            value=1.0,
            reviewer_id="rater-7",
            target_id="traj-123",
        )
        assert fb.kind == FeedbackKind.LABEL
        assert fb.target == FeedbackTarget.TRAJECTORY
        assert fb.value == 1.0
        assert fb.metadata == {}

    def test_preference_record(self):
        fb = HumanFeedback(
            kind=FeedbackKind.PREFERENCE,
            value="traj-a",
            against="traj-b",
        )
        assert fb.value == "traj-a"
        assert fb.against == "traj-b"

    def test_dimension_correction(self):
        fb = HumanFeedback(
            kind=FeedbackKind.CORRECTION,
            target=FeedbackTarget.DIMENSION,
            dimension="style",
            comment="too terse",
        )
        assert fb.target == FeedbackTarget.DIMENSION
        assert fb.dimension == "style"
        assert fb.comment == "too terse"

    def test_enums_are_strings(self):
        # str-Enum members serialize to plain strings (JSON-friendly).
        assert FeedbackKind.LABEL == "label"
        assert FeedbackTarget.SCENARIO == "scenario"

    def test_is_frozen(self):
        fb = HumanFeedback(kind=FeedbackKind.LABEL)
        with pytest.raises(Exception):
            fb.value = 2.0  # type: ignore[misc]


class _FakeLLMClient:
    """Minimal LLMClient stub returning a canned response."""

    def __init__(self, response: str):
        self._response = response

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        return self._response


class TestLLMJudgeFeedback:
    @pytest.mark.asyncio
    async def test_default_returns_float(self):
        judge = LLMJudge(
            prompt_template="{action}",
            client=_FakeLLMClient("Score: 0.9 — looks good"),
        )
        out = await judge("a", "o")
        assert isinstance(out, float)
        assert out == 0.9

    @pytest.mark.asyncio
    async def test_feedback_mode_returns_result_with_rationale(self):
        response = "Score: 0.9 — clear and correct, minor style nit"
        judge = LLMJudge(
            prompt_template="{action}",
            client=_FakeLLMClient(response),
            return_feedback=True,
        )
        out = await judge("a", "o")
        assert isinstance(out, RubricResult)
        assert out.reward == 0.9
        assert out.feedback == response
        assert out.metadata == {"grader": "llm_judge"}
        # Scalar reward is still available for training via last_score.
        assert judge.last_score == 0.9

    def test_return_feedback_in_state_dict(self):
        judge = LLMJudge(
            prompt_template="{action}",
            client=_FakeLLMClient("0.5"),
            return_feedback=True,
        )
        assert judge.state_dict()["return_feedback"] is True
