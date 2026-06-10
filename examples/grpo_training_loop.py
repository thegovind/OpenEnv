#!/usr/bin/env python3
"""Self-contained GRPO-style training loop over an OpenEnv environment.

This example shows the *client side* of a Tinker / Loom-style training split:
your local loop owns the data, the rollouts, the reward, and the advantage
computation, while a (pluggable) trainer owns the weight update. The structure
is identical whether the policy is a three-parameter lookup table or a
trillion-parameter model, so this example uses a tiny tabular policy and runs
with no GPU, no API key, and no Docker. Running it prints a reward curve that
visibly climbs.

The loop is the canonical RL fine-tuning loop:

    sample a group of rollouts  ->  grade  ->  group-relative advantages
        ->  drop groups with no signal  ->  trainer.train_step  ->  repeat

It maps directly onto the ``finetuning-cookbook`` recipes and onto Tinker:

    cookbook ``do_group_rollout``          ~ sample_group() here
    cookbook ``remove_constant_reward_*``  ~ groups with std == 0 are skipped
    cookbook ``compute_advantages``        ~ group_advantages() here
    Tinker ``forward_backward`` + ``optim_step``  ~ Trainer.train_step()

Swap ``TabularPolicy`` + ``TabularPolicyGradientTrainer`` for a real model and a
real trainer (for example Azure AI Fine-Tuning Sessions / Tinker, where
``train_step`` would call ``forward_backward`` then ``optim_step``, or a TRL
``GRPOTrainer``) to train an actual model against the same OpenEnv environment.

Run:
    python examples/grpo_training_loop.py --steps 40 --group-size 8
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    Message,
    ResourceSession,
    ResourceSessionFactory,
    ToolResult,
    VerifyResult,
)


# ---------------------------------------------------------------------------
# A tiny single-step OpenEnv environment with a verifiable reward.
# ---------------------------------------------------------------------------


@dataclass
class MultipleChoiceTask:
    """One verifiable task: pick the index of the correct option."""

    task_id: str
    question: str
    options: list[int]
    correct_index: int


class MultipleChoiceSession(ResourceSession):
    """A one-step ``ResourceSession``: submit a choice, get a 1/0 reward."""

    def __init__(self, task: MultipleChoiceTask) -> None:
        self._task = task
        self._reward: float | None = None

    def initial_messages(self) -> list[Message]:
        options = ", ".join(f"{i}:{v}" for i, v in enumerate(self._task.options))
        return [
            {
                "role": "user",
                "content": f"{self._task.question} Options -> {options}",
            }
        ]

    def list_tools(self) -> list[Tool]:
        return [
            Tool(
                name="submit_answer",
                description="Submit the index of the chosen option. Ends the episode.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "choice": {
                            "type": "integer",
                            "description": "Index into the options list.",
                        }
                    },
                    "required": ["choice"],
                },
            )
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name != "submit_answer":
            return ToolResult(error=f"unknown tool: {name}")
        choice = arguments.get("choice")
        reward = 1.0 if choice == self._task.correct_index else 0.0
        self._reward = reward
        # Reward lives in the environment and rides on the tool result.
        return ToolResult(
            data={"reward": reward}, done=True, metadata={"reward": reward}
        )

    def verify(
        self,
        transcript: list[Message],
        final_state: Any | None = None,
    ) -> VerifyResult:
        return VerifyResult(env_reward=self._reward or 0.0, done=True)

    def close(self) -> None:
        return None


class MultipleChoiceSessionFactory(ResourceSessionFactory):
    """Produce one fresh ``MultipleChoiceSession`` per rollout."""

    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> MultipleChoiceSession:
        if not isinstance(task, MultipleChoiceTask):
            raise TypeError("task must be a MultipleChoiceTask")
        return MultipleChoiceSession(task)


# ---------------------------------------------------------------------------
# A tiny tabular policy + trainer (stand-ins for a model + Tinker/Loom service).
# ---------------------------------------------------------------------------


def _softmax(logits: list[float]) -> list[float]:
    hi = max(logits)
    exps = [math.exp(x - hi) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


class Policy(Protocol):
    def sample(self, task: MultipleChoiceTask) -> int: ...
    def greedy(self, task: MultipleChoiceTask) -> int: ...


@dataclass
class TabularPolicy:
    """Per-task softmax over option indices. Stands in for a model's weights."""

    rng: random.Random
    _logits: dict[str, list[float]] = field(default_factory=dict)

    def _row(self, task: MultipleChoiceTask) -> list[float]:
        # An untrained policy starts with an arbitrary prior (here: a mild
        # preference for the first option) so the held-out curve has somewhere
        # to climb from.
        if task.task_id not in self._logits:
            row = [0.0] * len(task.options)
            if row:
                row[0] = 2.5
            self._logits[task.task_id] = row
        return self._logits[task.task_id]

    def probs(self, task: MultipleChoiceTask) -> list[float]:
        return _softmax(self._row(task))

    def sample(self, task: MultipleChoiceTask) -> int:
        probs = self.probs(task)
        roll = self.rng.random()
        cumulative = 0.0
        for index, prob in enumerate(probs):
            cumulative += prob
            if roll <= cumulative:
                return index
        return len(probs) - 1

    def greedy(self, task: MultipleChoiceTask) -> int:
        row = self._row(task)
        return max(range(len(row)), key=lambda i: row[i])


@dataclass
class Sample:
    """One graded rollout, ready for the trainer."""

    task: MultipleChoiceTask
    action: int
    advantage: float


class Trainer(Protocol):
    def train_step(self, policy: Any, batch: list[Sample]) -> dict[str, float]: ...


@dataclass
class TabularPolicyGradientTrainer:
    """REINFORCE-with-baseline update on the tabular policy.

    The baseline is the group mean (that is what makes the advantages
    group-relative). For a real model this method would instead call a
    Tinker/Loom-style ``forward_backward`` then ``optim_step``.
    """

    lr: float = 0.5

    def train_step(
        self, policy: TabularPolicy, batch: list[Sample]
    ) -> dict[str, float]:
        for sample in batch:
            row = policy._row(sample.task)
            probs = _softmax(row)
            # d/dlogit_i  log pi(a) = 1[i == a] - p_i  (softmax score function)
            for i in range(len(row)):
                indicator = 1.0 if i == sample.action else 0.0
                row[i] += self.lr * sample.advantage * (indicator - probs[i])
        return {"updated": float(len(batch))}


# ---------------------------------------------------------------------------
# The training loop (the Tinker/Loom client side).
# ---------------------------------------------------------------------------


def group_advantages(rewards: list[float]) -> list[float] | None:
    """Group-relative advantages, or None if the group carries no signal."""

    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std == 0.0:
        # All rewards identical -> no contrast -> nothing to learn. Drop it.
        return None
    return [(r - mean) / std for r in rewards]


def sample_group(
    *,
    factory: ResourceSessionFactory,
    policy: Policy,
    task: MultipleChoiceTask,
    group_size: int,
) -> tuple[list[int], list[float]]:
    """Roll out ``group_size`` attempts at one task through the OpenEnv session."""

    actions: list[int] = []
    rewards: list[float] = []
    for _ in range(group_size):
        session = factory.create(task=task)
        try:
            action = policy.sample(task)
            result = session.call_tool("submit_answer", {"choice": action})
            actions.append(action)
            rewards.append(float(result.metadata.get("reward", 0.0)))
        finally:
            session.close()
    return actions, rewards


def evaluate(
    *,
    factory: ResourceSessionFactory,
    policy: Policy,
    tasks: list[MultipleChoiceTask],
) -> float:
    """Greedy (argmax) accuracy across tasks: the held-out signal you ship on."""

    correct = 0
    for task in tasks:
        session = factory.create(task=task)
        try:
            result = session.call_tool("submit_answer", {"choice": policy.greedy(task)})
            correct += int(result.metadata.get("reward", 0.0) >= 1.0)
        finally:
            session.close()
    return correct / len(tasks)


def train(
    *,
    factory: ResourceSessionFactory,
    policy: Policy,
    trainer: Trainer,
    tasks: list[MultipleChoiceTask],
    steps: int,
    group_size: int,
) -> None:
    for step in range(1, steps + 1):
        batch: list[Sample] = []
        step_rewards: list[float] = []
        dropped = 0
        for task in tasks:
            actions, rewards = sample_group(
                factory=factory,
                policy=policy,
                task=task,
                group_size=group_size,
            )
            step_rewards.extend(rewards)
            advantages = group_advantages(rewards)
            if advantages is None:
                dropped += 1
                continue
            batch.extend(
                Sample(task=task, action=a, advantage=adv)
                for a, adv in zip(actions, advantages)
            )

        trainer.train_step(policy, batch)

        train_reward = sum(step_rewards) / len(step_rewards)
        held_out = evaluate(factory=factory, policy=policy, tasks=tasks)
        bar = "#" * round(held_out * 20)
        print(
            f"step {step:3d} | train_reward={train_reward:.2f} "
            f"held_out_acc={held_out:.2f} dropped_groups={dropped} "
            f"|{bar:<20}|"
        )


_TASKS = [
    MultipleChoiceTask("t1", "12 * 7 = ?", [80, 84, 91, 77], 1),
    MultipleChoiceTask("t2", "250 - 4*18 = ?", [178, 182, 196, 168], 0),
    MultipleChoiceTask("t3", "3 * 9 * 4 = ?", [96, 108, 120, 99], 1),
    MultipleChoiceTask("t4", "144 / 12 = ?", [11, 12, 14, 10], 1),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRPO-style training loop over a self-contained OpenEnv env.",
    )
    parser.add_argument("--steps", type=int, default=40, help="Training steps.")
    parser.add_argument(
        "--group-size", type=int, default=8, help="Rollouts sampled per task per step."
    )
    parser.add_argument("--lr", type=float, default=0.2, help="Trainer learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    factory = MultipleChoiceSessionFactory()
    policy = TabularPolicy(rng=rng)
    trainer = TabularPolicyGradientTrainer(lr=args.lr)

    start = evaluate(factory=factory, policy=policy, tasks=_TASKS)
    print(f"held-out accuracy before training: {start:.2f}\n")
    train(
        factory=factory,
        policy=policy,
        trainer=trainer,
        tasks=_TASKS,
        steps=args.steps,
        group_size=args.group_size,
    )
    final = evaluate(factory=factory, policy=policy, tasks=_TASKS)
    print(f"\nheld-out accuracy after training: {final:.2f}")


if __name__ == "__main__":
    main()
