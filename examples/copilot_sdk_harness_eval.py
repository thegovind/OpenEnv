#!/usr/bin/env python3
"""Experimental eval example: GitHub Copilot SDK as a black-box harness.

This example shows how to drive an OpenEnv environment with the GitHub Copilot
SDK (https://github.com/github/copilot-sdk). Copilot runs as an autonomous,
*black-box* agent: it owns its own planning and tool-calling loop, so this path
is evaluation-only. It is not suitable for white-box RL training, which needs
per-step token ids and logprobs (use ``MCPHarnessAdapter`` for that).

How the integration works:

- The environment exposes MCP-style tools through ``ResourceSession.list_tools()``.
- Each tool is registered with the Copilot session as a custom SDK tool whose
  handler calls back into ``ResourceSession.call_tool()``.
- The environment reward is read from the tool trace via the runtime's
  ``_resolve_env_reward`` helper and is **never** synthesized in the harness
  layer (an OpenEnv invariant).

For clarity this file ships a tiny self-contained arithmetic environment, so it
runs with no Docker, no HF Space, and no GPU. Swap ``ArithmeticSessionFactory``
for any ``ResourceSessionFactory`` (for example ``OpenSpielSessionFactory`` or
``BrowserGymSessionFactory``) to evaluate Copilot on a real OpenEnv gym.

Manual prerequisites:
- ``pip install github-copilot-sdk`` (provides the ``copilot`` package)
- GitHub Copilot CLI installed and authenticated (``copilot`` on PATH), or pass
  a token via ``--github-token`` / the ``GITHUB_TOKEN`` environment variable.

Run:
    python examples/copilot_sdk_harness_eval.py --episodes 3 --model gpt-5
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import operator
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    HarnessAdapter,
    HarnessRolloutResult,
    HarnessRunLimits,
    Message,
    ModelStep,
    ResourceSession,
    ResourceSessionFactory,
    RolloutEvent,
    ToolResult,
    ToolTraceEntry,
    VerifyResult,
    _resolve_env_reward,
)

# The GitHub Copilot SDK is an optional runtime dependency. Import it lazily so
# this module still imports (and the example is discoverable) without it.
try:
    from copilot import CopilotClient
    from copilot.session import PermissionHandler
    from copilot.session_events import AssistantMessageData, SessionIdleData
    from copilot.tools import Tool as CopilotTool
    from copilot.tools import ToolResult as CopilotToolResult

    _COPILOT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    CopilotClient = None  # type: ignore[assignment,misc]
    _COPILOT_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# A tiny self-contained environment (no Docker / HF Space / GPU required).
# ---------------------------------------------------------------------------

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(expression: str) -> float:
    """Evaluate a basic arithmetic expression without using ``eval``."""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            return _BINARY_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError("only numbers and + - * / % ** are allowed")

    return _eval(ast.parse(expression, mode="eval"))


@dataclass
class ArithmeticTask:
    """A single verifiable arithmetic word problem."""

    question: str
    answer: int


class ArithmeticSession(ResourceSession):
    """A minimal ``ResourceSession`` with two tools and a verifiable reward."""

    def __init__(self, task: ArithmeticTask) -> None:
        self._task = task
        self._reward: float | None = None
        self._submitted = False

    def initial_messages(self) -> list[Message]:
        return [{"role": "user", "content": self._task.question}]

    def list_tools(self) -> list[Tool]:
        return [
            Tool(
                name="compute",
                description=(
                    "Evaluate a basic arithmetic expression, "
                    "for example '12 * (3 + 4)'."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Expression using + - * / % ** and ().",
                        }
                    },
                    "required": ["expression"],
                },
            ),
            Tool(
                name="submit_answer",
                description="Submit the final integer answer. Ends the episode.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "integer",
                            "description": "The final integer answer.",
                        }
                    },
                    "required": ["value"],
                },
            ),
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name == "compute":
            return self._compute(str(arguments.get("expression", "")))
        if name == "submit_answer":
            return self._submit(arguments.get("value"))
        return ToolResult(error=f"unknown tool: {name}")

    def _compute(self, expression: str) -> ToolResult:
        try:
            value = _safe_eval(expression)
        except Exception as exc:  # noqa: BLE001 - report any parse/eval failure
            return ToolResult(error=f"could not evaluate {expression!r}: {exc}")
        return ToolResult(data={"result": value})

    def _submit(self, value: Any) -> ToolResult:
        try:
            correct = int(value) == self._task.answer
        except (TypeError, ValueError):
            return ToolResult(error="value must be an integer")
        self._reward = 1.0 if correct else 0.0
        self._submitted = True
        # The reward lives in the environment; the harness only forwards it.
        return ToolResult(
            data={"reward": self._reward, "correct": correct},
            done=True,
            metadata={"reward": self._reward},
        )

    def verify(
        self,
        transcript: list[Message],
        final_state: Any | None = None,
    ) -> VerifyResult:
        reward = 0.0 if self._reward is None else self._reward
        return VerifyResult(
            env_reward=reward,
            done=True,
            metrics={"submitted": self._submitted},
        )

    def close(self) -> None:
        return None


class ArithmeticSessionFactory(ResourceSessionFactory):
    """Produce one isolated ``ArithmeticSession`` per rollout."""

    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> ArithmeticSession:
        if not isinstance(task, ArithmeticTask):
            raise TypeError("task must be an ArithmeticTask")
        return ArithmeticSession(task)


# ---------------------------------------------------------------------------
# The Copilot SDK black-box harness adapter.
# ---------------------------------------------------------------------------


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _prompt_from_messages(messages: list[Message], max_turns: int) -> str:
    context = "\n\n".join(
        str(message.get("content", "")).strip()
        for message in messages
        if str(message.get("content", "")).strip()
    )
    return (
        "You are solving a task inside an OpenEnv environment.\n"
        "Use only the provided tools. Think step by step, use 'compute' as "
        "needed, then call 'submit_answer' exactly once with the final integer.\n"
        f"Take at most {max_turns} tool calls.\n\n"
        "Task:\n"
        f"{context}"
    )


class CopilotSDKHarnessAdapter(HarnessAdapter):
    """Drive an OpenEnv ``ResourceSession`` with the GitHub Copilot SDK.

    This is a black-box adapter: Copilot owns its own agent loop. The
    environment's tools are surfaced to Copilot as custom SDK tools, and the
    environment reward is recovered from the tool trace by the caller.
    """

    def __init__(self, *, model: str = "gpt-5", github_token: str | None = None):
        self._model = model
        self._github_token = github_token

    def run_white_box(
        self,
        model_step: ModelStep,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        raise NotImplementedError(
            "Copilot SDK is a black-box harness. Use run_black_box for "
            "evaluation, or MCPHarnessAdapter for white-box RL training."
        )

    def run_black_box(
        self,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        if CopilotClient is None:
            raise RuntimeError(
                "github-copilot-sdk is not installed. Install it with "
                "'pip install github-copilot-sdk'."
            ) from _COPILOT_IMPORT_ERROR
        run_limits = limits or HarnessRunLimits()
        return asyncio.run(self._run(session, run_limits))

    async def _run(
        self,
        session: ResourceSession,
        limits: HarnessRunLimits,
    ) -> HarnessRolloutResult:
        tool_trace: list[ToolTraceEntry] = []
        state = {"done": False}

        def make_handler(tool_name: str):
            async def handler(invocation: Any) -> Any:
                arguments = dict(getattr(invocation, "arguments", {}) or {})
                result = session.call_tool(tool_name, arguments)
                tool_trace.append(ToolTraceEntry(tool_name, arguments, result))
                if result.done:
                    state["done"] = True
                payload = result.error if result.error else _to_text(result.data)
                return CopilotToolResult(
                    text_result_for_llm=payload,
                    result_type="error" if result.error else "success",
                )

            return handler

        copilot_tools = [
            CopilotTool(
                name=spec.name,
                description=spec.description,
                parameters=spec.input_schema,
                handler=make_handler(spec.name),
            )
            for spec in session.list_tools()
        ]

        prompt = _prompt_from_messages(session.initial_messages(), limits.max_turns)
        assistant_chunks: list[str] = []
        idle = asyncio.Event()

        async with CopilotClient(github_token=self._github_token) as client:
            async with await client.create_session(
                model=self._model,
                tools=copilot_tools,
                on_permission_request=PermissionHandler.approve_all,
            ) as cop_session:

                def on_event(event: Any) -> None:
                    data = getattr(event, "data", None)
                    if isinstance(data, AssistantMessageData):
                        assistant_chunks.append(data.content)
                    elif isinstance(data, SessionIdleData):
                        idle.set()

                cop_session.on(on_event)
                await cop_session.send(prompt)
                await idle.wait()

        final_message = "\n".join(chunk for chunk in assistant_chunks if chunk)
        return HarnessRolloutResult(
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": final_message},
            ],
            tool_trace=tool_trace,
            events=[
                RolloutEvent(
                    type="copilot_session",
                    payload={"model": self._model, "tool_calls": len(tool_trace)},
                )
            ],
            done=state["done"],
            metrics={"mode": "black_box", "tool_calls": len(tool_trace)},
        )


# ---------------------------------------------------------------------------
# Episode runner + CLI.
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    episode_id: str
    reward: float
    done: bool
    tool_calls: int


def run_episode(
    *,
    session_factory: ResourceSessionFactory,
    adapter: CopilotSDKHarnessAdapter,
    task: ArithmeticTask,
    limits: HarnessRunLimits,
    episode_id: str,
) -> EpisodeResult:
    """Run one black-box episode and resolve the environment reward."""

    session = session_factory.create(task=task, episode_id=episode_id)
    try:
        rollout = adapter.run_black_box(session=session, limits=limits)
        verify = session.verify(rollout.messages)
        reward = _resolve_env_reward(rollout, verify)
        return EpisodeResult(
            episode_id=episode_id,
            reward=reward,
            done=rollout.done,
            tool_calls=len(rollout.tool_trace),
        )
    finally:
        session.close()


_DEFAULT_TASKS = [
    ArithmeticTask("A box holds 12 rows of 7 apples. How many apples?", 84),
    ArithmeticTask("Start with 250, subtract 4 lots of 18. What remains?", 178),
    ArithmeticTask("Three teams of 9 each score 4 points. Total points?", 108),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the GitHub Copilot SDK as a black-box OpenEnv harness.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=len(_DEFAULT_TASKS),
        help="Number of episodes to evaluate (cycles through the sample tasks).",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Model id for the Copilot session (for example gpt-5).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Maximum tool calls Copilot may take per episode.",
    )
    parser.add_argument(
        "--github-token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub token. Defaults to GITHUB_TOKEN or the logged-in user.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_factory = ArithmeticSessionFactory()
    adapter = CopilotSDKHarnessAdapter(
        model=args.model,
        github_token=args.github_token,
    )
    limits = HarnessRunLimits(max_turns=args.max_turns)

    results: list[EpisodeResult] = []
    for index in range(1, args.episodes + 1):
        task = _DEFAULT_TASKS[(index - 1) % len(_DEFAULT_TASKS)]
        episode = run_episode(
            session_factory=session_factory,
            adapter=adapter,
            task=task,
            limits=limits,
            episode_id=f"copilot-sdk-eval-{index}",
        )
        results.append(episode)
        print(
            f"[{episode.episode_id}] reward={episode.reward:.2f} "
            f"done={episode.done} tool_calls={episode.tool_calls} "
            f"| {task.question}"
        )

    if results:
        avg_reward = sum(r.reward for r in results) / len(results)
        success_rate = sum(1 for r in results if r.reward >= 1.0) / len(results)
        print(
            f"\nAggregate: episodes={len(results)} "
            f"avg_reward={avg_reward:.2f} success_rate={success_rate:.2%}"
        )


if __name__ == "__main__":
    main()
