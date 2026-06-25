#!/usr/bin/env python3
"""Hello-world example running the Echo environment on Modal.

Boots the echo-env server inside a Modal sandbox via ``ModalProvider``, then
talks to it through ``EchoEnv`` over the encrypted Modal tunnel (https/wss).

Usage:
    pip install "modal>=1.4"
    modal setup  (one-time auth)
    PYTHONPATH=src:envs uv run python examples/modal_echo_env.py

Requires:
    A configured Modal account/token (see https://modal.com/docs/guide).
"""

import asyncio

from echo_env import EchoEnv
from openenv.core.containers.runtime.modal_provider import ModalProvider


async def _interact(base_url: str) -> None:
    """Run the async WebSocket interaction against the running sandbox."""
    async with EchoEnv(base_url=base_url) as env:
        await env.reset()

        tools = await env.list_tools()
        print("Available tools:", [t.name for t in tools])

        echoed = await env.call_tool("echo_message", message="Hello, World!")
        print("echo_message ->", echoed)


def main() -> int:
    image = ModalProvider.image_from_dockerfile("envs/echo_env/server/Dockerfile")

    # Provision the sandbox synchronously. The Modal SDK's blocking API warns
    # when driven from inside a running event loop, so the provider lifecycle is
    # kept out of asyncio; only the WebSocket client runs under asyncio.run().
    provider = ModalProvider(app_name="openenv-echo")
    base_url = provider.start_container(image)
    provider.wait_for_ready(base_url, timeout_s=180)
    try:
        asyncio.run(_interact(base_url))
    finally:
        provider.stop_container()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
