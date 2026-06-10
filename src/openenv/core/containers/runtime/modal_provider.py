# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Modal container provider for running OpenEnv environments in Modal sandboxes.

Requires the ``modal`` SDK: ``pip install modal>=1.4``

Supports both the stable Sandbox API (``modal.Sandbox.create``) and the beta
Sandbox v2 API (``modal.Sandbox._experimental_create``). Sandbox v2 is opt-in
via ``use_sandbox_v2=True`` because it is an experimental feature; see
https://modal.com/docs/guide/sandbox-v2 .
"""

from __future__ import annotations

import json
import logging
import shlex
import time
from typing import Any, Dict, Optional

import yaml

from .providers import ContainerProvider

logger = logging.getLogger(__name__)


class ModalProvider(ContainerProvider):
    """
    Container provider that runs environments in Modal sandboxes.

    Example:
        >>> provider = ModalProvider(app_name="openenv")
        >>> image = ModalProvider.image_from_dockerfile("envs/echo_env/server/Dockerfile")
        >>> base_url = provider.start_container(image)
        >>> provider.wait_for_ready(base_url)
        >>> provider.stop_container()

    Sandbox v2 (beta) is opt-in:
        >>> provider = ModalProvider(app_name="openenv", use_sandbox_v2=True)
    """

    _dockerfile_registry: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        *,
        app_name: str = "openenv",
        use_sandbox_v2: bool = False,
        cpu: Optional[float] = None,
        memory: Optional[int] = None,
        timeout: int = 300,
        cmd: Optional[str] = None,
    ):
        """
        Args:
            app_name (`str`, *optional*, defaults to `"openenv"`):
                Modal app name the sandbox is created under. Looked up (and
                created if missing) via ``modal.App.lookup``.
            use_sandbox_v2 (`bool`, *optional*, defaults to `False`):
                When `True`, sandboxes are created via the beta
                ``modal.Sandbox._experimental_create`` API (Sandbox v2). This
                is an experimental feature and is therefore off by default; see
                https://modal.com/docs/guide/sandbox-v2 .
            cpu (`float`, *optional*):
                Number of CPU cores to request for the sandbox.
            memory (`int`, *optional*):
                Memory in MiB to request. Ignored when `use_sandbox_v2` is
                `True` (Sandbox v2 does not accept a memory request).
            timeout (`int`, *optional*, defaults to `300`):
                Maximum sandbox lifetime in seconds.
            cmd (`str`, *optional*):
                Shell command to start the server inside the sandbox. When
                omitted, the command is auto-discovered from ``openenv.yaml``
                (falling back to the Dockerfile ``CMD``).
        """
        # Import eagerly so configuration errors surface at construction time.
        import modal

        self._modal = modal
        self._app_name = app_name
        self._use_sandbox_v2 = use_sandbox_v2
        self._cpu = cpu
        self._memory = memory
        self._timeout = timeout
        self._cmd = cmd
        self._sandbox: Any = None
        self._base_url: Optional[str] = None

        if use_sandbox_v2:
            logger.info(
                "Using Modal Sandbox v2 (experimental). This feature must be enabled."
            )

    def _exec_capture(self, cmd: str, timeout: int = 10) -> str:
        """Run *cmd* inside the sandbox and return its captured stdout."""
        proc = self._sandbox.exec("bash", "-c", cmd, timeout=timeout)
        try:
            out = proc.stdout.read()
        except Exception:
            out = ""
        proc.wait()
        return out or ""

    def _discover_server_cmd(self, port: int = 8000) -> str:
        """Discover the server command from ``openenv.yaml`` inside the sandbox.

        Finds the file, reads the ``app`` field, and constructs a command
        of the form ``cd <env_root> && python -m uvicorn <app> --host 0.0.0.0 --port <port>``.

        Raises:
            ValueError: If ``openenv.yaml`` is not found or lacks an ``app`` field.
        """
        yaml_path = self._find_openenv_yaml()
        if yaml_path is None:
            raise ValueError(
                "Could not find openenv.yaml inside the sandbox. "
                "Pass an explicit cmd= to ModalProvider or start_container()."
            )

        content = self._exec_capture(f"cat {shlex.quote(yaml_path)}")
        app = self._parse_app_field(content)
        if app is None:
            raise ValueError(
                f"openenv.yaml at {yaml_path} does not contain an 'app' field. "
                "Pass an explicit cmd= to ModalProvider or start_container()."
            )

        # The directory containing openenv.yaml is the env root
        env_root = yaml_path.rsplit("/", 1)[0]
        return (
            f"cd {shlex.quote(env_root)} && "
            f"python -m uvicorn {shlex.quote(app)} --host 0.0.0.0 --port {port}"
        )

    def _find_openenv_yaml(self) -> Optional[str]:
        """Locate ``openenv.yaml`` inside the sandbox.

        Tries the modern layout path ``/app/env/openenv.yaml`` first,
        then falls back to a ``find`` command for the old layout.
        """
        # Fast path: modern Dockerfile layout
        out = self._exec_capture("test -f /app/env/openenv.yaml && echo found")
        if "found" in (out or ""):
            return "/app/env/openenv.yaml"

        # Fallback: search for it (redirect stderr so error messages
        # like "No such file or directory" don't get mistaken for paths).
        path = self._exec_capture(
            "find /app -maxdepth 4 -name openenv.yaml -print -quit 2>/dev/null"
        ).strip()
        if path and path.startswith("/"):
            return path

        return None

    @staticmethod
    def _parse_app_field(yaml_content: str) -> Optional[str]:
        """Extract the ``app`` value from raw openenv.yaml content.

        Uses PyYAML to handle comments, quotes, and nested keys correctly.
        """
        try:
            data = yaml.safe_load(yaml_content) or {}
        except Exception:
            return None

        if not isinstance(data, dict):
            return None

        value = data.get("app")
        if isinstance(value, str):
            value = value.strip()
            return value if value else None
        return None

    @staticmethod
    def _parse_dockerfile_cmd(dockerfile_content: str) -> Optional[str]:
        """Extract the server command from the last ``CMD`` in a Dockerfile.

        Handles exec form (``CMD ["prog", "arg"]``) and shell form
        (``CMD prog arg``).  When a Dockerfile has multiple ``CMD``
        instructions (e.g. multi-stage builds), the last one wins - same
        semantics as Docker itself.  Lines where ``CMD`` appears inside a
        comment are ignored.

        Returns:
            The command as a single string, or ``None`` if no ``CMD`` found.
        """
        import re

        last_cmd: Optional[str] = None
        for line in dockerfile_content.splitlines():
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            match = re.match(r"CMD\s+(.+)", stripped, flags=re.IGNORECASE)
            if match:
                last_cmd = match.group(1).strip()

        if last_cmd is None:
            return None

        # Exec form: CMD ["executable", "param1", ...]
        if last_cmd.startswith("["):
            try:
                parts = json.loads(last_cmd)
                if isinstance(parts, list) and all(isinstance(p, str) for p in parts):
                    return " ".join(parts)
            except (json.JSONDecodeError, TypeError):
                pass

        # Shell form: CMD executable param1 ...
        return last_cmd if last_cmd else None

    @classmethod
    def image_from_dockerfile(
        cls,
        dockerfile_path: str,
        context_dir: str | None = None,
    ) -> str:
        """Validate a Dockerfile and return a ``dockerfile:`` URI for
        :meth:`start_container`.

        Eagerly validates the Dockerfile (existence, COPY sources) and stores
        its content in an internal registry.  The actual ``modal.Image`` is
        created later inside ``start_container``.

        Args:
            dockerfile_path (`str`):
                Path to the Dockerfile on disk.
            context_dir (`str`, *optional*):
                Build context directory.  Defaults to the Dockerfile's
                grandparent directory, matching the ``openenv init``
                convention where Dockerfiles live in
                ``<env>/server/Dockerfile`` and the build context is
                ``<env>/``.  Pass explicitly for non-standard layouts
                (e.g. ``context_dir="."`` for repo-root contexts).

        Returns:
            `str`: A ``"dockerfile:<abs_path>"`` string to pass to
            ``start_container``.

        Raises:
            FileNotFoundError: If *dockerfile_path* does not exist.
            ValueError: If *context_dir* is given but does not exist,
                or if COPY sources in the Dockerfile cannot be found
                under the resolved context directory.
        """
        import pathlib
        import re

        src = pathlib.Path(dockerfile_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

        if context_dir is not None:
            ctx = pathlib.Path(context_dir)
            if not ctx.is_dir():
                raise ValueError(f"context_dir does not exist: {context_dir}")
        else:
            # Default: grandparent of the Dockerfile, matching the
            # openenv init layout (<env>/server/Dockerfile -> <env>/).
            ctx = src.parent.parent

        content = src.read_text()

        # Validate that COPY sources exist under the context directory.
        # This catches mismatches early (e.g. a Dockerfile expecting repo
        # root as context when we defaulted to the env directory).
        for line in content.splitlines():
            m = re.match(r"^\s*COPY\s+(?!--from=)(\S+)\s+", line, re.IGNORECASE)
            if not m:
                continue
            copy_src = m.group(1)
            if copy_src.startswith("/"):
                continue
            resolved = ctx / copy_src
            if not resolved.exists() and not any(ctx.glob(copy_src)):
                raise ValueError(
                    f"Dockerfile COPY source '{copy_src}' not found "
                    f"under context_dir '{ctx}'. This Dockerfile may "
                    f"expect a different build context (e.g. the repo "
                    f"root). Pass context_dir explicitly."
                )

        # Parse CMD from the Dockerfile so start_container can use it as a
        # fallback when openenv.yaml is unavailable.
        parsed_cmd = cls._parse_dockerfile_cmd(content)

        cls._dockerfile_registry[str(src)] = {
            "dockerfile_path": str(src),
            "context_dir": str(ctx),
            "server_cmd": parsed_cmd,
        }

        return f"dockerfile:{src}"

    def _build_image(self, image: str) -> Any:
        """Build the ``modal.Image`` for *image* (registry tag or dockerfile:)."""
        modal = self._modal

        if image.startswith("dockerfile:"):
            dockerfile_path = image[len("dockerfile:") :]
            meta = self._dockerfile_registry.get(dockerfile_path)
            if meta is None:
                raise ValueError(
                    f"No registered Dockerfile metadata for {dockerfile_path}. "
                    "Call ModalProvider.image_from_dockerfile() first."
                )
            return modal.Image.from_dockerfile(
                meta["dockerfile_path"], context_dir=meta["context_dir"]
            )

        # Plain registry tag (e.g. "echo-env:latest").
        return modal.Image.from_registry(image)

    def start_container(
        self,
        image: str,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Create a Modal sandbox from a Docker image or Dockerfile.

        The sandbox is started with a keep-alive process and the server
        command is launched via ``exec`` afterwards, mirroring the discovery
        flow used by other cloud providers. The server command is resolved in
        order:

        1. Explicit ``cmd`` passed to the constructor.
        2. ``cmd`` key in ``**kwargs`` (popped before forwarding).
        3. Auto-discovered from ``openenv.yaml`` inside the sandbox.
        4. ``CMD`` parsed from the Dockerfile (when *image* came from
           ``image_from_dockerfile``).

        Args:
            image (`str`):
                Registry image tag (e.g. ``"echo-env:latest"``) or
                ``"dockerfile:<path>"`` returned by
                :meth:`image_from_dockerfile`.
            port (`int`, *optional*):
                Must be ``None`` or ``8000``. Modal exposes port 8000 via an
                encrypted tunnel; other ports raise ``ValueError``.
            env_vars (`dict`, *optional*):
                Environment variables forwarded to the sandbox.
            **kwargs:
                ``cmd`` (`str`) to override the server command.

        Returns:
            `str`: HTTPS tunnel URL for the sandbox (base_url).
        """
        if port is not None and port != 8000:
            raise ValueError(
                f"ModalProvider only supports port 8000 (got {port}). "
                "The Modal tunnel routes to port 8000 inside the sandbox."
            )

        # Resolve the server command (may be None; discovery happens after
        # sandbox creation when we can inspect the filesystem).
        cmd = kwargs.pop("cmd", None) or self._cmd

        # CMD parsed from Dockerfile (populated for "dockerfile:" images).
        parsed_cmd: Optional[str] = None
        if image.startswith("dockerfile:"):
            meta = self._dockerfile_registry.get(image[len("dockerfile:") :])
            if meta is not None:
                parsed_cmd = meta.get("server_cmd")

        modal = self._modal
        app = modal.App.lookup(self._app_name, create_if_missing=True)
        modal_image = self._build_image(image)

        # Common creation kwargs shared by both APIs.
        create_kwargs: Dict[str, Any] = {
            "app": app,
            "image": modal_image,
            "timeout": self._timeout,
            "encrypted_ports": [8000],
        }
        if env_vars:
            create_kwargs["env"] = dict(env_vars)
        if self._cpu is not None:
            create_kwargs["cpu"] = self._cpu

        if self._use_sandbox_v2:
            # Sandbox v2 (beta) does not accept a memory request.
            self._sandbox = modal.Sandbox._experimental_create(
                "sleep", "infinity", **create_kwargs
            )
        else:
            if self._memory is not None:
                create_kwargs["memory"] = self._memory
            self._sandbox = modal.Sandbox.create("sleep", "infinity", **create_kwargs)

        try:
            # Discover server command from openenv.yaml if not explicitly set.
            if cmd is None:
                try:
                    cmd = self._discover_server_cmd()
                except ValueError:
                    # Fall back to CMD parsed from Dockerfile (if available).
                    if parsed_cmd:
                        cmd = parsed_cmd
                    else:
                        raise

            # Launch the server in the background. Write the PID so we can
            # check whether the process crashed in wait_for_ready().
            escaped_cmd = shlex.quote(cmd)
            self._exec_capture(
                f"nohup bash -c {escaped_cmd} > /tmp/openenv-server.log 2>&1 &"
                " echo $! > /tmp/openenv-server.pid"
            )

            # Resolve the public tunnel URL for port 8000.
            tunnels = self._sandbox.tunnels()
            self._base_url = tunnels[8000].url
        except Exception:
            self.stop_container()
            raise

        return self._base_url

    def stop_container(self) -> None:
        """Terminate the Modal sandbox."""
        if self._sandbox is None:
            return

        try:
            self._sandbox.terminate()
        finally:
            self._sandbox = None
            self._base_url = None

    def wait_for_ready(self, base_url: str, timeout_s: float = 120.0) -> None:
        """
        Poll the /health endpoint until the sandbox is ready.

        Uses a longer default timeout (120s) than local Docker providers
        because Modal sandboxes may have cold-start latency.

        Args:
            base_url (`str`):
                Tunnel URL returned by ``start_container()``.
            timeout_s (`float`, *optional*, defaults to `120.0`):
                Maximum seconds to wait.

        Raises:
            TimeoutError: If the sandbox doesn't become ready in time.
            RuntimeError: If the server process died (detected via PID check).
        """
        import requests

        health_url = f"{base_url}/health"

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                response = requests.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass

            # Early exit: if the server process died, raise immediately
            # instead of waiting for the full health-check timeout.
            if self._sandbox is not None:
                out = self._exec_capture(
                    "kill -0 $(cat /tmp/openenv-server.pid) 2>/dev/null"
                    " && echo RUNNING || echo DEAD"
                )
                if "DEAD" in (out or ""):
                    log = self._exec_capture("cat /tmp/openenv-server.log 2>/dev/null")
                    raise RuntimeError(f"Server process died.\nLog:\n{log}")

            time.sleep(1.0)

        raise TimeoutError(
            f"Modal sandbox at {base_url} did not become ready within {timeout_s}s"
        )
