# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ModalProvider. All tests mock the modal SDK."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake modal SDK module (so tests run without ``pip install modal``)
# ---------------------------------------------------------------------------
def _install_fake_modal():
    """Install a minimal fake ``modal`` package into sys.modules."""
    modal_mod = types.ModuleType("modal")

    class _FakeProcess:
        """Mimics modal ContainerProcess: .stdout.read() + .wait()."""

        def __init__(self, output: str = ""):
            self.stdout = MagicMock()
            self.stdout.read = MagicMock(return_value=output)

        def wait(self):
            return 0

    class _FakeTunnel:
        def __init__(self, url: str):
            self.url = url

    def _default_exec_output(cmd: str) -> str:
        if "test -f /app/env/openenv.yaml" in cmd:
            return "found"
        if cmd.startswith("cat /app/env/openenv.yaml"):
            return "spec_version: 1\nname: test\napp: server.app:app\nport: 8000\n"
        if "kill -0" in cmd:
            return "RUNNING"
        return ""

    class _FakeSandbox:
        # Records the (args, kwargs) of the most recent create call.
        last_create: tuple = ()
        last_create_v2: tuple = ()

        def __init__(self):
            self._exec_fn = _default_exec_output
            self.terminated = False
            self.exec_calls: list[str] = []

        def exec(self, *args, **kwargs):
            # Provider always invokes ("bash", "-c", <cmd>).
            cmd = args[2] if len(args) >= 3 else " ".join(args)
            self.exec_calls.append(cmd)
            return _FakeProcess(self._exec_fn(cmd))

        def tunnels(self, timeout: int = 50):
            return {8000: _FakeTunnel("https://sb-abc-8000.modal.host")}

        def terminate(self, *, wait: bool = False):
            self.terminated = True
            return 0

    class _FakeSandboxCls:
        @staticmethod
        def create(*args, **kwargs):
            sb = _FakeSandbox()
            _FakeSandboxCls.last_create = (args, kwargs)
            return sb

        @staticmethod
        def _experimental_create(*args, **kwargs):
            sb = _FakeSandbox()
            _FakeSandboxCls.last_create_v2 = (args, kwargs)
            return sb

    class _FakeApp:
        def __init__(self, name):
            self.name = name

    class _FakeAppCls:
        @staticmethod
        def lookup(name, *, create_if_missing=False, **kwargs):
            return _FakeApp(name)

    class _FakeImage:
        def __init__(self, kind, ref, **kwargs):
            self.kind = kind
            self.ref = ref
            self.kwargs = kwargs

        @staticmethod
        def from_dockerfile(path, **kwargs):
            return _FakeImage("dockerfile", str(path), **kwargs)

        @staticmethod
        def from_registry(tag, *args, **kwargs):
            return _FakeImage("registry", tag, **kwargs)

    modal_mod.Sandbox = _FakeSandboxCls
    modal_mod.App = _FakeAppCls
    modal_mod.Image = _FakeImage
    # Exposed for tests that need to introspect fakes.
    modal_mod._FakeSandbox = _FakeSandbox

    sys.modules["modal"] = modal_mod
    return modal_mod


_fake_modal = _install_fake_modal()

# Now safe to import the provider
from openenv.core.containers.runtime.modal_provider import ModalProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def provider():
    """Return a ModalProvider with default settings."""
    return ModalProvider(app_name="test-app")


@pytest.fixture(autouse=True)
def _fast_provider_sleep():
    """Avoid real sleeps in ModalProvider (wait_for_ready)."""
    with patch("openenv.core.containers.runtime.modal_provider.time.sleep"):
        yield


@pytest.fixture(autouse=True)
def _clean_dockerfile_registry():
    """Clear the Dockerfile registry between tests."""
    ModalProvider._dockerfile_registry.clear()
    yield
    ModalProvider._dockerfile_registry.clear()


def _write_dockerfile(tmp_path, body):
    server = tmp_path / "server"
    server.mkdir(exist_ok=True)
    df = server / "Dockerfile"
    df.write_text(body)
    return df


# ---------------------------------------------------------------------------
# Tests: use_sandbox_v2 flag
# ---------------------------------------------------------------------------
class TestSandboxV2Flag:
    def test_default_is_off(self):
        p = ModalProvider(app_name="a")
        assert p._use_sandbox_v2 is False

    def test_v2_emits_log(self, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            ModalProvider(app_name="a", use_sandbox_v2=True)
        assert any(
            "Sandbox v2" in r.message and "must be enabled" in r.message
            for r in caplog.records
        )

    def test_off_emits_no_log(self, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            ModalProvider(app_name="a")
        assert not any("Sandbox v2" in r.message for r in caplog.records)

    def test_v2_uses_experimental_create(self):
        p = ModalProvider(app_name="a", use_sandbox_v2=True)
        p.start_container("echo-env:latest")
        # The v2 path must have been taken.
        assert _fake_modal.Sandbox.last_create_v2
        args, kwargs = _fake_modal.Sandbox.last_create_v2
        assert args[:2] == ("sleep", "infinity")
        assert kwargs["encrypted_ports"] == [8000]

    def test_v1_uses_create(self):
        p = ModalProvider(app_name="a", use_sandbox_v2=False)
        p.start_container("echo-env:latest")
        args, kwargs = _fake_modal.Sandbox.last_create
        assert args[:2] == ("sleep", "infinity")
        assert kwargs["encrypted_ports"] == [8000]

    def test_v2_drops_memory(self):
        """Sandbox v2 does not accept a memory request."""
        p = ModalProvider(app_name="a", use_sandbox_v2=True, memory=2048)
        p.start_container("echo-env:latest")
        _, kwargs = _fake_modal.Sandbox.last_create_v2
        assert "memory" not in kwargs

    def test_v1_forwards_memory(self):
        p = ModalProvider(app_name="a", memory=2048)
        p.start_container("echo-env:latest")
        _, kwargs = _fake_modal.Sandbox.last_create
        assert kwargs["memory"] == 2048


# ---------------------------------------------------------------------------
# Tests: start_container
# ---------------------------------------------------------------------------
class TestStartContainer:
    def test_returns_tunnel_url(self, provider):
        url = provider.start_container("echo-env:latest")
        assert url == "https://sb-abc-8000.modal.host"

    def test_rejects_non_8000_port(self, provider):
        with pytest.raises(ValueError, match="only supports port 8000"):
            provider.start_container("echo-env:latest", port=9000)

    def test_port_8000_allowed(self, provider):
        url = provider.start_container("echo-env:latest", port=8000)
        assert url.startswith("https://")

    def test_registry_image_used(self, provider):
        provider.start_container("echo-env:latest")
        _, kwargs = _fake_modal.Sandbox.last_create
        assert kwargs["image"].kind == "registry"
        assert kwargs["image"].ref == "echo-env:latest"

    def test_env_vars_forwarded(self, provider):
        provider.start_container("echo-env:latest", env_vars={"FOO": "bar"})
        _, kwargs = _fake_modal.Sandbox.last_create
        assert kwargs["env"] == {"FOO": "bar"}

    def test_cpu_forwarded(self):
        p = ModalProvider(app_name="a", cpu=4.0)
        p.start_container("echo-env:latest")
        _, kwargs = _fake_modal.Sandbox.last_create
        assert kwargs["cpu"] == 4.0

    def test_server_started_in_background(self, provider):
        provider.start_container("echo-env:latest")
        assert any(
            "nohup" in c and "uvicorn server.app:app" in c
            for c in provider._sandbox.exec_calls
        )


# ---------------------------------------------------------------------------
# Tests: server command resolution
# ---------------------------------------------------------------------------
class TestServerCmd:
    def test_explicit_cmd_constructor(self):
        p = ModalProvider(app_name="a", cmd="python -m myserver")
        p.start_container("echo-env:latest")
        assert any("python -m myserver" in c for c in p._sandbox.exec_calls)

    def test_kwarg_cmd_overrides(self, provider):
        provider.start_container("echo-env:latest", cmd="custom-cmd")
        assert any("custom-cmd" in c for c in provider._sandbox.exec_calls)

    def test_discovers_from_yaml(self, provider):
        provider.start_container("echo-env:latest")
        assert any(
            "cd /app/env && python -m uvicorn server.app:app" in c
            for c in provider._sandbox.exec_calls
        )

    def test_no_yaml_raises(self, provider):
        def _no_yaml(cmd):
            return ""

        original = _fake_modal.Sandbox.create

        def patched(*args, **kwargs):
            sb = original(*args, **kwargs)
            sb._exec_fn = _no_yaml
            return sb

        with patch.object(_fake_modal.Sandbox, "create", staticmethod(patched)):
            with pytest.raises(ValueError, match="Could not find openenv.yaml"):
                provider.start_container("no-yaml:latest")

    def test_falls_back_to_dockerfile_cmd(self, provider, tmp_path):
        df = _write_dockerfile(
            tmp_path,
            'FROM python:3.11\nCOPY . /app\nCMD ["uvicorn", "fallback:app"]\n',
        )
        image = ModalProvider.image_from_dockerfile(str(df))

        def _no_yaml(cmd):
            return ""

        original = _fake_modal.Sandbox.create

        def patched(*args, **kwargs):
            sb = original(*args, **kwargs)
            sb._exec_fn = _no_yaml
            return sb

        with patch.object(_fake_modal.Sandbox, "create", staticmethod(patched)):
            provider.start_container(image)
        assert any("uvicorn fallback:app" in c for c in provider._sandbox.exec_calls)


# ---------------------------------------------------------------------------
# Tests: image_from_dockerfile
# ---------------------------------------------------------------------------
class TestImageFromDockerfile:
    def test_returns_dockerfile_uri(self, tmp_path):
        df = _write_dockerfile(tmp_path, "FROM python:3.11\nCMD uvicorn app:app\n")
        result = ModalProvider.image_from_dockerfile(str(df))
        assert result.startswith("dockerfile:")
        assert str(df.resolve()) in result

    def test_missing_dockerfile_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ModalProvider.image_from_dockerfile(str(tmp_path / "nope"))

    def test_bad_context_dir_raises(self, tmp_path):
        df = _write_dockerfile(tmp_path, "FROM python:3.11\n")
        with pytest.raises(ValueError, match="context_dir does not exist"):
            ModalProvider.image_from_dockerfile(str(df), context_dir="/no/such/dir")

    def test_parses_cmd_into_registry(self, tmp_path):
        df = _write_dockerfile(
            tmp_path, 'FROM python:3.11\nCMD ["uvicorn", "app:app"]\n'
        )
        ModalProvider.image_from_dockerfile(str(df))
        key = str(df.resolve())
        assert (
            ModalProvider._dockerfile_registry[key]["server_cmd"] == "uvicorn app:app"
        )

    def test_dockerfile_image_built_from_dockerfile(self, provider, tmp_path):
        df = _write_dockerfile(
            tmp_path, "FROM python:3.11\nCOPY . /app\nCMD uvicorn app:app\n"
        )
        image = ModalProvider.image_from_dockerfile(str(df))
        provider.start_container(image)
        _, kwargs = _fake_modal.Sandbox.last_create
        assert kwargs["image"].kind == "dockerfile"

    def test_unregistered_dockerfile_raises(self, provider):
        with pytest.raises(ValueError, match="No registered Dockerfile metadata"):
            provider.start_container("dockerfile:/tmp/never-registered")


# ---------------------------------------------------------------------------
# Tests: _parse_app_field
# ---------------------------------------------------------------------------
class TestParseAppField:
    def test_simple(self):
        assert ModalProvider._parse_app_field("app: server.app:app") == "server.app:app"

    def test_missing(self):
        assert ModalProvider._parse_app_field("name: test\nport: 8000") is None

    def test_invalid_yaml(self):
        assert ModalProvider._parse_app_field("::: not yaml :::") is None


# ---------------------------------------------------------------------------
# Tests: stop_container
# ---------------------------------------------------------------------------
class TestStopContainer:
    def test_terminates_sandbox(self, provider):
        provider.start_container("echo-env:latest")
        sb = provider._sandbox
        provider.stop_container()
        assert sb.terminated is True
        assert provider._sandbox is None

    def test_stop_without_start_is_noop(self, provider):
        provider.stop_container()  # should not raise


# ---------------------------------------------------------------------------
# Tests: wait_for_ready
# ---------------------------------------------------------------------------
class TestWaitForReady:
    def test_ready_when_health_200(self, provider):
        provider.start_container("echo-env:latest")
        with patch("requests.get", return_value=MagicMock(status_code=200)):
            provider.wait_for_ready("https://x.modal.host", timeout_s=5)

    def test_dead_process_raises(self, provider):
        import requests

        provider.start_container("echo-env:latest")
        provider._sandbox._exec_fn = lambda cmd: (
            "DEAD" if "kill -0" in cmd else "boom traceback"
        )
        with patch("requests.get", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(RuntimeError, match="Server process died"):
                provider.wait_for_ready("https://x.modal.host", timeout_s=5)

    def test_timeout_raises(self, provider):
        import requests

        provider.start_container("echo-env:latest")
        # Keep the process "RUNNING" so the death check doesn't short-circuit.
        provider._sandbox._exec_fn = lambda cmd: "RUNNING" if "kill -0" in cmd else ""
        with (
            patch("requests.get", side_effect=requests.ConnectionError("refused")),
            patch(
                "openenv.core.containers.runtime.modal_provider.time.time"
            ) as mock_time,
        ):
            mock_time.side_effect = [0, 1, 100]  # start, first loop, past deadline
            with pytest.raises(TimeoutError, match="did not become ready"):
                provider.wait_for_ready("https://x.modal.host", timeout_s=5)
