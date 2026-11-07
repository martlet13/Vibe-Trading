"""Tests for the API server bind default and non-loopback warning.

Covers the secure-by-default behavior added for #333:
  - `_is_loopback_bind_host` classification (IPv4 / IPv6 / hostname / edge)
  - `serve_main` defaults the bind address to loopback (127.0.0.1)
  - binding a non-loopback address without API_AUTH_KEY emits a startup warning,
    while loopback or a configured key stays quiet

Warning assertions match the bind warning's own text rather than the bare
``[warn]`` prefix, so an unrelated startup warning (e.g. a missing frontend
build in CI) cannot satisfy or break them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import api_server

# Unique substring of the non-loopback bind warning (api_server.py:3513).
_BIND_WARN = "without API_AUTH_KEY set"


@pytest.mark.unit
@pytest.mark.parametrize(
    "host, expected",
    [
        ("127.0.0.1", True),
        ("127.0.0.2", True),
        ("::1", True),
        ("0:0:0:0:0:0:0:1", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("::", False),
        ("192.168.1.5", False),
        ("", False),
    ],
)
def test_is_loopback_bind_host(host: str, expected: bool) -> None:
    assert api_server._is_loopback_bind_host(host) is expected


def _run_serve(argv: list[str]) -> str | None:
    """Invoke serve_main with uvicorn stubbed; return the host it bound to.

    The frontend mount / static-file branches are short-circuited because
    uvicorn.run raises SystemExit before reaching the server loop.
    """
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> None:
        captured["host"] = kwargs.get("host") or (args[1] if len(args) > 1 else None)
        raise SystemExit(0)

    with mock.patch("uvicorn.run", fake_run):
        try:
            api_server.serve_main(argv)
        except SystemExit:
            pass
    return captured.get("host")  # type: ignore[return-value]


@pytest.mark.unit
def test_serve_defaults_to_loopback() -> None:
    assert _run_serve([]) == "127.0.0.1"


@pytest.mark.unit
def test_serve_honors_explicit_host() -> None:
    assert _run_serve(["--host", "0.0.0.0"]) == "0.0.0.0"


@pytest.mark.unit
def test_non_loopback_without_key_warns(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("API_AUTH_KEY", raising=False)
    monkeypatch.setattr(api_server, "_API_KEY", None, raising=False)

    _run_serve(["--host", "0.0.0.0"])

    out = capsys.readouterr().out
    assert _BIND_WARN in out


@pytest.mark.unit
def test_loopback_does_not_warn(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("API_AUTH_KEY", raising=False)
    monkeypatch.setattr(api_server, "_API_KEY", None, raising=False)

    _run_serve(["--host", "127.0.0.1"])

    out = capsys.readouterr().out
    assert _BIND_WARN not in out


@pytest.mark.unit
def test_non_loopback_with_key_does_not_warn(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("API_AUTH_KEY", "secret")
    monkeypatch.setattr(api_server, "_API_KEY", "secret", raising=False)

    _run_serve(["--host", "0.0.0.0"])

    out = capsys.readouterr().out
    assert _BIND_WARN not in out


@pytest.mark.unit
def test_serve_mounts_frontend_when_routes_include_router_without_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_api_file = tmp_path / "agent" / "api_server.py"
    fake_dist = tmp_path / "frontend" / "dist"
    fake_dist.mkdir(parents=True)
    fake_api_file.parent.mkdir(parents=True)
    fake_api_file.write_text("# test module path\n", encoding="utf-8")
    (fake_dist / "index.html").write_text("<div>Vibe-Trading</div>\n", encoding="utf-8")

    monkeypatch.setattr(api_server, "__file__", str(fake_api_file))
    api_server.app.routes.insert(0, SimpleNamespace())
    try:
        assert _run_serve([]) == "127.0.0.1"
    finally:
        api_server.app.routes.pop(0)


@pytest.mark.unit
def test_direct_script_run_registers_routes() -> None:
    """`python agent/api_server.py` must not fail on the route-module host lookup.

    Run as a script, the module lives in sys.modules only as ``__main__``;
    register_runs_routes used to raise before argparse could even print --help.
    """
    script = Path(api_server.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parent.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "--port" in result.stdout
