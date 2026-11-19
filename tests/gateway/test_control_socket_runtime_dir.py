"""Locale the control socket off the shared folder (fork patch, v0.23.2-p9-14).

``bind(2)`` on an AF_UNIX node inside a VirtualBox shared folder answers ``EPERM``. On the
documented Docker layout ``HERMES_HOME=/opt/data`` is a vboxsf mount, and because that path still
fits ``sun_path`` upstream's over-long-home fallback never fired: the mandatory listener failed,
``start()`` returned False and bootstrap aborted with
``gateway session bootstrap control listener unavailable``. The fork walks an ordered candidate
list instead (private runtime dir -> temp dir -> /tmp -> legacy in-home leaf).

Every path here is built under ``mkdtemp`` in ``/tmp``: a deep ``tmp_path`` would blow the
``sun_path`` budget and silently exercise the wrong branch (same rule as
``test_loop_tick_socket_runtime_dir.py``).
"""

from __future__ import annotations

import asyncio
import errno
import shutil
import stat
import tempfile
from pathlib import Path

import pytest

from gateway.control_socket import (
    GatewayControlServer,
    _home_hash,
    control_socket_candidates,
    identify_gateway,
    resolve_client_socket_path,
)

pytestmark = pytest.mark.platforms("posix")


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def roots():
    """A short, disposable root plus ``home`` and a private ``xdg`` directory inside it."""
    try:
        root = Path(tempfile.mkdtemp(prefix="hgw-", dir="/tmp"))
    except OSError:
        pytest.skip("/tmp not writable on this host")
    home = root / "home" / ".hermes"
    home.mkdir(parents=True)
    xdg = root / "xdg"
    xdg.mkdir()
    xdg.chmod(0o700)  # what $XDG_RUNTIME_DIR promises: ours, no group/other access
    try:
        yield root, home, xdg
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _server(home: Path) -> GatewayControlServer:
    return GatewayControlServer(home, verb_handlers={"identify": lambda: {"pid": 4242}})


def _refuse(monkeypatch, *, predicate, err: int) -> list[str]:
    """``start_unix_server`` that refuses matching paths with ``err`` and otherwise delegates.

    This is what a vboxsf mount looks like from inside the process: the syscall — not a permission
    check — answers "I cannot host an AF_UNIX node here". Returns the paths that were attempted,
    in order.
    """
    real = asyncio.start_unix_server
    seen: list[str] = []

    async def fake(client_connected_cb, path=None, **kwargs):
        seen.append(str(path))
        if predicate(Path(path)):
            raise OSError(err, "Operation not permitted" if err == errno.EPERM else "unsupported")
        return await real(client_connected_cb, path=path, **kwargs)

    monkeypatch.setattr(asyncio, "start_unix_server", fake)
    return seen


def test_private_runtime_dir_is_the_primary_candidate(roots, monkeypatch):
    """1. With ``XDG_RUNTIME_DIR`` set the socket lands in ``hermes-gw-<hash>/`` there, never in
    ``HERMES_HOME`` — and the 0600 pointer in the home still names it."""
    _root, home, xdg = roots
    home = home.resolve()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg))
    stale = home / "gateway.sock"
    stale.touch()  # what a rolled-back pre-patch gateway leaves behind

    async def scenario():
        server = _server(home)
        assert await server.start()
        try:
            return server._bind_path, server._pointer_file, resolve_client_socket_path(home)
        finally:
            await server.stop()

    bind_path, pointer_file, resolved = _run(scenario())
    expected_dir = xdg / f"hermes-gw-{_home_hash(home)}"
    assert bind_path == expected_dir / "control.sock"
    assert bind_path.parent.parent == xdg
    # the stale in-home node must not shadow the pointer for clients
    assert not stale.exists()
    assert not (home / "gateway.sock").exists()
    assert resolved == bind_path
    assert pointer_file == home / "gateway.sock.path"
    # stop() removed both artefacts, so a later gateway for the same home starts clean
    assert not bind_path.exists() and not pointer_file.exists()


def test_without_xdg_runtime_dir_falls_back_to_tempdir(roots, monkeypatch):
    """2. No usable ``XDG_RUNTIME_DIR`` -> the temp-dir candidate, still shaped ``hermes-gw-<hash>``."""
    root, home, _xdg = roots
    home = home.resolve()
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    fake_tmp = root / "tmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(fake_tmp))  # what gettempdir() will answer

    async def scenario():
        server = _server(home)
        assert await server.start()
        try:
            return server._bind_path
        finally:
            await server.stop()

    bind_path = _run(scenario())
    assert bind_path == fake_tmp / f"hermes-gw-{_home_hash(home)}" / "control.sock"
    assert not (home / "gateway.sock").exists()


def test_eperm_on_the_first_candidate_moves_to_the_next(roots, monkeypatch):
    """3. vboxsf answers EPERM for the runtime candidate -> the walk continues and the listener is
    up where it can answer (the real cutover failure replayed with no shared folder needed)."""
    _root, home, xdg = roots
    home = home.resolve()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg))
    seen = _refuse(monkeypatch, predicate=lambda p: p.is_relative_to(xdg), err=errno.EPERM)

    async def scenario():
        server = _server(home)
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(None, lambda: identify_gateway(home))
            return server._bind_path, answer
        finally:
            await server.stop()

    bind_path, answer = _run(scenario())
    assert answer == {"pid": 4242}  # a client resolves it through the pointer
    assert seen[0].startswith(str(xdg))  # the refused candidate was tried first
    assert not bind_path.is_relative_to(xdg)
    assert bind_path.name == "control.sock"


def test_every_candidate_refusing_keeps_bootstrap_fatal(roots, monkeypatch):
    """4. The invariant is NOT weakened: when no filesystem accepts a socket every candidate is
    tried, ``start()`` returns False — the exact input ``run_bootstrap`` turns into the fatal
    ``RuntimeError`` (gateway/run_bootstrap.py) — and the refusal errno is preserved."""
    _root, home, xdg = roots
    home = home.resolve()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg))
    seen = _refuse(monkeypatch, predicate=lambda _p: True, err=errno.EPERM)

    server = _server(home)

    async def start_then_report():
        started = await server.start()
        try:
            await server._bind_first_available()
        except OSError as exc:
            return started, exc.errno
        raise AssertionError("the walk must raise when every candidate refuses")

    started, err = _run(start_then_report())
    assert started is False
    assert err == errno.EPERM
    candidates = [str(bind) for bind, _pointer in control_socket_candidates(home)]
    assert len(candidates) >= 3  # runtime dir + temp dir + legacy in-home leaf
    assert seen[: len(candidates)] == candidates  # every candidate was tried, best first
    assert len(seen) == 2 * len(candidates)  # start() walked once, the direct call walked again
    assert not (home / "gateway.sock.path").exists()


def test_pointer_is_published_0600_and_reads_back(roots, monkeypatch):
    """5. The discovery pointer is atomic, owner-only and round-trips to the real listener."""
    _root, home, xdg = roots
    home = home.resolve()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg))

    async def scenario():
        server = _server(home)
        assert await server.start()
        try:
            pointer = home / "gateway.sock.path"
            target = Path(pointer.read_text(encoding="utf-8").strip())
            return (stat.S_IMODE(pointer.lstat().st_mode), target,
                    stat.S_IMODE(target.parent.lstat().st_mode),
                    stat.S_IMODE(target.lstat().st_mode),
                    resolve_client_socket_path(home))
        finally:
            await server.stop()

    pointer_mode, target, dir_mode, socket_mode, resolved = _run(scenario())
    assert pointer_mode == 0o600
    assert dir_mode == 0o700
    assert socket_mode == 0o600
    assert target == xdg / f"hermes-gw-{_home_hash(home)}" / "control.sock"
    assert resolved == target
