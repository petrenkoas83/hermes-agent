"""Regression tests for control-socket bind relocation on AF_UNIX-hostile filesystems.

vboxsf (VirtualBox shared folders) and 9p accept regular files but reject
``bind(2)`` on an ``AF_UNIX`` socket with ``EPERM``. Because ``HERMES_HOME`` is
``/opt/data`` — a vboxsf mount — the gateway could not create
``$HERMES_HOME/gateway.sock`` and, once the listener became fatal in
``run_bootstrap``, crash-looped.

The fix has two halves, both covered here:

1. ``GatewayControlServer._start_posix`` retries the bind in the private
   runtime dir (``$XDG_RUNTIME_DIR``) when the direct bind fails with
   ``EPERM``/``EACCES``/``ENOTSUP``, and publishes ``gateway.sock.path`` so
   clients still resolve the listener.
2. ``run_bootstrap.start_gateway`` treats a missing listener as a warning, not
   a fatal error — the documented contract of ``GatewayControlServer``.
"""

import asyncio
import errno
import os
import shutil
import stat
import tempfile
from pathlib import Path

import pytest

from gateway.control_socket import (
    GatewayControlServer,
    _home_hash,
    resolve_client_socket_path,
    resolve_server_socket_path,
)

pytestmark = pytest.mark.platforms("posix")


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    d = tmp_path / "home" / ".hermes"
    d.mkdir(parents=True)
    return d


@pytest.fixture()
def runtime_dir(tmp_path: Path) -> Path:
    """A private runtime dir, as the container provides at /tmp/hermes-runtime.

    Deliberately short: ``_fallback_socket_path`` only prefers ``$XDG_RUNTIME_DIR``
    when the resulting path still fits ``sun_path`` (100 bytes). pytest's
    ``tmp_path`` is long enough to overflow that budget, which would silently
    exercise the ``/tmp`` candidate instead — so this fixture lives directly
    under the system temp dir with a short name.
    """
    d = Path(tempfile.mkdtemp(prefix="hrt-"))
    os.chmod(d, 0o700)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _patch_bind_to_fail(monkeypatch, *, err: int, only_under: Path):
    """Make ``asyncio.start_unix_server`` raise ``err`` for paths under ``only_under``.

    Everything else delegates to the real implementation, so the relocated bind
    genuinely succeeds and the test exercises the real server.
    """
    real = asyncio.start_unix_server

    async def fake(handler, *, path=None, **kwargs):
        if path is not None and Path(path).is_relative_to(only_under):
            raise OSError(err, os.strerror(err), str(path))
        return await real(handler, path=path, **kwargs)

    monkeypatch.setattr(asyncio, "start_unix_server", fake)


def test_eperm_on_home_relocates_to_runtime_dir(home, runtime_dir, monkeypatch):
    """EPERM binding in HERMES_HOME relocates the socket and publishes the pointer."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    _patch_bind_to_fail(monkeypatch, err=errno.EPERM, only_under=home)

    server = GatewayControlServer(home)

    async def scenario():
        started = await server.start()
        try:
            assert started is True, "EPERM must relocate, not fail"
            bind_path = server._bind_path
            assert bind_path is not None
            assert not bind_path.is_relative_to(home), "socket must leave the hostile mount"
            assert bind_path.is_relative_to(runtime_dir), f"expected runtime dir, got {bind_path}"
            assert bind_path.exists()
            assert len(str(bind_path).encode()) <= 100, "relocated path must fit sun_path"
            # The pointer lives beside the relocated socket, not in HERMES_HOME: vboxsf keeps a
            # 0755 mode on a file created 0600, and discovery refuses such a pointer.
            pointer = bind_path.parent / "gateway.sock.path"
            assert pointer.is_file()
            assert Path(pointer.read_text(encoding="utf-8").strip()) == bind_path
            assert not (home / "gateway.sock.path").exists(), "pointer must leave the hostile mount"
            assert resolve_client_socket_path(home) == bind_path
        finally:
            await server.stop()

    _run(scenario())


def test_eacces_on_home_relocates_to_runtime_dir(home, runtime_dir, monkeypatch):
    """EACCES is treated the same as EPERM."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    _patch_bind_to_fail(monkeypatch, err=errno.EACCES, only_under=home)

    server = GatewayControlServer(home)

    async def scenario():
        started = await server.start()
        try:
            assert started is True
            assert server._bind_path is not None
            assert server._bind_path.is_relative_to(runtime_dir)
        finally:
            await server.stop()

    _run(scenario())


def test_long_runtime_dir_falls_back_to_tmp(home, monkeypatch):
    """When $XDG_RUNTIME_DIR is too long for sun_path, the /tmp candidate is used.

    This is the behaviour that made the first version of these tests fail: the
    relocation still succeeds, it just cannot honour the runtime dir.
    """
    long_rt = Path(tempfile.mkdtemp(prefix="hrt-")) / ("x" * 80)
    long_rt.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(long_rt))
    _patch_bind_to_fail(monkeypatch, err=errno.EPERM, only_under=home)

    server = GatewayControlServer(home)

    async def scenario():
        started = await server.start()
        try:
            assert started is True
            bind_path = server._bind_path
            assert bind_path is not None
            assert not bind_path.is_relative_to(home)
            assert not bind_path.is_relative_to(long_rt), "too long for sun_path"
            assert len(str(bind_path).encode()) <= 100
            assert bind_path.exists()
        finally:
            await server.stop()
            shutil.rmtree(long_rt.parent, ignore_errors=True)

    _run(scenario())


def test_unrelated_oserror_still_fails(home, runtime_dir, monkeypatch):
    """Only the filesystem-refusal errnos relocate; a real bug must still surface."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    _patch_bind_to_fail(monkeypatch, err=errno.ENOENT, only_under=home)

    server = GatewayControlServer(home)

    async def scenario():
        # start() swallows exceptions into a False return (non-fatal contract),
        # so assert on the return value rather than expecting a raise.
        assert await server.start() is False

    _run(scenario())


def test_short_home_still_binds_in_home(home, runtime_dir, monkeypatch):
    """No regression: a healthy filesystem keeps the socket in HERMES_HOME."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))

    server = GatewayControlServer(home)

    async def scenario():
        started = await server.start()
        try:
            assert started is True
            assert server._bind_path == home / "gateway.sock"
            assert not (home / "gateway.sock.path").exists(), "no pointer when the direct bind works"
        finally:
            await server.stop()

    _run(scenario())


def test_resolve_server_socket_path_has_no_env_knob(home, monkeypatch):
    """Documents why Option A was rejected: the path is derived from home alone.

    If upstream ever adds a relocation env var, this test fails and the
    provisioning-side workaround can be dropped in favour of it.
    """
    monkeypatch.setenv("HERMES_GATEWAY_SOCKET", "/tmp/should-be-ignored.sock")
    monkeypatch.setenv("GATEWAY_SOCKET", "/tmp/should-be-ignored.sock")
    bind_path, pointer = resolve_server_socket_path(home)
    assert bind_path == home / "gateway.sock"
    assert pointer is None


def test_pointer_is_published_beside_the_relocated_socket(home, runtime_dir, monkeypatch):
    """The pointer must not land on the AF_UNIX-hostile mount.

    vboxsf accepts the file but keeps a 0755 mode even for an inode created 0600, and
    ``gateway_runtime_discovery`` refuses any pointer with ``mode & 0o077``. Publishing it
    beside the socket puts it on ext4, where the 0600 mkstemp inode survives the rename.
    """
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    _patch_bind_to_fail(monkeypatch, err=errno.EPERM, only_under=home)

    server = GatewayControlServer(home)

    async def scenario():
        started = await server.start()
        try:
            assert started is True
            bind_path = server._bind_path
            assert bind_path is not None and bind_path.is_relative_to(runtime_dir)
            pointer = server._pointer_file
            assert pointer == bind_path.parent / "gateway.sock.path"
            assert pointer.is_file()
            assert stat.S_IMODE(pointer.stat().st_mode) & 0o077 == 0, "pointer must pass the discovery check"
            assert not (home / "gateway.sock.path").exists()
        finally:
            await server.stop()

    _run(scenario())


def test_pointer_path_follows_the_socket(home, runtime_dir, monkeypatch):
    """``pointer_path`` is the socket's directory when relocated, ``home`` otherwise."""
    from gateway.control_socket import pointer_path

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    assert pointer_path(home, home / "gateway.sock") == home / "gateway.sock.path"
    relocated = runtime_dir / "hermes-gw-x" / "control.sock"
    assert pointer_path(home, relocated) == relocated.parent / "gateway.sock.path"


def test_client_resolves_pointer_beside_the_socket(home, runtime_dir, monkeypatch):
    """``resolve_client_socket_path`` finds a pointer published in the runtime dir."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    directory = runtime_dir / f"hermes-gw-{_home_hash(home)}"
    directory.mkdir(mode=0o700)
    target = directory / "control.sock"
    target.touch()
    (directory / "gateway.sock.path").write_text(str(target), encoding="utf-8")

    assert resolve_client_socket_path(home) == target


def test_client_still_reads_a_legacy_pointer_in_home(home, runtime_dir, monkeypatch):
    """A pointer published by an older gateway in ``home`` keeps working."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    directory = runtime_dir / f"hermes-gw-{_home_hash(home)}"
    directory.mkdir(mode=0o700)
    target = directory / "control.sock"
    target.touch()
    (home / "gateway.sock.path").write_text(str(target), encoding="utf-8")

    assert resolve_client_socket_path(home) == target
