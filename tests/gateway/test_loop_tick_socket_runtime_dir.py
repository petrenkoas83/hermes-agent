"""Loop-tick witness socket relocation off the shared folder (#P3).

``$HERMES_HOME`` is a VirtualBox shared folder (vboxsf) on the affected stand: ``bind(2)`` on an
AF_UNIX node there answers ``EPERM``, so the witness silently disarmed (``loop_tick_socket:false``).
The upstream control socket already moved to ``$XDG_RUNTIME_DIR/hermes-gw-<hash>/``; these tests pin
that the loop-tick witness reuses the same directory (different leaf), falls back to the system temp
directory, switches candidates on ``EPERM`` at bind time, and stays non-fatal when nothing is usable.

All tests are hermetic: every socket binds under a per-test private runtime/temp directory, never
under the real ``$XDG_RUNTIME_DIR`` or ``/tmp``.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

import gateway.shutdown_watchdog as shutdown_watchdog_module

from gateway.control_socket import _home_hash
from gateway.shutdown_watchdog import (
    get_loop_heartbeat_path,
    get_loop_tick_socket_path,
    loop_heartbeat_forever,
    loop_tick_socket_candidates,
    write_loop_heartbeat,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="the AF_UNIX witness is POSIX-only (Windows arms the TCP-loopback witness)",
)


def _short_dir(prefix: str):
    """Short ``mkdtemp`` root right under ``/tmp``.

    pytest's ``tmp_path`` nests deep enough that a witness path under it exceeds the AF_UNIX
    ``sun_path`` limit, so the candidate would be filtered as unusable — an artifact of the test
    harness, not of the code under test. Real ``$XDG_RUNTIME_DIR`` paths are short.
    """
    return Path(tempfile.mkdtemp(prefix=prefix, dir=tempfile.gettempdir()))


@pytest.fixture()
def home():
    path = _short_dir("p3h-")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def private_runtime():
    """A stand-in for ``$XDG_RUNTIME_DIR``: owned by us, ``0700`` — what ``_private_directory`` wants."""
    path = _short_dir("p3rt-")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def private_tempdir():
    path = _short_dir("p3tmp-")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


async def _await_payload(home, timeout_s=10.0):
    """Wait (task stays running) until the first heartbeat payload lands."""
    heartbeat_path = get_loop_heartbeat_path(home)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if heartbeat_path.is_file():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
                if payload:
                    return payload
        await asyncio.sleep(0.05)
    pytest.fail(f"heartbeat payload did not appear at {heartbeat_path} within {timeout_s}s")


async def _run_heartbeat_until_payload(home, timeout_s=10.0):
    """Run the real producer until the first heartbeat lands, then cancel it cleanly."""
    task = asyncio.ensure_future(loop_heartbeat_forever(interval_s=1.0, home=home))
    try:
        return await _await_payload(home, timeout_s)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task  # any other exception (a crashing heartbeat) fails the test here


def test_xdg_runtime_dir_is_the_preferred_witness_directory(home, private_runtime, monkeypatch):
    """Candidate #1 is the control socket's directory with a loop-tick leaf — never HERMES_HOME."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(private_runtime))

    candidates = loop_tick_socket_candidates(home, pid=4321)
    runtime_dir = private_runtime / f"hermes-gw-{_home_hash(home)}"

    assert candidates[0] == runtime_dir / "loop-tick.4321.sock"
    assert get_loop_tick_socket_path(home, 4321) == candidates[0]
    # Same directory formula as the control socket ($XDG_RUNTIME_DIR/hermes-gw-<hash>), same hash.
    assert candidates[0].parent == runtime_dir
    # The shared folder is not the preferred home for the node.
    assert home not in candidates[0].parents
    # The legacy HERMES_HOME path is only a last resort, after the runtime candidates.
    assert candidates.index(home / "state" / "gateway.loop-tick.4321.sock") == len(candidates) - 1


def test_missing_xdg_runtime_dir_falls_back_to_the_temp_directory(
    home, private_tempdir, monkeypatch
):
    """Without a private ``$XDG_RUNTIME_DIR`` the witness moves to the 0700 temp-dir root."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(private_tempdir))

    candidates = loop_tick_socket_candidates(home, pid=7)

    expected = private_tempdir / f"hermes-gw-{_home_hash(home)}" / "loop-tick.7.sock"
    assert candidates[0] == expected
    assert get_loop_tick_socket_path(home, 7) == expected
    # Nothing is created inside HERMES_HOME (the shared folder) as the first choice.
    assert home not in expected.parents


def test_an_already_bound_candidate_wins_over_the_env_preference(
    home, private_runtime, private_tempdir, monkeypatch
):
    """A daemon started without ``$XDG_RUNTIME_DIR`` binds in the temp dir; a probe that has one
    must still resolve that existing node (producer and probe environments can differ)."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(private_runtime))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(private_tempdir))

    bound = private_tempdir / f"hermes-gw-{_home_hash(home)}" / "loop-tick.99.sock"
    bound.parent.mkdir(mode=0o700)
    bound.touch()  # the resolver keys on node presence; a live bind is exercised by the EPERM test

    assert get_loop_tick_socket_path(home, 99) == bound


def test_published_heartbeat_path_wins_over_local_derivation(
    home, private_runtime, private_tempdir, monkeypatch
):
    """The producer can bind under a root the probe would never derive (e.g. the scratch ``TMPDIR``
    redirect): the path published in the heartbeat is authoritative for that PID."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(private_runtime))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(private_tempdir))

    bound = private_tempdir / "elsewhere" / f"loop-tick.{os.getpid()}.sock"
    write_loop_heartbeat(
        pid=os.getpid(),
        home=home,
        extra={"loop_tick_socket": True, "loop_tick_socket_path": str(bound)},
    )

    assert get_loop_tick_socket_path(home) == bound
    # A payload for a different PID is not evidence for this one.
    assert get_loop_tick_socket_path(home, os.getpid() + 1) == (
        private_runtime / f"hermes-gw-{_home_hash(home)}" / f"loop-tick.{os.getpid() + 1}.sock"
    )


@pytest.mark.asyncio
async def test_eperm_on_the_first_candidate_switches_to_the_next_one(
    home, private_runtime, private_tempdir, monkeypatch
):
    """A vboxsf-style ``EPERM`` at the first candidate re-binds on the next, and the witness answers."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(private_runtime))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(private_tempdir))

    blocked = loop_tick_socket_candidates(home)[0]
    real_start_unix_server = asyncio.start_unix_server

    async def fake_start_unix_server(client_connected_cb, path=None, **kwargs):
        if str(path) == str(blocked):
            raise PermissionError(errno.EPERM, "Operation not permitted", str(path))
        return await real_start_unix_server(client_connected_cb, path=path, **kwargs)

    monkeypatch.setattr(
        shutdown_watchdog_module.asyncio, "start_unix_server", fake_start_unix_server
    )

    task = asyncio.ensure_future(loop_heartbeat_forever(interval_s=1.0, home=home))
    try:
        payload = await _await_payload(home)

        assert payload["loop_tick_socket"] is True
        armed_path = Path(payload["loop_tick_socket_path"])
        assert armed_path == private_tempdir / f"hermes-gw-{_home_hash(home)}" / f"loop-tick.{os.getpid()}.sock"
        assert not blocked.exists(), "the refused first candidate must not leave a node behind"
        # The relocated witness really answers the loop-scheduling ping (the task is still serving).
        reader, writer = await asyncio.open_unix_connection(str(armed_path))
        try:
            assert await asyncio.wait_for(reader.read(1), timeout=2) == b"1"
        finally:
            writer.close()
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_all_candidates_unavailable_degrades_non_fatally(
    home, private_runtime, private_tempdir, monkeypatch, caplog
):
    """Both runtime directories refusing the socket must still leave a healthy heartbeat task."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(private_runtime))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(private_tempdir))

    async def always_eperm(*args, **kwargs):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(
        shutdown_watchdog_module.asyncio, "start_unix_server", always_eperm
    )

    with caplog.at_level(logging.WARNING, logger="gateway.shutdown_watchdog"):
        payload = await _run_heartbeat_until_payload(home)

    # The beacon still works; only the witness is flagged as absent — exactly the pre-existing contract.
    assert payload["pid"] == os.getpid()
    assert payload["loop_tick_socket"] is False
    assert "loop_tick_socket_path" not in payload
    assert any(
        record.levelname == "WARNING" and "Loop tick socket unavailable" in record.getMessage()
        for record in caplog.records
    )
