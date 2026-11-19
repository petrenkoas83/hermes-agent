"""Out-of-loop shutdown and event-loop liveness backstops.

A frozen asyncio loop takes every asyncio-based recovery path down with it, and launchd/systemd
KeepAlive only restarts a *dead* process. Hence: (1) an OS-thread shutdown watchdog that dumps
stacks and ``os._exit``s past ``restart_drain_timeout + grace``; (2) a heartbeat file at
``<HERMES_HOME>/state/gateway.heartbeat`` so supervisors can tell "process alive" from "loop
frozen"; (3) a lifetime thread watchdog that hard-exits when the loop is too frozen to run its
own callbacks; (4) a self-rescheduling floor timer that keeps the selector timeout finite."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import faulthandler
import json
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
from hermes_constants import get_hermes_home, get_process_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

# Extra leash beyond ``agent.restart_drain_timeout`` so a slow-but-progressing drain survives.
# Matches the issue #66892 suggested hardening.
DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S = 60.0
DEFAULT_HEARTBEAT_INTERVAL_S = 30.0
DEFAULT_LOOP_FLOOR_TIMER_INTERVAL_S = 5.0
DEFAULT_LOOP_WATCHDOG_INTERVAL_S = 30.0
DEFAULT_LOOP_WATCHDOG_TIMEOUT_S = 10.0
# 3 sustained misses (~90-120s of loop block) escalate; stays tight because the heartbeat write is
# off-loop. Slow loops tune gateway.loop_watchdog_* in config.yaml.
# The false-positive class that motivated raising this (the watchdog's own on-loop heartbeat fsync stalling
# the loop it monitors) is fixed at the root by the off-loop heartbeat write + two-witness probe (#90502),
# so the default stays tight for genuine wedges.
DEFAULT_LOOP_WATCHDOG_MAX_STRIKES = 3
_HEARTBEAT_RELATIVE = ("state", "gateway.heartbeat")
_WATCHDOG_DUMP_RELATIVE = ("logs", "gateway-shutdown-watchdog.log")
# A filesystem that cannot host an AF_UNIX node answers these at bind time: a VirtualBox shared
# folder (vboxsf) refuses with EPERM, Docker Desktop's virtiofs with EOPNOTSUPP. They mean "this
# filesystem cannot host a socket", not "the gateway has no witness", so the next runtime candidate
# is tried. Not EADDRINUSE: a live sibling must keep failing loudly.
_LOOP_TICK_UNSUPPORTED_ERRNOS = frozenset(
    {errno.EPERM, errno.EACCES, errno.EOPNOTSUPP, errno.ENOTSUP})


def _coerce_float(value: Any, default: float, floor: float = 0.0) -> float:
    """``max(float(value), floor)``, or ``default`` when not coercible."""
    try:
        return max(float(value), floor)
    except (TypeError, ValueError):
        return default


class _LoopFloorTimerHandle:
    """Cancelable owner for the currently scheduled selector floor timer."""
    def __init__(self, loop: asyncio.AbstractEventLoop, interval: float):
        self._loop, self._interval, self._cancelled = loop, interval, False
        self._timer: Optional[asyncio.TimerHandle] = None
        self._tick()

    def _tick(self) -> None:
        if not self._cancelled:
            self._timer = self._loop.call_later(self._interval, self._tick)

    def cancel(self) -> None:
        self._cancelled = True
        if self._timer is not None:
            self._timer.cancel()


class _LoopLivenessWatchdogHandle:
    def __init__(self, stop_event: threading.Event, thread: threading.Thread):
        self._stop_event = stop_event
        self.stop, self.join, self.is_alive = stop_event.set, thread.join, thread.is_alive


def _arm_loop_floor_timer(
    loop: asyncio.AbstractEventLoop, interval: float = DEFAULT_LOOP_FLOOR_TIMER_INTERVAL_S
) -> _LoopFloorTimerHandle:
    """Keep at least one timer pending so selector waits remain bounded."""
    iv = _coerce_float(interval, 0.0)
    return _LoopFloorTimerHandle(loop, iv if iv > 0 else DEFAULT_LOOP_FLOOR_TIMER_INTERVAL_S)


def start_loop_liveness_watchdog(
    loop: asyncio.AbstractEventLoop, *, probe_interval: float = DEFAULT_LOOP_WATCHDOG_INTERVAL_S,
    probe_timeout: float = DEFAULT_LOOP_WATCHDOG_TIMEOUT_S,
    max_strikes: int = DEFAULT_LOOP_WATCHDOG_MAX_STRIKES,
    exit_code: int = GATEWAY_SERVICE_RESTART_EXIT_CODE,
) -> Optional[_LoopLivenessWatchdogHandle]:
    """Start an out-of-loop watchdog that hard-exits after missed probes. The caller
    (``GatewayRunner._start_loop_liveness_guards``) enforces the ``gateway.loop_watchdog: false``
    opt-out."""
    stop_event = threading.Event()
    # The thread a post-mortem has to read first is the loop's own. ``BaseEventLoop`` records it
    # while the loop runs, and this starter runs on the loop thread, so its own ident covers the
    # mock and not-yet-running cases.
    loop_thread_id = getattr(loop, "_thread_id", None)
    if not isinstance(loop_thread_id, int):
        loop_thread_id = threading.get_ident()

    def _watchdog() -> None:
        strikes = 0
        while not stop_event.wait(timeout=probe_interval):
            probe_event = threading.Event()
            try:
                loop.call_soon_threadsafe(probe_event.set)
            except RuntimeError:  # normally closed loop: nothing left to backstop
                return
            except Exception:
                logger.debug("Failed to schedule gateway loop liveness probe", exc_info=True)
                return
            deadline = time.monotonic() + probe_timeout
            while not stop_event.is_set():  # poll so a stop() mid-wait is honoured within ~50ms
                remaining = deadline - time.monotonic()
                if remaining <= 0 or probe_event.wait(timeout=min(remaining, 0.05)):
                    break
            else:
                return
            if probe_event.is_set():
                strikes = 0
                continue
            if stop_event.is_set():  # re-checked before each irreversible step: a late stop() wins
                return
            strikes += 1
            if strikes < max_strikes:
                continue
            if stop_event.is_set():
                return
            with contextlib.suppress(Exception):
                logger.critical(
                    "Gateway event loop missed %d consecutive liveness probes; dumping all thread "
                    "stacks and exiting with code %d so the service supervisor can restart it.",
                    strikes, exit_code)
            # File + stderr: a console-less launcher (Windows .vbs login item) discards stderr, so a
            # liveness death used to leave no stack anywhere (#120356, #91097).
            _write_watchdog_dump(
                get_shutdown_watchdog_dump_path(), delay_s=probe_timeout * strikes,
                snapshot={"strikes": strikes, "probe_interval_s": probe_interval},
                event="loop_liveness_watchdog_fired", loop_thread_id=loop_thread_id)
            if stop_event.is_set():
                return
            _mark_exited_quietly(exit_code, "loop_liveness_watchdog")
            _hard_exit(exit_code)
    thread = threading.Thread(target=_watchdog, daemon=True, name="gateway-loop-liveness-watchdog")
    try:
        thread.start()
    except Exception:
        logger.debug("Failed to start gateway loop liveness watchdog", exc_info=True)
        return None
    return _LoopLivenessWatchdogHandle(stop_event, thread)


def _hard_exit(exit_code: int) -> None:
    """``os._exit`` skips every cleanup: SIGKILL in-flight foreground commands first, they run in their
    own process group and would outlive the gateway, reparented to init."""
    with contextlib.suppress(Exception):
        from tools.environments.base import kill_live_foreground_processes
        kill_live_foreground_processes(now=True)
    os._exit(exit_code)


def _mark_exited_quietly(exit_code: int, reason: str) -> None:
    """Best-effort terminal stamp on BOTH lifecycle records before ``os._exit`` skips teardown:
    the lifecycle ledger (so the next boot names the watchdog, not SIGKILL/OOM) and
    ``gateway_state.json`` (so ``hermes gateway status`` and every other reader of that file stop
    seeing ``running`` for a process the watchdog killed — #113372). The runtime-status write goes
    LAST: it is the record housekeeping refreshes, so nothing may overwrite it after we stamp it."""
    with contextlib.suppress(Exception):
        from gateway.lifecycle_ledger import mark_exited
        mark_exited(exit_code, reason=reason)
    with contextlib.suppress(Exception):  # os._exit skips atexit: stamp the exit-metrics marker now
        from hermes_cli.observability.shared_metrics_process import stamp_exit
        stamp_exit("watchdog")
    with contextlib.suppress(Exception):
        from gateway.status import write_runtime_status
        # Only the supervisor-restart code asserts a restart; other codes leave the recorded
        # operator intent (a restart-drain that wedged is still a requested restart) untouched.
        restart = {"restart_requested": True} if exit_code == GATEWAY_SERVICE_RESTART_EXIT_CODE else {}
        write_runtime_status(
            gateway_state="degraded", exit_reason=reason, wait_timeout=0.25, **restart)


def _process_hermes_home() -> Path:
    """HERMES_HOME for process-level identity files (ignore profile overrides)."""
    return get_process_hermes_home() if os.environ.get("HERMES_HOME", "").strip() else get_hermes_home()


def _home(home: Optional[Path]) -> Path:
    return home if home is not None else _process_hermes_home()


def get_loop_heartbeat_path(home: Optional[Path] = None) -> Path:
    return _home(home).joinpath(*_HEARTBEAT_RELATIVE)


def loop_tick_socket_candidates(
    home: Optional[Path] = None, pid: Optional[int] = None
) -> list[Path]:
    """Ordered candidate paths for the loop-tick witness socket, best first.

    The witness must never live on a shared folder: ``bind(2)`` on VirtualBox vboxsf answers
    ``EPERM`` (#P3), which would silently disarm the loop-scheduling witness. The upstream control
    socket already relocated for exactly this reason, so the first candidate reuses its private
    directory formula — ``$XDG_RUNTIME_DIR/hermes-gw-<hash>/`` with the same ``<hash>`` of
    ``HERMES_HOME`` — and only the leaf differs (``loop-tick.<pid>.sock`` vs ``control.sock``).
    Then ``tempfile.gettempdir()`` (created ``0700`` by the caller), then ``/tmp``, and the legacy
    ``<HERMES_HOME>/state/gateway.loop-tick.<pid>.sock`` only as the last resort for hosts with no
    usable runtime directory. Only candidates that fit ``sun_path`` are returned; an empty list
    means AF_UNIX is unusable here and the caller must arm the TCP-loopback witness instead.
    """
    # Reuse the upstream helpers rather than copying the control-socket relocation code.
    from gateway.control_socket import _fits_sun_path, _home_hash, _private_runtime_dir

    pid = int(pid if pid is not None else os.getpid())
    name = Path(f"hermes-gw-{_home_hash(_home(home))}") / f"loop-tick.{pid}.sock"
    runtime = _private_runtime_dir()
    candidates: list[Path] = [runtime / name] if runtime is not None else []
    candidates.append(Path(tempfile.gettempdir()) / name)
    if os.name != "nt":  # no-tmp: ok — the short /tmp root keeps AF_UNIX under the sun_path limit
        candidates.append(Path("/tmp") / name)
    candidates.append(_home(home) / "state" / f"gateway.loop-tick.{pid}.sock")
    unique: list[Path] = []  # tempfile.gettempdir() is usually /tmp itself
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return [candidate for candidate in unique if _fits_sun_path(candidate)]


def _published_tick_socket_path(home: Path, pid: int) -> Optional[Path]:
    """The witness path a running gateway published in its heartbeat, if it did.

    Path derivation alone cannot align a probe with the producer: a daemon started without
    ``$XDG_RUNTIME_DIR`` (or with the scratch ``TMPDIR`` redirect pointing into ``HERMES_HOME``)
    binds under a different root than the CLI that probes it. The heartbeat is the one place both
    processes agree on, so the producer publishes the path it actually bound and the probe trusts it
    — the same "pointer" role ``gateway.sock.path`` plays for the control socket. Best-effort and
    read-only: any error or mismatch means "nothing published". The leaf must still be this PID's
    witness name, so a stale or foreign payload cannot point the probe at an arbitrary node.
    """
    try:
        payload = json.loads(get_loop_heartbeat_path(home).read_text(encoding="utf-8-sig"))
        if int(payload.get("pid") or 0) != int(pid):
            return None
        raw = payload.get("loop_tick_socket_path")
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(raw, str) or not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute() or not candidate.name.endswith(f"loop-tick.{pid}.sock"):
        return None
    return candidate


def get_loop_tick_socket_path(home: Optional[Path] = None, pid: Optional[int] = None) -> Path:
    """Preferred witness path — the first candidate from ``loop_tick_socket_candidates``.

    PID-suffixed so a stale node from a dead process is never mistaken for this gateway's witness.
    Served by the loop itself (``_tick_socket_handler``), so an answer proves the loop dispatches;
    the heartbeat cannot (#90502). The producer binds the first candidate its filesystem accepts and
    the consumer (the ``hermes_cli.gateway`` liveness probe) resolves the same deterministic path,
    so both sides agree without a pointer file; a pointer would have to live on the shared folder
    and is therefore not published.

    Falls back to the legacy ``<HERMES_HOME>/state/gateway.loop-tick.<pid>.sock`` when no runtime
    candidate fits ``sun_path``.

    A probe may run in a process provisioned differently from the gateway (a daemon started without
    ``$XDG_RUNTIME_DIR`` binds under the temp directory while the CLI has one), so resolution goes
    through, in order: the path the running gateway published in its heartbeat
    (``_published_tick_socket_path`` — authoritative), an already-bound candidate (the pid-suffixed
    node can only exist where the producer bound it), then the env-derived preference.
    """
    pid = int(pid if pid is not None else os.getpid())
    published = _published_tick_socket_path(_home(home), pid)
    if published is not None:
        return published
    candidates = loop_tick_socket_candidates(home, pid)
    for candidate in candidates:
        with contextlib.suppress(OSError):  # an un-stat-able node is absent, not an error
            if candidate.exists():
                return candidate
    if candidates:
        return candidates[0]
    return _home(home) / "state" / f"gateway.loop-tick.{pid}.sock"


def get_shutdown_watchdog_dump_path(home: Optional[Path] = None) -> Path:
    return _home(home).joinpath(*_WATCHDOG_DUMP_RELATIVE)


def _thread_name(ident: Optional[int]) -> Optional[str]:
    """Name of the live thread with this ident, or None once it has exited."""
    return next((thread.name for thread in threading.enumerate() if thread.ident == ident), None)


def write_loop_heartbeat(
    *, pid: Optional[int] = None, start_time: Optional[float] = None,
    home: Optional[Path] = None, extra: Optional[dict[str, Any]] = None) -> Path:
    """Atomically rewrite the loop-liveness heartbeat file; never raises.
    ``start_time`` (process start, epoch seconds) lets supervisors detect PID reuse."""
    path = get_loop_heartbeat_path(home)
    payload: dict[str, Any] = {"pid": int(pid if pid is not None else os.getpid()),
                               "updated_at": datetime.now(UTC).isoformat(),
                               "monotonic": time.monotonic()}
    if start_time is not None:
        payload["start_time"] = float(start_time)
    with contextlib.suppress(Exception):  # after an unclean death this is the last memory record
        from gateway.lifecycle_ledger import sample_memory
        if mem := sample_memory():
            payload["mem"] = mem
    if extra:
        payload.update(extra)
    try:
        atomic_json_write(path, payload, indent=None)
    except Exception:
        logger.debug("Failed to write gateway loop heartbeat", exc_info=True)
    return path


def resolve_shutdown_watchdog_delay(
    drain_timeout: float, *, grace_s: float = DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S) -> float:
    """Return the wall-clock leash for the shutdown watchdog thread."""
    grace = _coerce_float(grace_s, DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S)
    return _coerce_float(drain_timeout, 0.0) + grace


def _write_watchdog_dump(dump_path: Path, *, delay_s: float, snapshot: Optional[dict[str, Any]],
                         event: str = "shutdown_watchdog_fired",
                         loop_thread_id: Optional[int] = None) -> None:
    """Best-effort faulthandler + metadata dump before hard-exit (file, then stderr).

    ``loop_thread_id`` names the monitored event loop's thread, whose own stack is written before
    the all-threads dump: that dump does not say which thread runs the loop, which is the first
    thing a post-mortem of a liveness death needs.
    """
    header: dict[str, Any] = {"event": event, "pid": os.getpid(), "delay_s": delay_s,
                              "fired_at": datetime.now(UTC).isoformat(), "snapshot": snapshot or {}}
    loop_section = ""
    if loop_thread_id is not None:
        header["monitored_loop_thread_id"] = loop_thread_id
        header["monitored_loop_thread_name"] = _thread_name(loop_thread_id)
        loop_section = _format_loop_thread_section(loop_thread_id)
    with contextlib.suppress(OSError):  # unwritable dir: skip only the file copy, stderr still runs
        dump_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Exception), open(dump_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(header, default=str) + "\n" + loop_section)
        fh.flush()
        fh.write("--- faulthandler dump (all threads) ---\n")
        fh.flush()
        try:
            faulthandler.dump_traceback(file=fh, all_threads=True)
        except Exception:
            fh.write("(faulthandler.dump_traceback failed)\n")
        fh.write("--- end dump ---\n")
        fh.flush()
    with contextlib.suppress(Exception):  # stderr too: journald/launchd get it if disk is wedged
        # The loop section goes to stderr as well: with the disk wedged, journald/launchd keep only
        # this copy, and its faulthandler dump is capped at 100 threads, newest first.
        sys.stderr.write(f"Gateway {event.replace('_', ' ')} after {delay_s:.0f}s "
                         f"(pid={os.getpid()}); dumping all thread stacks.\n" + loop_section)
        sys.stderr.flush()
        faulthandler.dump_traceback(all_threads=True)


def _format_loop_thread_section(loop_thread_id: int) -> str:
    """The monitored loop thread's stack, formatted once for both dump copies. Built from
    ``sys._current_frames()`` + ``traceback.format_stack``: file, line, function and source line
    only, never frame locals, so no token or message text a frame holds can reach the dump.
    ``faulthandler`` cannot be asked for one thread, and its all-threads list stops after 100
    threads, newest first: the loop thread, started before every worker, is the one it drops."""
    try:
        frame = sys._current_frames().get(loop_thread_id)
        stack = ("".join(traceback.format_stack(frame)) if frame is not None
                 else "(monitored loop thread stack unavailable: thread may have exited)\n")
    except Exception as exc:  # health: allow BLE001 -- a hard-exit diagnostic path must never raise
        stack = f"(monitored loop thread stack unavailable: {type(exc).__name__})\n"
    return "--- monitored gateway loop thread stack ---\n" + stack


def arm_shutdown_watchdog(
    delay_s: float, *, done_event: Optional[threading.Event] = None,
    snapshot_fn: Optional[Callable[[], dict[str, Any]]] = None, exit_code: int = 1,
    dump_path: Optional[Path] = None, name: str = "gateway-shutdown-watchdog") -> threading.Event:
    """Arm a daemon-thread hard-exit backstop for a wedged shutdown path: exits quietly if
    ``done_event`` is set within ``delay_s``, else dumps diagnostics and ``os._exit(exit_code)``.
    Never raises; returns ``done_event`` for disarming."""
    done = done_event if done_event is not None else threading.Event()
    delay = _coerce_float(delay_s, DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S)
    if delay <= 0:
        return done

    def _watchdog() -> None:
        deadline = time.monotonic() + delay  # chunked wait so a late disarm is observed within ~1s
        while time.monotonic() < deadline:
            if done.wait(timeout=min(deadline - time.monotonic(), 1.0)):
                return
        if done.is_set():
            return
        try:
            snapshot = snapshot_fn() if snapshot_fn is not None else None
        except Exception as exc:
            snapshot = {"snapshot_error": repr(exc)}
        target = dump_path if dump_path is not None else get_shutdown_watchdog_dump_path()
        _write_watchdog_dump(target, delay_s=delay, snapshot=snapshot)
        with contextlib.suppress(Exception):
            logger.critical("Shutdown watchdog fired after %.0fs — forcing process exit "
                            "(asyncio drain path appears wedged; see %s)", delay, target)
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        # Mirror _exit_after_graceful_shutdown: release PID file + runtime lock BEFORE the log drain
        # (never strand locks), then drain the log queue so logger.critical lands before os._exit.
        with contextlib.suppress(Exception):
            # Mirror _exit_after_graceful_shutdown: release PID file + runtime lock BEFORE the log drain
            # (locks must never be stranded), then drain the async log queue so the logger.critical above
            # actually reaches the file before os._exit bypasses atexit. (#66892)
            from gateway.status import remove_pid_file, release_gateway_runtime_lock
            remove_pid_file()
            release_gateway_runtime_lock()
        with contextlib.suppress(Exception):
            from hermes_logging import drain_log_queue
            drain_log_queue(timeout=1.0)
        _mark_exited_quietly(exit_code, "shutdown_watchdog")
        _hard_exit(exit_code)
    try:
        threading.Thread(target=_watchdog, daemon=True, name=name).start()
    except Exception:
        logger.debug("Failed to arm shutdown watchdog", exc_info=True)
    return done


async def _tick_socket_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Answer a liveness ping with one byte; never raises. Runs on the gateway loop, so a reply
    witnesses loop schedulability; the write is a socket-buffer copy (no fsync), immune to the
    stalls that age the heartbeat."""
    try:
        writer.write(b"1")
        await writer.drain()
    except Exception:
        pass
    finally:  # close even on CancelledError (BaseException), as on BASE
        with contextlib.suppress(Exception):
            writer.close()


def _sweep_stale_tick_sockets(own_path: Path) -> None:
    """Unlink loop-tick socket nodes left by dead PIDs (POSIX only; never raises).
    create_unix_server removes a leftover node at OUR path (os._exit / SIGKILL skip the
    finally-unlink) but not SIBLING nodes from other dead PIDs. os.kill(pid, 0) is a liveness
    probe only on POSIX (Windows would TerminateProcess). The glob matches both the runtime leaf
    (``loop-tick.<pid>.sock``) and the legacy ``gateway.loop-tick.<pid>.sock``."""
    try:
        for stale in (p for p in own_path.parent.glob("*loop-tick.*.sock") if p != own_path):
            try:
                os.kill(int(stale.name.split(".")[-2]), 0)  # windows-footgun: ok — POSIX-only
            except (ValueError, IndexError, OSError):
                stale.unlink(missing_ok=True)
    except Exception:
        logger.debug("stale loop-tick socket sweep failed", exc_info=True)


async def _start_tick_unix_server(
    home: Optional[Path] = None,
) -> tuple[Path, asyncio.AbstractServer]:
    """Bind the loop-tick witness on the first candidate filesystem that accepts a socket.

    A shared folder (vboxsf) answers ``EPERM`` to ``bind(2)`` and Docker Desktop's virtiofs answers
    ``EOPNOTSUPP``: those mean "this filesystem cannot host a socket", so the next runtime candidate
    is tried instead of disarming the witness. Any other error is re-raised for the caller's single
    non-fatal warning. Returns ``(bind_path, server)``."""
    last_error: Optional[OSError] = None
    for candidate in loop_tick_socket_candidates(home):
        try:
            candidate.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _sweep_stale_tick_sockets(candidate)
            server = await asyncio.start_unix_server(_tick_socket_handler, path=str(candidate))
        except OSError as exc:
            if exc.errno not in _LOOP_TICK_UNSUPPORTED_ERRNOS:
                raise
            logger.debug("Loop-tick socket refused at %s (%s); trying the next runtime candidate",
                         candidate, exc.strerror or exc)
            last_error = exc
            continue
        with contextlib.suppress(OSError):
            os.chmod(candidate, 0o600)
        return candidate, server
    raise last_error or OSError(
        errno.ENOTSUP, "no runtime candidate can host the loop-tick socket")


async def loop_heartbeat_forever(
    *, interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S, start_time: Optional[float] = None,
    home: Optional[Path] = None, should_continue: Optional[Callable[[], bool]] = None) -> None:
    """Rewrite the loop heartbeat file on a cadence until cancelled / gated off. Runs on the
    gateway loop so a frozen loop lets the file age for monitors. The fsync write goes to a thread
    (inline, a stalled filesystem blocked the loop inside its own heartbeat and the liveness
    watchdog killed it: WSL2 fsync stalls p99 31s / max 112s vs a ~90-120s budget) but is *awaited*,
    never fire-and-forget: an unawaited task would keep the file fresh while the loop was wedged,
    and one in-flight write at a time keeps a long stall from queuing a thread per interval.
    Off-loop, file freshness no longer proves loop schedulability, so this task also arms a
    loop-scheduling witness (``_tick_socket_handler``, flagged as ``loop_tick_socket``); probes must
    require the witness to agree before classifying WEDGED."""
    interval = _coerce_float(interval_s, DEFAULT_HEARTBEAT_INTERVAL_S, floor=1.0)
    # The witness exposes only a scheduling ping, not control/auth operations. It lives in the
    # private runtime directory (never on the shared folder: bind(2) there answers EPERM); the TCP
    # loopback transport is used only when no AF_UNIX candidate fits sun_path. Consumers already
    # discover the port in the heartbeat, independent of their TMPDIR.
    tick_server = tick_socket_path = tick_tcp_port = None
    try:
        if os.name == "posix" and loop_tick_socket_candidates(home):
            tick_socket_path, tick_server = await _start_tick_unix_server(home)
        else:
            tick_server = await asyncio.start_server(_tick_socket_handler, host="127.0.0.1", port=0)
            for _s in tick_server.sockets or []:
                with contextlib.suppress(Exception):
                    if isinstance(_sname := _s.getsockname(), tuple) and len(_sname) >= 2:
                        tick_tcp_port = int(_sname[1])
                        break
    except Exception:
        tick_server = tick_socket_path = tick_tcp_port = None
        logger.warning("Loop tick socket unavailable — liveness probes will have no "
                       "loop-scheduling witness and will not escalate on a stale heartbeat",
                       exc_info=True)
    extra: dict[str, Any] = {"loop_tick_socket": tick_server is not None,
                             "loop_tick_tcp_port": tick_tcp_port}
    if tick_socket_path is not None:  # diagnostics only; consumers resolve the same path themselves
        extra["loop_tick_socket_path"] = str(tick_socket_path)
    try:
        while True:  # first write is immediate so monitors see a fresh file at once
            try:
                await asyncio.to_thread(write_loop_heartbeat, start_time=start_time, home=home,
                                        extra=extra)
            except Exception:  # write_loop_heartbeat never raises: executor problem, keep the task
                logger.debug("Loop heartbeat write failed off-loop", exc_info=True)
            if should_continue is not None and not should_continue():
                return
            await asyncio.sleep(interval)
            if should_continue is not None and not should_continue():
                return
    finally:
        if tick_server is not None:
            tick_server.close()
            with contextlib.suppress(Exception):
                await tick_server.wait_closed()
            if tick_socket_path is not None:
                with contextlib.suppress(Exception):
                    tick_socket_path.unlink(missing_ok=True)
