"""Regression tests for the Telegram heartbeat probe budget (patch 14).

Incident 2026-11-11: the heartbeat loop probes ``bot.get_me()`` every 90 s with
``asyncio.wait_for(bot.get_me(), PROBE_TIMEOUT)``. That request travels through
``TelegramFallbackTransport``, which walks the configured IPv4 literals *before*
the dual-stack hostname and applies **no per-target deadline**: every candidate
gets its own ``HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT`` (default 10 s). One seed
literal, ``149.154.167.220``, is blackholed from this guest (TCP connect never
completes; measured 4/4 timeouts at ~6-12 s) while ``149.154.166.110`` connects
instantly.

So the worst-case walk cost ``10 s + 10 s = 20 s`` before the hostname path was
even attempted, but ``PROBE_TIMEOUT`` was 15 s. The probe raised
``asyncio.TimeoutError`` mid-walk and ``_schedule_polling_recovery`` stopped and
restarted a *healthy* poller — 8 generations in one day, 17-21 s deaf windows
each (32 s worst), plus one full adapter rebuild when ``updater.stop()`` blew
its own budget.

The fix has two halves, both pinned here:

1. ``PROBE_TIMEOUT`` must exceed the worst-case two-candidate walk (arithmetic),
   and stay below the probe cadence so probes never overlap.
2. A tick whose first candidate stalls must still complete and must NOT
   schedule polling recovery.
"""

import asyncio
import inspect
import logging
import re

import httpx
import pytest

import plugins.platforms.telegram.telegram_network as tg_network
from gateway.config import Platform  # noqa: E402
from plugins.platforms.telegram import adapter as tg_adapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402

# Measured cost of one TCP connect attempt to the blackholed literal
# 149.154.167.220 from the guest (see incident writeup in
# docs/21-state-db-local-disk.md). The *floor* the probe budget must clear is
# two such candidates in a row, which is what the arithmetic test asserts.
MEASURED_BLACKHOLE_CONNECT_SECONDS = 12.0

_LOGGER_NAME = "plugins.platforms.telegram.adapter"


# ---------------------------------------------------------------------------
# Source-derived constants
# ---------------------------------------------------------------------------

def _float_const(pattern, source):
    match = re.search(pattern, source)
    assert match, "constant not found in source: %r" % pattern
    return float(match.group(1))


def _heartbeat_constants():
    """PROBE_TIMEOUT / HEARTBEAT_INTERVAL are locals of the heartbeat loop."""
    source = inspect.getsource(TelegramAdapter._polling_heartbeat_loop)
    return (
        _float_const(r"PROBE_TIMEOUT\s*=\s*([\d.]+)", source),
        _float_const(r"HEARTBEAT_INTERVAL\s*=\s*([\d.]+)", source),
    )


def _http_connect_timeout_default():
    """Default of HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT (the per-candidate cost)."""
    source = inspect.getsource(tg_adapter)
    return _float_const(
        r'HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT",\s*([\d.]+)', source
    )


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _StallingTransport(httpx.AsyncBaseTransport):
    """First candidate stalls (blackholed-IP shape), the rest succeed."""

    def __init__(self, calls, stall_host, stall_seconds):
        self.calls = calls
        self.stall_host = stall_host
        self.stall_seconds = stall_seconds

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        self.calls.append(host)
        if host == self.stall_host:
            await asyncio.sleep(self.stall_seconds)
            # A blackholed endpoint surfaces as a retryable connect error,
            # never as a protocol error — see _is_retryable_connect_error.
            raise httpx.ConnectTimeout("connect timed out")
        return httpx.Response(200, request=request, text="ok")

    async def aclose(self) -> None:
        return None


def _install_transport_factory(monkeypatch, calls, stall_host, stall_seconds):
    def factory(*args, **kwargs):
        return _StallingTransport(calls, stall_host, stall_seconds)

    monkeypatch.setattr(tg_network.httpx, "AsyncHTTPTransport", factory)


def _bare_adapter():
    a = TelegramAdapter.__new__(TelegramAdapter)
    a.platform = Platform.TELEGRAM
    a._fatal_error_code = None
    a._fatal_error_message = None
    a._fatal_error_retryable = True
    a._polling_teardown_started = False
    a._polling_error_task = None
    a._polling_network_error_count = 0
    a._polling_generation = 1
    a._background_tasks = set()
    a._send_path_degraded = False
    return a


def _collapse_heartbeat_interval(monkeypatch):
    """Skip the 90 s wait between probes; keep real sleeps for the stall."""
    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *args, **kwargs):
        if delay and delay >= 1:  # HEARTBEAT_INTERVAL / reconnect back-off
            return await real_sleep(0)
        return await real_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(tg_adapter.asyncio, "sleep", fast_sleep)


# ---------------------------------------------------------------------------
# 1. The budget must clear the worst-case walk
# ---------------------------------------------------------------------------

class TestProbeBudgetArithmetic:
    def test_probe_timeout_covers_two_dead_candidates(self):
        probe_timeout, _ = _heartbeat_constants()
        connect_timeout = _http_connect_timeout_default()
        worst_case_walk = 2 * connect_timeout
        assert probe_timeout >= worst_case_walk, (
            "PROBE_TIMEOUT=%ss does not cover a two-candidate fallback walk "
            "(%ss): the heartbeat would time out mid-walk and restart a "
            "healthy poller" % (probe_timeout, worst_case_walk)
        )

    def test_probe_timeout_covers_measured_blackhole(self):
        probe_timeout, _ = _heartbeat_constants()
        assert probe_timeout > MEASURED_BLACKHOLE_CONNECT_SECONDS, (
            "PROBE_TIMEOUT=%ss is at or below the measured connect cost to the "
            "blackholed literal (%ss)"
            % (probe_timeout, MEASURED_BLACKHOLE_CONNECT_SECONDS)
        )

    def test_probe_timeout_below_heartbeat_interval(self):
        probe_timeout, interval = _heartbeat_constants()
        assert probe_timeout < interval, (
            "PROBE_TIMEOUT=%ss must stay below HEARTBEAT_INTERVAL=%ss so "
            "probes never overlap" % (probe_timeout, interval)
        )


# ---------------------------------------------------------------------------
# 2. A slow first candidate must not trip recovery
# ---------------------------------------------------------------------------

class TestSlowCandidateDoesNotTripRecovery:
    @pytest.mark.asyncio
    async def test_tick_with_stalled_first_candidate_schedules_no_recovery(
        self, monkeypatch, caplog
    ):
        """One heartbeat tick, candidate #1 stalls, candidate #2 answers.

        The tick must finish through the SECOND candidate and must not call
        _schedule_polling_recovery. Before patch 14 the probe budget (15 s) was
        shorter than the stalled walk (~20 s), so this tick raised
        asyncio.TimeoutError and tore down a healthy poller.

        The stall is scaled down for suite speed (0.3 s instead of the measured
        12 s), but the REAL PROBE_TIMEOUT is in force: the assertion that matters
        is the arithmetic one above, and this test proves the walk continues
        past a failed candidate without escalating.
        """
        calls = []
        _install_transport_factory(
            monkeypatch, calls, stall_host="149.154.167.220", stall_seconds=0.3
        )
        _collapse_heartbeat_interval(monkeypatch)

        # Real fallback transport: the shipped walk, not a stub.
        transport = tg_network.TelegramFallbackTransport(
            ["149.154.167.220", "149.154.166.110"]
        )

        async def get_me():
            return await transport.handle_async_request(
                httpx.Request("GET", "https://api.telegram.org/bot<redacted>/getMe")
            )

        bot = type("Bot", (), {"get_me": staticmethod(get_me), "username": "hermes_bot"})()
        a = _bare_adapter()
        a._app = type("App", (), {"bot": bot, "updater": None})()

        recovery = []
        a._schedule_polling_recovery = lambda err, *, reason: recovery.append(
            (err, reason)
        )

        async def _noop(*args, **kwargs):
            return None

        monkeypatch.setattr(a, "_probe_pending_updates", _noop, raising=False)
        monkeypatch.setattr(a, "_check_polling_stall", _noop, raising=False)

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            task = asyncio.create_task(a._polling_heartbeat_loop())
            try:
                for _ in range(300):  # <= ~3 s wall clock
                    if getattr(a, "_bot_identity_checked_at", None):
                        break
                    await asyncio.sleep(0.01)
            finally:
                task.cancel()
                # The loop handles CancelledError internally (`return`), so
                # awaiting the cancelled task completes instead of raising.
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        assert getattr(a, "_bot_identity_checked_at", None), (
            "heartbeat tick never completed get_me()"
        )
        assert recovery == [], (
            "a slow-but-reachable path scheduled polling recovery: %r" % (recovery,)
        )
        # The walk really did visit both candidates, in order.
        assert calls[0] == "149.154.167.220"
        assert "149.154.166.110" in calls
        # ...and the surviving candidate became the sticky path.
        assert transport._sticky_ip == "149.154.166.110"
        await transport.aclose()


class TestFailedCandidateDemotion:
    """The transport's short-lived negative cache (patch 14 defensive half)."""

    @pytest.mark.asyncio
    async def test_failed_candidate_is_demoted_not_dropped(self, monkeypatch):
        calls = []
        _install_transport_factory(
            monkeypatch, calls, stall_host="149.154.167.220", stall_seconds=0.0
        )
        transport = tg_network.TelegramFallbackTransport(
            ["149.154.167.220", "149.154.166.110"]
        )
        response = await transport.handle_async_request(
            httpx.Request("GET", "https://api.telegram.org/bot<redacted>/getMe")
        )
        assert response.status_code == 200

        order = transport._attempt_order()
        # Sticky winner first, then the hostname, and the dead literal LAST —
        # still present, so an IPv4-only network can still reach it.
        assert order == ["149.154.166.110", None, "149.154.167.220"]
        await transport.aclose()

    @pytest.mark.asyncio
    async def test_demotion_expires(self, monkeypatch):
        calls = []
        _install_transport_factory(
            monkeypatch, calls, stall_host="149.154.167.220", stall_seconds=0.0
        )
        transport = tg_network.TelegramFallbackTransport(["149.154.167.220"])
        await transport.handle_async_request(
            httpx.Request("GET", "https://api.telegram.org/bot<redacted>/getMe")
        )
        transport._sticky_ip = tg_network._UNSET
        transport._failed_until["149.154.167.220"] = 0.0  # cooldown expired
        assert transport._attempt_order() == ["149.154.167.220", None]
        await transport.aclose()
