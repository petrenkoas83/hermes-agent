"""Regression tests for the Telegram heartbeat probe budget (patch 14).

Incident 2026-11-11: the heartbeat loop probes ``bot.get_me()`` every 90 s under
``asyncio.wait_for(bot.get_me(), PROBE_TIMEOUT)``. That request travels through
``TelegramFallbackTransport``, whose walk tries every configured IPv4 literal
before the dual-stack hostname. One seed literal, ``149.154.167.220``, is
blackholed from this guest (TCP connect never completes; measured 4/4 timeouts
at ~6-12 s) while ``149.154.166.110`` connects instantly, so the walk burned
more connect time than the probe budget and ``_schedule_polling_recovery``
stopped and restarted a *healthy* poller — 8 generations in one day, 17-21 s
deaf windows each (32 s worst), plus one full adapter rebuild when
``updater.stop()`` blew its own budget.

On v0.23.2 upstream bounds the walk (``_WALK_CONNECT_BUDGET_S``, #136412), which
removes the unbounded half of the bug, but the probe budget still has to clear
that walk plus the read and pool phases of the same request. The invariant
pinned here is:

    PROBE_TIMEOUT >= _WALK_CONNECT_BUDGET_S + read_timeout + pool_timeout

and ``PROBE_TIMEOUT < HEARTBEAT_INTERVAL`` so probes never overlap.

These tests read the two modules as source text instead of importing them: the
telegram adapter needs ``python-telegram-bot`` at import time, and this
invariant is about constants, not behaviour — keeping it runnable anywhere.
"""

import re
from pathlib import Path

_TG = Path(__file__).resolve().parents[2] / "plugins" / "platforms" / "telegram"
_ADAPTER_SRC = (_TG / "adapter.py").read_text(encoding="utf-8")
_NETWORK_SRC = (_TG / "telegram_network.py").read_text(encoding="utf-8")

# Measured cost of one TCP connect attempt to the blackholed literal
# 149.154.167.220 from the guest (see the incident writeup in
# docs/21-state-db-local-disk.md). A single such candidate must not be able to
# consume the whole probe budget.
MEASURED_BLACKHOLE_CONNECT_SECONDS = 12.0


def _function_source(source: str, name: str) -> str:
    """Source of one method/function, up to the next same-or-less-indented def."""
    start = re.search(rf"^( *)(?:async )?def {re.escape(name)}\(", source, re.M)
    assert start, f"function not found: {name}"
    indent = start.group(1)
    # Skip the rest of the signature line: its closing ``) -> T:`` sits at column 0
    # for module-level defs and would end the slice immediately.
    body_start = source.index("\n", start.end()) + 1
    rest = source[body_start:]
    end = re.search(rf"^{indent}(?:async )?def |^{indent}\S", rest, re.M)
    return rest[: end.start()] if end else rest


def _float_const(pattern: str, source: str) -> float:
    match = re.search(pattern, source)
    assert match, f"constant not found in source: {pattern!r}"
    return float(match.group(1))


def _heartbeat_constants() -> tuple[float, float]:
    """PROBE_TIMEOUT / HEARTBEAT_INTERVAL are locals of the heartbeat loop."""
    body = _function_source(_ADAPTER_SRC, "_polling_heartbeat_loop")
    return (
        _float_const(r"PROBE_TIMEOUT\s*=\s*([\d.]+)", body),
        _float_const(r"HEARTBEAT_INTERVAL\s*=\s*([\d.]+)", body),
    )


def _walk_budget() -> float:
    """Whole-walk CONNECT budget shared by every fallback candidate (#136412)."""
    return _float_const(r"_WALK_CONNECT_BUDGET_S\s*=\s*([\d.]+)", _NETWORK_SRC)


def _request_timeout_defaults() -> tuple[float, float]:
    """(read, pool) defaults of the PTB request the probe rides on.

    They live in ``telegram_network.ptb_request_kwargs()``, which the adapter
    applies to both of its PTB clients.
    """
    kwargs_src = _function_source(_NETWORK_SRC, "ptb_request_kwargs")
    return (
        _float_const(r'HERMES_TELEGRAM_HTTP_READ_TIMEOUT",\s*([\d.]+)', kwargs_src),
        _float_const(r'HERMES_TELEGRAM_HTTP_POOL_TIMEOUT",\s*([\d.]+)', kwargs_src),
    )


class TestProbeBudgetArithmetic:

    def test_probe_timeout_covers_walk_read_and_pool(self):
        probe_timeout, _ = _heartbeat_constants()
        walk_budget = _walk_budget()
        read_timeout, pool_timeout = _request_timeout_defaults()
        worst_case = walk_budget + read_timeout + pool_timeout
        assert probe_timeout >= worst_case, (
            "PROBE_TIMEOUT=%ss does not cover one get_me() through the fallback "
            "walk: %ss connect budget + %ss read + %ss pool = %ss. The probe "
            "would time out mid-request and restart a healthy poller"
            % (probe_timeout, walk_budget, read_timeout, pool_timeout, worst_case)
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

    def test_walk_budget_is_bounded_at_all(self):
        """The unbounded-walk shape this patch was written against must stay gone."""
        assert _walk_budget() > 0
        assert "connect_budget" in _function_source(
            _NETWORK_SRC, "handle_async_request"
        ), "TelegramFallbackTransport lost its connect budget (#136412)"
