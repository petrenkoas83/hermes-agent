"""Regression tests for the Telegram polling health-confirmation log (#90504).

Both reconnect paths end on ``health pending getUpdates progress`` and
``_record_polling_progress`` used to complete silently, so the log stream for
"reconnected and healthy" was byte-identical to "reconnected and hung". The
first confirmed getUpdates round-trip of each generation now emits an anchor
line, turning the pending line into a resolvable pair whose *absence* after a
reconnect is a reliable hung-poll signature.

The anchors are WARNING (patch 13, incident 2026-11-11): the gateway's
stderr->docker handler is pinned to WARNING at default verbosity, so an INFO
anchor never reaches ``docker logs`` and the ansible health watchdog — which
greps that stream — could never clear a degradation verdict. The tests below
capture at WARNING, i.e. at exactly the level the watchdog reads.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gateway.config import Platform  # noqa: E402
from plugins.platforms.telegram import adapter as tg_adapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _bare_adapter():
    a = TelegramAdapter.__new__(TelegramAdapter)
    a.platform = Platform.TELEGRAM
    a._fatal_error_code = None
    a._fatal_error_message = None
    a._fatal_error_retryable = True
    a._polling_teardown_started = False
    a._polling_progress_accepting = True
    a._polling_generation = 1
    a._polling_progress_event = asyncio.Event()
    a._polling_network_error_count = 2
    a._polling_conflict_count = 3
    a._polling_conflict_recovery_generation = None
    a._send_path_degraded = True
    return a


def _app_stub():
    """Minimal PTB application double: no running updater, so the reconnect
    ladders skip updater.stop() and go straight to start_polling()."""
    app = MagicMock()
    app.updater = MagicMock()
    app.updater.running = False
    return app


def _speed_up_sleeps(monkeypatch):
    """Collapse the reconnect back-off waits (5-60 s) so ladder tests are fast.

    Only delays >= 1 s are collapsed; everything shorter (event-loop yields,
    test polling helpers) keeps its real semantics.
    """
    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *args, **kwargs):
        if delay and delay >= 1:
            return await real_sleep(0)
        return await real_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(tg_adapter.asyncio, "sleep", fast_sleep)


class TestPollingHealthConfirmation:
    def test_first_progress_emits_confirmed_healthy(self, caplog):
        a = _bare_adapter()
        with caplog.at_level(logging.INFO, logger="plugins.platforms.telegram.adapter"):
            a._record_polling_progress(1)
        rendered = " | ".join(rec.getMessage() for rec in caplog.records)
        assert "confirmed healthy" in rendered
        assert "generation 1" in rendered
        assert a._polling_progress_event.is_set()

    def test_confirmed_healthy_survives_warning_floor(self, caplog):
        """The anchor must be visible at WARNING — that is the floor of the
        gateway's stderr->docker handler, and therefore the only level the
        watchdog can ever see (patch 13)."""
        a = _bare_adapter()
        caplog.set_level(logging.WARNING, logger="plugins.platforms.telegram.adapter")
        a._record_polling_progress(1)
        records = [
            rec for rec in caplog.records if "confirmed healthy" in rec.getMessage()
        ]
        assert records, (
            "confirmed-healthy anchor is below WARNING and invisible to "
            "`docker logs`; the health watchdog can never go green"
        )
        assert records[0].levelno == logging.WARNING

    @pytest.mark.asyncio
    async def test_network_restart_anchor_survives_warning_floor(self, caplog, monkeypatch):
        """`polling restarted after network error` is the other half of the
        pending/confirmed pair; it must clear the WARNING floor too."""
        _speed_up_sleeps(monkeypatch)
        a = _bare_adapter()
        a._polling_network_error_count = 0
        a._app = _app_stub()
        a._polling_error_callback_ref = None
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.telegram.adapter"), \
             patch.object(a, "_drain_polling_connections", new=AsyncMock()), \
             patch.object(a, "_start_polling_once", new=AsyncMock()), \
             patch.object(
                 a, "_looks_like_pool_timeout", return_value=False, create=True
             ):
            await a._handle_polling_network_error(RuntimeError("net down"))
        records = [
            rec
            for rec in caplog.records
            if "restarted after network error" in rec.getMessage()
        ]
        assert records, (
            "network-restart anchor is below WARNING and invisible to `docker logs`"
        )
        assert records[0].levelno == logging.WARNING

    @pytest.mark.asyncio
    async def test_conflict_restart_anchor_survives_warning_floor(self, caplog, monkeypatch):
        """`polling restarted after conflict retry` must clear the same floor."""
        _speed_up_sleeps(monkeypatch)
        a = _bare_adapter()
        a._polling_conflict_count = 0
        a._app = _app_stub()
        a._polling_error_callback_ref = None
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.telegram.adapter"), \
             patch.object(a, "_drain_polling_connections", new=AsyncMock()), \
             patch.object(a, "_start_polling_once", new=AsyncMock()):
            await a._handle_polling_conflict(RuntimeError("409 conflict"))
        records = [
            rec
            for rec in caplog.records
            if "restarted after conflict retry" in rec.getMessage()
        ]
        assert records, (
            "conflict-restart anchor is below WARNING and invisible to `docker logs`"
        )
        assert records[0].levelno == logging.WARNING

    def test_subsequent_progress_is_silent(self, caplog):
        """Only the FIRST round-trip of a generation logs — a quiet evening
        must not spam one anchor per getUpdates poll."""
        a = _bare_adapter()
        a._record_polling_progress(1)  # first — logs
        # Drop the first record: at WARNING (patch 13) it is captured even
        # without at_level(), and this test is about the follow-ups.
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="plugins.platforms.telegram.adapter"):
            a._record_polling_progress(1)  # second — silent
            a._record_polling_progress(1)  # third — silent
        assert not [
            rec for rec in caplog.records if "confirmed healthy" in rec.getMessage()
        ]

    def test_new_generation_logs_again(self, caplog):
        """A reconnect starts a new generation with a fresh event; its first
        progress must re-emit the confirmation so the pending line of THAT
        reconnect also resolves."""
        a = _bare_adapter()
        a._record_polling_progress(1)
        # reconnect: new generation, event reset, counters possibly nonzero
        a._polling_generation = 2
        a._polling_progress_event = asyncio.Event()
        a._polling_network_error_count = 1
        a._send_path_degraded = True
        with caplog.at_level(logging.INFO, logger="plugins.platforms.telegram.adapter"):
            a._record_polling_progress(2)
        rendered = " | ".join(rec.getMessage() for rec in caplog.records)
        assert "confirmed healthy" in rendered
        assert "generation 2" in rendered

    def test_stale_generation_progress_stays_silent(self, caplog):
        """Progress from an abandoned generation must neither log nor set the
        current event (pre-existing guard, pinned here because the log line
        must inherit the same generation-scoping)."""
        a = _bare_adapter()
        a._polling_generation = 2
        a._polling_progress_event = asyncio.Event()
        with caplog.at_level(logging.INFO, logger="plugins.platforms.telegram.adapter"):
            a._record_polling_progress(1)
        assert not [
            rec for rec in caplog.records if "confirmed healthy" in rec.getMessage()
        ]
        assert not a._polling_progress_event.is_set()
