"""The reserved-profile authority accepts ``state.db`` in the home OR in ``home/db``.

LOCAL PATCH (fork) coverage.  The vboxsf layout keeps the store on guest-local
ext4, bind-mounted at ``HERMES_HOME/db`` and reached through the symlink
``HERMES_HOME/state.db -> db/state.db`` (docs/21-state-db-local-disk.md).  The
resolved store therefore lives in ``HERMES_HOME/db``, not in ``HERMES_HOME``.
The fork relaxes exactly that one comparison; a store outside the profile's own
subtree must still be refused, and the upstream ``home`` case must keep working
byte-identically.

Hermetic: ``open_profile_store`` and ``initialize_session_authority`` are
fakes, so no SQLite file is ever created (nothing lands on vboxsf).
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import run_runtime


class _Registry:
    def __init__(self):
        self.added, self.replaced, self.removed = [], [], []

    def add(self, home, value, name=None):
        self.added.append(home)

    def replace(self, home, authority):
        self.replaced.append(home)

    def remove(self, home):
        self.removed.append(home)


@pytest.fixture
def build_authority(monkeypatch):
    """Run ``_build_profile_authority`` against fakes, with the store reported by path."""
    import gateway.run as gateway_run
    import gateway.session_authority as session_authority

    reported = {}

    def fake_open(runner, home):
        return SimpleNamespace(db_path=reported['db_path'])

    async def fake_initialize(runner, *, profile_id, instance_id, db, register):
        return SimpleNamespace(profile_id=profile_id, db=db)

    monkeypatch.setattr(run_runtime, 'open_profile_store', fake_open)
    monkeypatch.setattr(gateway_run, '_profile_runtime_scope',
                        lambda home, hydrate_secrets=False: contextlib.nullcontext())
    monkeypatch.setattr(session_authority, 'initialize_session_authority', fake_initialize)

    async def run(home, db_path):
        reported['db_path'] = str(db_path)
        registry = _Registry()
        runner = SimpleNamespace(session_authorities=registry,
                                 session_runtime_descriptor={'instance_id': 'test-instance'})
        authority = await run_runtime._build_profile_authority(runner, 'worker', home, register=False)
        return authority, registry

    return run


@pytest.mark.asyncio
async def test_state_db_directly_in_the_profile_home_is_accepted(build_authority, tmp_path):
    """The upstream layout: resolved parent == home."""
    home = (tmp_path / 'home').resolve()
    home.mkdir()
    authority, registry = await build_authority(home, home / 'state.db')
    assert authority.profile_id == str(home)
    assert registry.replaced == [home]


@pytest.mark.asyncio
async def test_state_db_under_hermes_home_db_is_accepted(build_authority, tmp_path):
    """Our layout: the store is bind-mounted at HERMES_HOME/db."""
    home = (tmp_path / 'home').resolve()
    (home / 'db').mkdir(parents=True)
    authority, registry = await build_authority(home, home / 'db' / 'state.db')
    assert authority.profile_id == str(home)
    assert registry.replaced == [home]


@pytest.mark.asyncio
async def test_symlinked_state_db_resolves_into_home_db(build_authority, tmp_path):
    """HERMES_HOME/state.db -> db/state.db, exactly as docs/21 creates it.

    The symlink target need not exist: ``Path.resolve()`` is used non-strictly,
    as upstream does, so the check never creates or opens the store.
    """
    home = (tmp_path / 'home').resolve()
    (home / 'db').mkdir(parents=True)
    os.symlink(Path('db') / 'state.db', home / 'state.db')
    authority, registry = await build_authority(home, home / 'state.db')
    assert authority.profile_id == str(home)
    assert registry.replaced == [home]


@pytest.mark.asyncio
async def test_a_sibling_profile_home_is_still_refused(build_authority, tmp_path):
    home = (tmp_path / 'home').resolve()
    other = (tmp_path / 'home2').resolve()
    home.mkdir()
    other.mkdir()
    with pytest.raises(RuntimeError, match='does not belong to the reserved profile'):
        await build_authority(home, other / 'state.db')


@pytest.mark.asyncio
async def test_a_store_outside_the_profile_subtree_is_still_refused(build_authority, tmp_path):
    home = (tmp_path / 'home').resolve()
    outside = (tmp_path / 'x').resolve()
    home.mkdir()
    outside.mkdir()
    with pytest.raises(RuntimeError, match='does not belong to the reserved profile'):
        await build_authority(home, outside / 'state.db')


@pytest.mark.asyncio
async def test_a_parent_escape_is_still_refused(build_authority, tmp_path):
    """``home/../x/state.db`` resolves to a sibling and must not ride the new allowance."""
    home = (tmp_path / 'home').resolve()
    escaped = home / '..' / 'x'
    home.mkdir()
    escaped.mkdir()
    with pytest.raises(RuntimeError, match='does not belong to the reserved profile'):
        await build_authority(home, escaped / 'state.db')
