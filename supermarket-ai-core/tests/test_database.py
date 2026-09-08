"""Placeholder tests for the persistence layer (M9)."""

import pytest


@pytest.mark.skip(reason="M9: Database not implemented yet")
def test_schema_initialisation_is_idempotent():
    """init_schema() should be safe to call repeatedly."""


@pytest.mark.skip(reason="M9: Database not implemented yet")
def test_persons_events_and_journeys_round_trip():
    """Saved records should be readable back unchanged."""


@pytest.mark.skip(reason="M9: Database not implemented yet")
def test_backend_is_swappable():
    """The same interface should work against SQLite and PostgreSQL."""
