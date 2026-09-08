"""Placeholder tests for the event engine (M9)."""

import pytest


@pytest.mark.skip(reason="M9: EventEngine not implemented yet")
def test_event_created_with_status_new():
    """Emitted events must start with status NEW."""


@pytest.mark.skip(reason="M9: EventEngine not implemented yet")
def test_engine_never_sets_verified():
    """VERIFIED must only ever be set by a human reviewer, never the engine."""


@pytest.mark.skip(reason="M9: EventEngine not implemented yet")
def test_duplicate_observations_are_deduplicated():
    """Repeated observations should not spam duplicate events."""


@pytest.mark.skip(reason="M9: EventEngine not implemented yet")
def test_event_evidence_is_attached():
    """Events should carry frame references and the thresholds applied."""
