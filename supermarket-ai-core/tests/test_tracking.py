"""Placeholder tests for single-camera tracking (M2)."""

import pytest


@pytest.mark.skip(reason="M2: Tracker not implemented yet")
def test_tracker_assigns_stable_local_ids():
    """The same person across frames should keep one local track id."""


@pytest.mark.skip(reason="M2: Tracker not implemented yet")
def test_local_track_id_format():
    """Local ids should be camera-scoped, e.g. 'C1_07'."""


@pytest.mark.skip(reason="M2: Tracker not implemented yet")
def test_tracker_handles_short_occlusion():
    """A briefly occluded person should not receive a new local id."""
