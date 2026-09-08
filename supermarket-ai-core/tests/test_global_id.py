"""Placeholder tests for cross-camera global identity (M4)."""

import pytest


@pytest.mark.skip(reason="M4: GlobalIDManager not implemented yet")
def test_new_person_receives_new_global_id():
    """An unmatched track should create a fresh anonymous global id."""


@pytest.mark.skip(reason="M4: GlobalIDManager not implemented yet")
def test_person_enters_searching_state_after_leaving_camera():
    """Losing a track should move the person to SEARCHING."""


@pytest.mark.skip(reason="M4: GlobalIDManager not implemented yet")
def test_match_restricted_to_topology_next_cameras():
    """CAM1 exits should only be matched on CAM2/CAM3 per cameras.yaml."""


@pytest.mark.skip(reason="M4: GlobalIDManager not implemented yet")
def test_identity_expires_after_global_id_timeout():
    """Identities unseen beyond GLOBAL_ID_TIMEOUT should expire."""


@pytest.mark.skip(reason="M4: GlobalIDManager not implemented yet")
def test_journey_records_camera_history_in_order():
    """camera_history should reflect CAM1 -> CAM2 -> CAM4 ordering."""
