"""Placeholder tests for behaviour analysis (M5-M7)."""

import pytest


@pytest.mark.skip(reason="M5: BehaviorStateMachine not implemented yet")
def test_state_machine_starts_idle_and_transitions_legally():
    """IDLE -> PERSON_DETECTED -> PRODUCT_INTERACTION should be permitted."""


@pytest.mark.skip(reason="M5: BehaviorStateMachine not implemented yet")
def test_illegal_transitions_are_rejected():
    """IDLE -> POTENTIAL_CONCEALMENT should not be reachable directly."""


@pytest.mark.skip(reason="M6: ConcealmentDetector not implemented yet")
def test_concealment_flagged_as_potential_only():
    """A product entering a concealment zone should yield POTENTIAL_CONCEALMENT."""


@pytest.mark.skip(reason="M7: ConsumptionDetector not implemented yet")
def test_consumption_flagged_as_potential_only():
    """Product-to-face-region movement should yield POTENTIAL_CONSUMPTION."""


@pytest.mark.skip(reason="M5/M6: detectors not implemented yet")
def test_detectors_use_configured_confidence_threshold():
    """Observations below EVENT_CONFIDENCE_THRESHOLD should be suppressed."""
