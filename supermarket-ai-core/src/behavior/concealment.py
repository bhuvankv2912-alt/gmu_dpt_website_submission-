"""Potential concealment detection interface (M6 placeholder)."""

from __future__ import annotations

from typing import Any, Optional

from src.behavior.state_machine import BehaviorObservation


class ConcealmentDetector:
    """Flags a *potential* concealment for human review.

    Heuristic sketch (M6): a carried product enters a concealment zone (pocket,
    bag, waistband, under clothing) and does not re-emerge within a configured
    window. Output is always POTENTIAL and requires review; the detector must
    never assert theft.
    """

    def __init__(self, confidence_threshold: float) -> None:
        self.confidence_threshold = confidence_threshold

    def evaluate(self, context: Any) -> Optional[BehaviorObservation]:
        """Return a POTENTIAL_CONCEALMENT observation, or None."""
        raise NotImplementedError

    def concealment_zones(self, person_bbox: Any) -> Any:
        """Derive concealment-zone regions from a person bounding box."""
        raise NotImplementedError
