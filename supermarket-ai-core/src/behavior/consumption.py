"""Potential in-store consumption detection interface (M7 placeholder)."""

from __future__ import annotations

from typing import Any, Optional

from src.behavior.state_machine import BehaviorObservation


class ConsumptionDetector:
    """Flags *potential* in-store consumption for human review.

    Heuristic sketch (M7): a carried product repeatedly approaches the face
    region and its size/visibility decreases over time. The face region is a
    coarse geometric zone only — no facial recognition is performed.
    """

    def __init__(self, confidence_threshold: float) -> None:
        self.confidence_threshold = confidence_threshold

    def evaluate(self, context: Any) -> Optional[BehaviorObservation]:
        """Return a POTENTIAL_CONSUMPTION observation, or None."""
        raise NotImplementedError

    def face_region(self, person_bbox: Any) -> Any:
        """Derive the coarse face-region rectangle from a person bounding box."""
        raise NotImplementedError
