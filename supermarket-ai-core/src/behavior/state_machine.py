"""Behaviour state machine interface (M5 placeholder)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel


class BehaviorState(str, Enum):
    """States a tracked person can occupy during behaviour analysis."""

    IDLE = "IDLE"
    PERSON_DETECTED = "PERSON_DETECTED"
    PRODUCT_INTERACTION = "PRODUCT_INTERACTION"
    PRODUCT_PICKED = "PRODUCT_PICKED"
    PRODUCT_CARRIED = "PRODUCT_CARRIED"
    CONCEALMENT_ZONE = "CONCEALMENT_ZONE"
    FACE_REGION = "FACE_REGION"
    OCCLUSION = "OCCLUSION"
    POTENTIAL_CONCEALMENT = "POTENTIAL_CONCEALMENT"
    POTENTIAL_CONSUMPTION = "POTENTIAL_CONSUMPTION"


class BehaviorObservation(BaseModel):
    """One state observation for a global person at a point in time."""

    global_id: str
    camera_id: str
    local_track_id: str
    state: BehaviorState
    confidence: float
    frame_number: int
    timestamp: datetime
    notes: Optional[str] = None


class BehaviorStateMachine:
    """Drives per-person behaviour state transitions.

    `FACE_REGION` is a coarse spatial zone used for hand/product proximity only;
    no facial recognition or face identification is performed.
    """

    def __init__(self, global_id: str) -> None:
        self.global_id = global_id
        self.state: BehaviorState = BehaviorState.IDLE
        self.history: List[BehaviorObservation] = []

    def transition(self, observation: BehaviorObservation) -> BehaviorState:
        """Apply an observation and return the resulting state."""
        raise NotImplementedError

    def can_transition(self, target: BehaviorState) -> bool:
        """Whether a transition from the current state to `target` is legal."""
        raise NotImplementedError

    def reset(self) -> None:
        """Return the machine to IDLE and clear history."""
        raise NotImplementedError

    def evaluate(self, context: Any) -> Optional[BehaviorObservation]:
        """Derive an observation from pipeline context (tracks, detections)."""
        raise NotImplementedError
