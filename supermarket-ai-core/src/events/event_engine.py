"""Event engine interface and event schema (M9 placeholder)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Kinds of event the engine can emit."""

    POTENTIAL_CONCEALMENT = "potential_concealment"
    POTENTIAL_IN_STORE_CONSUMPTION = "potential_in_store_consumption"
    PERSON_ENTERED = "person_entered"
    PERSON_LEFT = "person_left"
    CAMERA_TRANSITION = "camera_transition"


class EventStatus(str, Enum):
    """Human review status.

    The engine only ever creates events with `NEW`. `VERIFIED` must never be set
    automatically — only a human reviewer may verify or dismiss an event.
    """

    NEW = "NEW"
    UNDER_REVIEW = "UNDER_REVIEW"
    VERIFIED = "VERIFIED"
    DISMISSED = "DISMISSED"


class Event(BaseModel):
    """A reviewable observation. Never a verdict."""

    event_id: str
    event_type: EventType
    global_id: str
    camera_id: str
    local_track_id: str
    timestamp: datetime
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Frame numbers, crop paths, state trace, thresholds used.",
    )
    status: EventStatus = EventStatus.NEW


class EventEngine:
    """Converts behaviour observations into reviewable events.

    Responsibilities (M9):
      * apply `EVENT_CONFIDENCE_THRESHOLD`
      * de-duplicate repeated observations for the same person and behaviour
      * attach evidence
      * emit events with `status = NEW` only; the engine must never set
        `VERIFIED` (or any other reviewer-owned status) itself
    """

    def __init__(self, confidence_threshold: float) -> None:
        self.confidence_threshold = confidence_threshold

    def emit(self, observation: Any) -> Optional[Event]:
        """Create an event from a behaviour observation, if it qualifies."""
        raise NotImplementedError

    def recent(self, global_id: str) -> List[Event]:
        """Recent events for one anonymous person."""
        raise NotImplementedError
