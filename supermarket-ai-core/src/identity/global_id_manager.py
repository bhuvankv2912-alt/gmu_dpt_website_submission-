"""Cross-camera global identity interface (M5 placeholder)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.reid.reid_model import PersonEmbedding


class GlobalIDState(str, Enum):
    """Lifecycle of an anonymous global identity."""

    ACTIVE = "active"           # currently visible on some camera
    SEARCHING = "searching"     # left a camera, awaiting reappearance downstream
    EXPIRED = "expired"         # not re-seen within GLOBAL_ID_TIMEOUT


class TrajectoryPoint(BaseModel):
    """One position sample in an anonymous journey."""

    camera_id: str
    local_track_id: str
    bbox: List[float] = Field(min_length=4, max_length=4)
    frame_number: int
    timestamp: datetime


class CameraVisit(BaseModel):
    """A contiguous appearance of a person on one camera."""

    camera_id: str
    local_track_id: str
    entered_at: datetime
    left_at: Optional[datetime] = None


class GlobalPerson(BaseModel):
    """An anonymous person tracked across cameras. No PII, no facial data."""

    global_id: str
    state: GlobalIDState = GlobalIDState.ACTIVE
    current_camera: Optional[str] = None
    current_local_track_id: Optional[str] = None
    first_seen: datetime
    last_seen: datetime
    camera_history: List[CameraVisit] = Field(default_factory=list)
    trajectory: List[TrajectoryPoint] = Field(default_factory=list)
    embeddings: List[PersonEmbedding] = Field(default_factory=list)
    events: List[str] = Field(default_factory=list, description="Event ids")


class GlobalIDManager:
    """Links camera-local tracks into anonymous cross-camera identities.

    Responsibilities (M5):
      * assign a new `global_id` to an unmatched local track
      * when a track disappears, move its person to `SEARCHING`
      * match a new track against `SEARCHING` people using Re-ID similarity,
        restricted to cameras reachable via `next_cameras` topology and within
        `GLOBAL_ID_SEARCH_TIMEOUT`
      * expire identities after `GLOBAL_ID_TIMEOUT`
    """

    def __init__(
        self,
        topology: Dict[str, Any],
        similarity_threshold: float,
        search_timeout: int,
        id_timeout: int,
    ) -> None:
        self.topology = topology
        self.similarity_threshold = similarity_threshold
        self.search_timeout = search_timeout
        self.id_timeout = id_timeout
        self.people: Dict[str, GlobalPerson] = {}

    def assign(self, embedding: PersonEmbedding) -> GlobalPerson:
        """Return the matching global person, creating one when no match exists."""
        raise NotImplementedError

    def mark_searching(self, global_id: str) -> None:
        """Move a person to the SEARCHING state after leaving a camera."""
        raise NotImplementedError

    def candidates_for(self, camera_id: str) -> List[GlobalPerson]:
        """SEARCHING people who could plausibly appear on `camera_id`."""
        raise NotImplementedError

    def expire_stale(self, now: Optional[datetime] = None) -> List[str]:
        """Expire identities older than `GLOBAL_ID_TIMEOUT`; return their ids."""
        raise NotImplementedError

    def journey(self, global_id: str) -> List[CameraVisit]:
        """Ordered camera history for one anonymous person."""
        raise NotImplementedError
