"""Storage abstraction (M10 placeholder).

Deliberately backend-agnostic: the default implementation will use SQLite, but
the interface must permit swapping in PostgreSQL without changing callers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.events.event_engine import Event
from src.identity.global_id_manager import CameraVisit, GlobalPerson


class Database:
    """Persistence interface for persons, cameras, journeys and events."""

    def __init__(self, connection_string: str) -> None:
        self.connection_string = connection_string

    def connect(self) -> None:
        """Open the connection and ensure the schema exists."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the connection cleanly."""
        raise NotImplementedError

    def init_schema(self) -> None:
        """Create tables if they do not exist."""
        raise NotImplementedError

    # -- cameras -----------------------------------------------------------
    def upsert_camera(self, camera_id: str, metadata: Dict[str, Any]) -> None:
        """Store or update a camera record."""
        raise NotImplementedError

    def list_cameras(self) -> List[Dict[str, Any]]:
        """Return all known cameras."""
        raise NotImplementedError

    # -- persons -----------------------------------------------------------
    def save_person(self, person: GlobalPerson) -> None:
        """Insert or update an anonymous global person."""
        raise NotImplementedError

    def get_person(self, global_id: str) -> Optional[GlobalPerson]:
        """Fetch one anonymous global person."""
        raise NotImplementedError

    # -- journeys ----------------------------------------------------------
    def save_journey(self, global_id: str, visits: List[CameraVisit]) -> None:
        """Persist a person's camera journey."""
        raise NotImplementedError

    def get_journey(self, global_id: str) -> List[CameraVisit]:
        """Fetch a person's camera journey."""
        raise NotImplementedError

    # -- events ------------------------------------------------------------
    def save_event(self, event: Event) -> None:
        """Persist a reviewable event."""
        raise NotImplementedError

    def get_event(self, event_id: str) -> Optional[Event]:
        """Fetch one event."""
        raise NotImplementedError

    def list_events(
        self,
        status: Optional[str] = None,
        camera_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Query events with optional filters."""
        raise NotImplementedError
