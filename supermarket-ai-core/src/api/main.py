"""FastAPI application (M10 placeholder).

Route stubs only. This layer stays free of detection, tracking and Re-ID code:
it reads persisted state so the API can be deployed separately from the AI worker.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Supermarket AI Core Engine API",
    description=(
        "Read-only review and monitoring API. All person identifiers are anonymous; "
        "no facial recognition is performed and events are review candidates only."
    ),
    version="0.1.0",
)

NOT_IMPLEMENTED = 501


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness probe. Implemented so deployments can be smoke-tested early."""
    return {"status": "ok", "milestone": "M1", "api_implemented": False}


@app.get("/cameras")
def list_cameras() -> Dict[str, Any]:
    """List configured cameras and their topology."""
    raise HTTPException(status_code=NOT_IMPLEMENTED, detail="Not implemented (M10)")


@app.get("/events")
def list_events() -> Dict[str, Any]:
    """List reviewable events."""
    raise HTTPException(status_code=NOT_IMPLEMENTED, detail="Not implemented (M10)")


@app.get("/events/{event_id}")
def get_event(event_id: str) -> Dict[str, Any]:
    """Fetch a single event and its evidence."""
    raise HTTPException(status_code=NOT_IMPLEMENTED, detail="Not implemented (M10)")


@app.get("/persons/{global_id}")
def get_person(global_id: str) -> Dict[str, Any]:
    """Fetch one anonymous person record."""
    raise HTTPException(status_code=NOT_IMPLEMENTED, detail="Not implemented (M10)")


@app.get("/persons/{global_id}/journey")
def get_person_journey(global_id: str) -> Dict[str, Any]:
    """Fetch the cross-camera journey for one anonymous person."""
    raise HTTPException(status_code=NOT_IMPLEMENTED, detail="Not implemented (M10)")


@app.get("/system/status")
def system_status() -> Dict[str, Any]:
    """Report pipeline health, active cameras and processing rates."""
    raise HTTPException(status_code=NOT_IMPLEMENTED, detail="Not implemented (M10)")
