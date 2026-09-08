"""Bounding-box rendering for the annotated output videos (M1 detection, M2 tracking)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import cv2

from src.detection.yolo_detector import Detection

if TYPE_CHECKING:  # avoids a src.tracking -> src.detection import cycle at runtime
    from src.tracking.tracker import TrackedDetection

FONT = cv2.FONT_HERSHEY_SIMPLEX
PALETTE: Sequence[Tuple[int, int, int]] = (
    (0, 200, 0),
    (0, 140, 255),
    (255, 120, 0),
    (200, 0, 200),
    (0, 215, 255),
    (180, 180, 0),
)
PERSON_COLOR: Tuple[int, int, int] = (0, 200, 0)


TRACK_PALETTE: Sequence[Tuple[int, int, int]] = (
    (0, 200, 0),
    (0, 140, 255),
    (255, 120, 0),
    (200, 0, 200),
    (0, 215, 255),
    (180, 180, 0),
    (60, 60, 255),
    (255, 200, 120),
)


def class_color(class_name: str) -> Tuple[int, int, int]:
    """Stable BGR colour per class so a class keeps one colour across frames."""
    if class_name.lower() == "person":
        return PERSON_COLOR
    return PALETTE[hash(class_name) % len(PALETTE)]


def track_color(track_number: int) -> Tuple[int, int, int]:
    """Stable BGR colour per track so one identity keeps one colour on screen."""
    return TRACK_PALETTE[(track_number - 1) % len(TRACK_PALETTE)]


def draw_detections(
    frame: Any,
    detections: List[Detection],
    overlay: Dict[str, Any] | None = None,
) -> Any:
    """Return a copy of `frame` with boxes, labels and an optional HUD drawn."""
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in detection.bbox)
        color = class_color(detection.class_name)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{detection.class_name} {detection.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        top = max(0, y1 - text_h - baseline - 2)
        cv2.rectangle(
            annotated, (x1, top), (x1 + text_w + 4, top + text_h + baseline + 2), color, -1
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, top + text_h),
            FONT,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    if overlay:
        _draw_hud(annotated, overlay)
    return annotated


def draw_tracks(
    frame: Any,
    tracks: List["TrackedDetection"],
    overlay: Dict[str, Any] | None = None,
) -> Any:
    """Return a copy of `frame` with tracked boxes labelled by track id."""
    annotated = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = (int(round(v)) for v in track.bbox)
        color = track_color(track.track_number)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{track.track_id} {track.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        top = max(0, y1 - text_h - baseline - 2)
        cv2.rectangle(
            annotated, (x1, top), (x1 + text_w + 4, top + text_h + baseline + 2), color, -1
        )
        cv2.putText(
            annotated, label, (x1 + 2, top + text_h), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    if overlay:
        _draw_hud(annotated, overlay)
    return annotated


def _draw_hud(frame: Any, overlay: Dict[str, Any]) -> None:
    text = "  ".join(f"{key}: {value}" for key, value in overlay.items())
    cv2.putText(frame, text, (10, 22), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
