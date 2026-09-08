"""Person re-identification interface (M4 placeholder).

Re-ID here is *appearance-based only* (clothing, body shape, colour). It performs
no facial recognition and produces no biometric identity.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class PersonEmbedding(BaseModel):
    """A normalized appearance embedding for one person crop."""

    camera_id: str
    local_track_id: str
    embedding: List[float] = Field(description="L2-normalized appearance vector")
    frame_number: int
    timestamp: datetime
    quality: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ReIDModel:
    """Maps a person crop to a normalized embedding.

    Responsibilities (M4):
      * load an OSNet / Torchreid checkpoint from `models/`
      * turn a cropped person image into an L2-normalized vector
      * compare embeddings by cosine similarity against
        `REID_SIMILARITY_THRESHOLD` (configurable, default 0.70)
    """

    def __init__(
        self,
        weights_path: str,
        similarity_threshold: float,
        device: Optional[str] = None,
    ) -> None:
        self.weights_path = weights_path
        self.similarity_threshold = similarity_threshold
        self.device = device

    def load(self) -> None:
        """Load Re-ID weights into memory."""
        raise NotImplementedError

    def extract(
        self,
        crop: Any,
        camera_id: str,
        local_track_id: str,
        frame_number: int,
    ) -> PersonEmbedding:
        """Compute a normalized embedding for a person crop."""
        raise NotImplementedError

    def similarity(self, a: PersonEmbedding, b: PersonEmbedding) -> float:
        """Cosine similarity between two embeddings."""
        raise NotImplementedError

    def is_match(self, a: PersonEmbedding, b: PersonEmbedding) -> bool:
        """Whether similarity meets the configured threshold."""
        raise NotImplementedError
