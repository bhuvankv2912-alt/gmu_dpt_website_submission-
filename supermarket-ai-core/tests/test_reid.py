"""Placeholder tests for Re-ID embeddings (M3)."""

import pytest


@pytest.mark.skip(reason="M3: ReIDModel not implemented yet")
def test_embedding_is_l2_normalized():
    """extract() should return a unit-norm embedding."""


@pytest.mark.skip(reason="M3: ReIDModel not implemented yet")
def test_same_person_similarity_above_threshold():
    """Two crops of one person should exceed REID_SIMILARITY_THRESHOLD (0.70)."""


@pytest.mark.skip(reason="M3: ReIDModel not implemented yet")
def test_different_people_similarity_below_threshold():
    """Crops of different people should fall below the threshold."""


@pytest.mark.skip(reason="M3: ReIDModel not implemented yet")
def test_reid_performs_no_face_matching():
    """Embeddings must be appearance-based only; no facial recognition."""
