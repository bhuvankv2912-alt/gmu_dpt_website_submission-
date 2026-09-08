# AI-Powered Supermarket Loss Prevention & Store Intelligence — AI CORE ENGINE

Multi-camera, privacy-preserving computer vision engine that tracks anonymous shoppers across a
store, links their local per-camera tracks into a single anonymous global identity, and flags
*potential* loss-prevention events for human review.

> **Status: Milestone M0 (Architecture & Setup) complete.** All AI modules in `src/` are
> interfaces/placeholders that raise `NotImplementedError`. Milestones M1-M12 are pending.

## Objective

Give store operators a single, anonymous, cross-camera view of shopper journeys and surface
behaviours that *may* indicate concealment or in-store consumption, always as review candidates —
never as accusations or automated decisions.

## Privacy Principles

- **No facial recognition.** Faces are never enrolled, encoded for identification, or matched
  against any watchlist. The face region is only used as a coarse spatial zone for behaviour
  heuristics (e.g. hand-to-face proximity).
- **Anonymous IDs only.** People are represented by ephemeral identifiers (`GP_000123`) and
  appearance embeddings. No names, no demographic inference, no external identity linkage.
- **POTENTIAL / `requires_review` terminology.** The engine emits `potential_concealment` and
  `potential_in_store_consumption` events with status `NEW` / `UNDER_REVIEW`. The engine must
  **never** auto-set `VERIFIED`; only a human reviewer can.
- **Data minimisation.** Embeddings and evidence are retained only as long as configured, and
  identities expire after `GLOBAL_ID_TIMEOUT`.

## Architecture Pipeline

```
Video sources (file / webcam / RTSP)
        |
   VideoManager  ->  per-camera frames
        |
   YOLODetector  ->  Detection(person, product, ...)
        |
   Tracker       ->  local track IDs (C1_07)
        |
   ReIDModel     ->  normalized appearance embeddings
        |
 GlobalIDManager ->  GlobalPerson (cross-camera anonymous identity, SEARCHING lifecycle)
        |
 Behavior layer  ->  state machine + concealment / consumption detectors
        |
   EventEngine   ->  Event(status=NEW, requires_review)
        |
    Database     ->  persons, cameras, journeys, events
        |
   FastAPI       ->  read-only review & monitoring API
```

See [`docs/architecture.md`](docs/architecture.md) for module responsibilities.

## Tech Stack

| Concern | Choice |
| --- | --- |
| Language | Python 3.10+ |
| Video I/O | OpenCV (`opencv-python`) |
| Detection | Ultralytics YOLO |
| Tracking | BoT-SORT / ByteTrack (via Ultralytics) |
| Deep learning | PyTorch / torchvision |
| Re-ID | OSNet via Torchreid |
| Numerics | NumPy |
| API | FastAPI + Uvicorn |
| Schemas / config validation | Pydantic |
| Config files | PyYAML |
| Storage | SQLite (abstracted so PostgreSQL can be swapped in) |

## Installation

```bash
cd supermarket-ai-core
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place model weights in `models/` and sample videos in `videos/` (both gitignored — see the
READMEs in those directories).

## How To Run The (Future) Demo

The CLI entrypoint exists but the pipeline is not implemented yet; it currently loads config,
sets up logging, and reports that processing is unavailable.

```bash
python -m src.main --source videos/cam1.mp4 --camera CAM1
```

The API can be started (all routes are stubs returning `501` except `/health`):

```bash
uvicorn src.api.main:app --reload
```

## Configuration

All tunable values live in `config/config.yaml`; camera topology lives in `config/cameras.yaml`.
No thresholds are hard-coded in `src/`.

```python
from src.config_loader import load_config, load_cameras
```

## Testing

```bash
pytest
```

Tests for `config_loader` and `logging_setup` are real; the AI-module tests are
`pytest.mark.skip` placeholders documenting what later milestones must verify.

## Milestone Roadmap

| Milestone | Scope | Status |
| --- | --- | --- |
| M0 | Architecture & setup: structure, config, logging, interfaces | **Complete** |
| M1 | Video input layer (file / webcam / RTSP, multi-source manager) | Pending |
| M2 | YOLO detection integration | Pending |
| M3 | Single-camera tracking with local track IDs | Pending |
| M4 | Re-ID embedding extraction | Pending |
| M5 | Global ID matching across cameras + SEARCHING lifecycle | Pending |
| M6 | Behaviour state machine | Pending |
| M7 | Potential concealment detection | Pending |
| M8 | Potential in-store consumption detection | Pending |
| M9 | Event engine + evidence capture | Pending |
| M10 | Database persistence layer | Pending |
| M11 | FastAPI endpoints | Pending |
| M12 | End-to-end multi-camera demo + evaluation | Pending |
