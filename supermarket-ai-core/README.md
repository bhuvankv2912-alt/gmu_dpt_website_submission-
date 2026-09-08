# AI-Powered Supermarket Loss Prevention & Store Intelligence — AI CORE ENGINE

Multi-camera, privacy-preserving computer vision engine that (eventually) tracks anonymous shoppers
across a store, links their per-camera tracks into one anonymous global identity, and flags
*potential* loss-prevention events for human review.

> **Status: M0 (Architecture & Setup) and M1 (Video Input & Object Detection) complete.**
> M1 runs YOLO over recorded video and produces an annotated video, structured JSON detections and
> processing metrics. Tracking, Re-ID, global IDs, behaviour/event logic, database, API and live
> CCTV/RTSP are **not** implemented — those modules are still M0 interfaces raising
> `NotImplementedError`.

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

M1 stores no identities at all: it writes only per-frame class labels, boxes and confidences.

## Architecture Pipeline

```
Video sources (recorded files today; RTSP later)
        |
   VideoManager  ->  per-camera frames            [M1 - done]
        |
   YOLODetector  ->  Detection(person, product)   [M1 - done]
        |
   Tracker       ->  local track IDs (C1_07)      [M2]
        |
   ReIDModel     ->  normalized embeddings        [M3]
        |
 GlobalIDManager ->  GlobalPerson                 [M4]
        |
 Behavior layer  ->  state machine + detectors    [M5-M7]
        |
   EventEngine   ->  Event(status=NEW)            [M8]
        |
    Database     ->  persons, cameras, events     [M9]
        |
   FastAPI       ->  read-only review API         [M10]
```

See [`docs/architecture.md`](docs/architecture.md) for module responsibilities.

## Tech Stack

| Concern | Choice |
| --- | --- |
| Language | Python 3.10+ |
| Video I/O | OpenCV (`opencv-python`) |
| Detection | Ultralytics YOLO |
| Tracking | BoT-SORT / ByteTrack (later) |
| Deep learning | PyTorch / torchvision |
| Re-ID | OSNet via Torchreid (later) |
| Numerics | NumPy |
| API | FastAPI + Uvicorn (later) |
| Schemas | Pydantic |
| Config files | PyYAML |
| Storage | SQLite, abstracted for PostgreSQL (later) |

## Installation

```bash
cd supermarket-ai-core
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put a YOLO checkpoint in `models/` (e.g. `models/yolov8n.pt`) and your recorded videos in
`videos/`. Both directories are gitignored except their READMEs — no weights or footage are ever
committed.

## M1 Usage

Run detection over one recorded video:

```bash
python -m src.main --source videos/cam1.mp4 --camera CAM1 --weights models/yolov8n.pt
```

Or let the camera id resolve its source from `config/cameras.yaml`:

```bash
python -m src.main --camera CAM1
```

Useful flags (all default to `config/config.yaml`, nothing is hard-coded):

| Flag | Purpose |
| --- | --- |
| `--source` | Path to a recorded `.mp4` / `.avi` / `.mov` file |
| `--camera` | Camera id from `config/cameras.yaml` (default `CAM1`) |
| `--weights` | Override `MODEL.WEIGHTS` |
| `--conf` | Override `DETECTION_CONFIDENCE` |
| `--classes person bottle ...` | Keep only these classes (default: every class the model supports) |
| `--frame-skip` | Override `FRAME_SKIP` |
| `--max-frames` | Stop after N processed frames (smoke runs) |
| `--output-dir` | Override `OUTPUT.DIR` |
| `--no-video` / `--no-json` | Skip an artefact |
| `--device` | `cpu`, `cuda`, `cuda:0`, `mps` |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### Outputs

Written to `OUTPUT.DIR` (default `output/`, gitignored):

1. `<camera>_<video>_annotated.mp4` — every processed frame with boxes, class labels,
   confidences and a small HUD (camera, frame number, detections in frame).
2. `<camera>_<video>_detections.json` — structured results:

```json
{
  "milestone": "M1",
  "camera_id": "CAM1",
  "source": "videos/cam1.mp4",
  "generated_at": "2026-01-01T10:00:00+00:00",
  "config": {"detection_confidence": 0.45, "frame_skip": 1, "resolution": [1280, 720],
             "weights": "models/yolov8n.pt", "classes": "all"},
  "metrics": {"frames_read": 300, "frames_processed": 150, "detections": 612,
              "detections_by_class": {"person": 480, "bottle": 132},
              "elapsed_seconds": 18.2, "inference_seconds": 16.4, "fps": 8.24,
              "average_detections_per_frame": 4.08},
  "frames": [
    {"frame_number": 1, "timestamp": "2026-01-01T10:00:00+00:00",
     "detections": [{"class_name": "person", "confidence": 0.91,
                     "bbox": [12.0, 40.5, 180.2, 460.8]}]}
  ]
}
```

3. **Metrics** — the same metrics block is logged and printed at the end of every run:
   frames read, frames processed, detection count, per-class counts, elapsed time, inference time
   and processing FPS.

## Configuration

`config/config.yaml` owns every tunable: `DETECTION_CONFIDENCE`, `FRAME_SKIP`, `VIDEO_RESOLUTION`
(`ENABLED`/`width`/`height`), `MODEL` (`WEIGHTS`, `DEVICE`, `CLASSES`) and `OUTPUT`
(`DIR`, `ANNOTATED_VIDEO`, `JSON`, `VIDEO_CODEC`, `LOG_EVERY`), plus the thresholds later
milestones will use. `config/cameras.yaml` owns the camera list and topology.

## Testing

```bash
pytest
```

M1 tests cover the video source (sequential reads, frame skip, resize, metadata, missing/corrupt/
unsupported files, idempotent release), the manager (source resolution, skipping unavailable
videos), the `Detection` schema, the annotator, both writers, the pipeline end to end against a
fake detector (so no weights are needed in CI), the metrics, and CLI argument handling. Later
milestones remain `pytest.mark.skip` placeholders.

## M1 Limitations

- **Recorded video only.** Webcam and RTSP/live CCTV are rejected on purpose.
- **No temporal reasoning.** Detections are per frame; the same person in consecutive frames is
  unrelated data. Counts are detection counts, not people counts.
- **Pretrained COCO classes only.** No retail-specific classes (products, trolleys, shelves); no
  fine-tuning has been done, so small products are detected poorly.
- **Fixed resize.** Frames are resized to `VIDEO_RESOLUTION` without letterboxing, so aspect ratio
  can change; set `VIDEO_RESOLUTION.ENABLED: false` to keep native size.
- **CPU throughput is modest** (single-digit to low-double-digit FPS with `yolov8n` at 720p);
  use `FRAME_SKIP`, a smaller model or a GPU.
- **Codec availability** depends on the OpenCV build; the writer falls back through
  `mp4v` → `avc1` → `MJPG` and fails loudly if none work.
- **No tracking, Re-ID, behaviour, events, database, API or dashboard** — later milestones.

## Milestone Roadmap

| Milestone | Scope | Status |
| --- | --- | --- |
| M0 | Architecture & setup: structure, config, logging, interfaces | **Complete** |
| M1 | Video input (recorded) + YOLO detection, annotated video, JSON, metrics | **Complete** |
| M2 | Single-camera tracking (ByteTrack / BoT-SORT), local track IDs | Pending |
| M3 | Re-ID embedding extraction | Pending |
| M4 | Global ID matching across cameras + SEARCHING lifecycle | Pending |
| M5 | Behaviour state machine | Pending |
| M6 | Potential concealment detection | Pending |
| M7 | Potential in-store consumption detection | Pending |
| M8 | Event engine + evidence capture | Pending |
| M9 | Database persistence layer | Pending |
| M10 | FastAPI endpoints | Pending |
| M11 | End-to-end multi-camera demo + evaluation | Pending |
