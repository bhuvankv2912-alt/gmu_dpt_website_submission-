# Architecture

## Design Goals

1. **Anonymous by construction** — no facial recognition, no identity linkage; only ephemeral
   IDs and appearance embeddings.
2. **Human-in-the-loop** — the engine produces *potential* events for review, never verdicts.
3. **Swappable components** — detection, tracking, Re-ID, and storage sit behind interfaces so
   implementations (YOLO variant, ByteTrack vs BoT-SORT, SQLite vs PostgreSQL) can change without
   touching callers.
4. **Config-driven** — every threshold and topology fact lives in `config/`.

## Pipeline

```
VideoManager -> YOLODetector -> Tracker -> ReIDModel -> GlobalIDManager
                                                 |
                                    BehaviorStateMachine
                                     /                  \
                        ConcealmentDetector      ConsumptionDetector
                                     \                  /
                                        EventEngine
                                             |
                                         Database
                                             |
                                      FastAPI (read-only)
```

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `src/config_loader.py` | Load and validate `config.yaml` / `cameras.yaml`. Single source of tunables. |
| `src/logging_setup.py` | Uniform logging format and level configuration. |
| `src/video/source.py` | `VideoSource`: one stream (file / webcam / RTSP), FPS and resolution control, graceful failure, clean shutdown. |
| `src/video/manager.py` | `VideoManager`: owns many `VideoSource` objects, yields per-camera frames, handles reconnection. |
| `src/detection/yolo_detector.py` | `YOLODetector`: frame -> `Detection` list (person, product, cart, ...). |
| `src/tracking/tracker.py` | `Tracker`: per-camera association producing local track IDs such as `C1_07`. |
| `src/reid/reid_model.py` | `ReIDModel`: person crop -> L2-normalized embedding; cosine similarity comparison. |
| `src/identity/global_id_manager.py` | `GlobalIDManager`: links local tracks into a `GlobalPerson`; SEARCHING lifecycle when a person leaves a camera; topology-aware candidate filtering. |
| `src/behavior/state_machine.py` | Behaviour state transitions (IDLE -> ... -> POTENTIAL_CONCEALMENT). |
| `src/behavior/concealment.py` | Heuristics for product disappearing into a concealment zone. |
| `src/behavior/consumption.py` | Heuristics for product moving to the face region. |
| `src/events/event_engine.py` | Turns behaviour outcomes into `Event`s with evidence and review status. |
| `src/database/database.py` | Persistence abstraction for persons, cameras, journeys, events. |
| `src/api/main.py` | Read-only FastAPI surface. Contains no AI code. |
| `src/main.py` | CLI demo entrypoint. |

## Identity Lifecycle

```
local track appears
      |
   new? --yes--> create GlobalPerson (ACTIVE)
      |
      no -> match embedding against SEARCHING candidates
            reachable via cameras.yaml topology and within
            GLOBAL_ID_SEARCH_TIMEOUT, using REID_SIMILARITY_THRESHOLD
      |
track lost -> GlobalPerson -> SEARCHING
      |
no re-appearance within GLOBAL_ID_TIMEOUT -> EXPIRED
```

## Event Semantics

Events are emitted with `status = NEW` and are only ever moved to `UNDER_REVIEW`, `VERIFIED`, or
`DISMISSED` by a human reviewer. `VERIFIED` is never set automatically by the engine.

## Separation Of Concerns

The API layer reads persisted state only; it never invokes detection, tracking, or Re-ID directly.
This allows the AI worker and the API service to be deployed and scaled independently.
