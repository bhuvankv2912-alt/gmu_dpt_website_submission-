# Sample Videos

Drop recorded video files here. M1 reads `.mp4`, `.avi` and `.mov` only — no webcam, no RTSP.

```
videos/cam1.mp4    # CAM1 - Entrance
videos/cam2.mp4    # CAM2 - Aisle A
videos/cam3.mp4    # CAM3 - Aisle B
```

`config/cameras.yaml` points at these paths by default, so `python -m src.main --camera CAM2`
picks up `videos/cam2.mp4`. Video files are gitignored — only this README is tracked.
