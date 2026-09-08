# Sample Videos

Drop local video files here to simulate a multi-camera store without live RTSP feeds:

```
videos/cam1.mp4    # CAM1 - Entrance
videos/cam2.mp4    # CAM2 - Aisle A
videos/cam3.mp4    # CAM3 - Aisle B
```

`config/cameras.yaml` points at these paths by default. Video files are gitignored
(`videos/*.mp4`) — only this README is tracked.
