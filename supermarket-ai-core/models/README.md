# Model Weights

Place pretrained weights here. **Everything in this directory except this README is gitignored**
(`*.pt`, `*.pth`, `*.onnx`), so weights are never committed.

## YOLO detection (used by M1)

Download an Ultralytics YOLO checkpoint and place it here, e.g.:

```
models/yolov8n.pt      # fast, for laptops / demos
models/yolov8m.pt      # better accuracy
```

Point `MODEL.WEIGHTS` in `config/config.yaml` at the file, or pass `--weights models/yolov8n.pt`.
If you give a bare name (`yolov8n.pt`) ultralytics will download it on first use, which requires
network access; a local file under `models/` avoids that.

## Re-ID (later milestone)

Place the OSNet / Torchreid appearance-embedding checkpoint here, e.g.:

```
models/osnet_x1_0_market1501.pth
```

Paths are resolved from configuration, not hard-coded in source.
