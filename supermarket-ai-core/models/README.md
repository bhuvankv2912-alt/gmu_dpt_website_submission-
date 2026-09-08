# Model Weights

Place pretrained weights here. **Everything in this directory except this README is gitignored**
(`*.pt`, `*.pth`, `*.onnx`), so weights are never committed.

## YOLO detection

Download an Ultralytics YOLO checkpoint and place it here, e.g.:

```
models/yolov8n.pt      # fast, for laptops / demos
models/yolov8m.pt      # better accuracy
```

## Re-ID

Place the OSNet / Torchreid appearance-embedding checkpoint here, e.g.:

```
models/osnet_x1_0_market1501.pth
```

Paths are resolved from configuration, not hard-coded in source.
