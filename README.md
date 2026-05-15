# License Plate Recognition System

A two-stage pipeline that detects and reads vehicle license plates from images using YOLOv8 and an OCR model. Primarily trained on Indian license plates.

---

## How It Works

1. **Detection** — YOLOv8n locates the license plate in the image
2. **Recognition** — An OCR model reads the plate text

---

## Model Performance

| Metric    | Value |
|-----------|-------|
| mAP50     | 0.976 |
| mAP50-95  | 0.841 |
| Precision | 0.975 |
| Recall    | 0.973 |

Trained for 50 epochs on ~417 images in ~35 minutes on an RTX 3050 Laptop GPU.

---

## Setup

```bash
pip install ultralytics
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Training

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="datasets/license_plates/data.yaml", epochs=50, imgsz=416, device=0)
```

## Inference

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model("path/to/image.jpg")
results[0].show()
```

---

## Known Issues

- OCR may pick up surrounding text like dealer stickers or the "IND" emblem
- Low-confidence results on blurry or angled plates

---

## Requirements

- Python 3.11+
- PyTorch 2.5.1 + CUDA 12.1
- Ultralytics 8.4.46