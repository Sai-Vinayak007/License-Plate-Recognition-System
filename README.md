# 🚘 Indian License Plate Recognition System

An end-to-end deep learning pipeline that detects and reads Indian vehicle license plates from images using **YOLOv8** for plate detection and **EasyOCR** for text recognition.

---

## 📌 Overview

This project combines object detection and optical character recognition (OCR) to automatically extract license plate numbers from vehicle images. It is trained on an Indian license plate dataset sourced from Kaggle and is designed to run on **Google Colab**.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Object Detection | [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) |
| OCR Engine | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| Dataset | [Kaggle – Indian License Plates with Labels](https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels) |
| Environment | Google Colab (GPU recommended) |
| Language | Python 3 |

---

## 📁 Project Structure

```
├── license_plate_recognition.py   # Main pipeline script
├── license_plate.yaml             # YOLOv8 dataset config
├── recognition_results.csv        # Output: detected plates + confidence scores
├── datasets/
│   └── license_plates/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
└── runs/
    └── detect/
        └── lp_detector/           # Training outputs (weights, metrics, plots)
```

---

## ⚙️ Pipeline

```
Raw Images
    │
    ▼
Train / Val Split (80/20, shuffled)
    │
    ▼
YOLOv8s Training (50 epochs, 640px, batch 16)
    │
    ▼
Plate Detection (confidence ≥ 0.50)
    │
    ▼
Crop Detected Region
    │
    ▼
EasyOCR Text Extraction
    │
    ▼
Post-processing (regex clean + uppercase format)
    │
    ▼
Results saved to CSV + annotated image display
```

---

## 🚀 Getting Started

### Prerequisites

- Google Colab account
- Kaggle API key (`kaggle.json`)
- GPU runtime recommended (`Runtime > Change runtime type > T4 GPU`)

### Steps

1. Upload your `kaggle.json` to the Colab session.
2. Run the script top-to-bottom — it handles dataset download, training, and inference automatically.
3. After training, upload any vehicle image in the final cell to test the model on your own photo.

---

## 📊 Model Details

| Parameter | Value |
|---|---|
| Base model | `yolov8s.pt` (small) |
| Input size | 640 × 640 |
| Epochs | 50 |
| Batch size | 16 |
| Train/Val split | 80 / 20 |
| Confidence threshold | 0.50 |
| Classes | 1 (`license_plate`) |

---

## 📈 Output Metrics

After training, the following metrics are reported:

- **mAP50** — Mean Average Precision at IoU 0.50
- **mAP50-95** — Mean Average Precision across IoU thresholds 0.50–0.95
- **mAP75** — Mean Average Precision at IoU 0.75

Results and confusion matrix plots are saved to `runs/detect/lp_detector/`.

---

## 📝 Output CSV

All predictions on the validation set are saved to `recognition_results.csv`:

| image | confidence | plate_text |
|---|---|---|
| car_001.jpg | 0.9312 | KA 05 AB 1234 |
| car_002.jpg | 0.8741 | MH 12 CD 5678 |

---

## 🔧 Key Design Choices

- **Shuffled split**: Images are randomly shuffled before the 80/20 train/val split to avoid ordering bias.
- **Confidence filtering**: Only detections with confidence ≥ 0.50 are passed to OCR, reducing false positives.
- **Post-processing**: OCR output is cleaned with regex — non-alphanumeric characters are stripped and text is uppercased to match the Indian plate format.
- **Error handling**: Each image is wrapped in a `try/except` block so a single corrupt file does not crash the pipeline.
- **Single validation call**: `model.val()` is called exactly once to avoid redundant computation.

---

## 📌 Limitations

- Performance may vary on heavily occluded, blurry, or night-time images.
- EasyOCR can occasionally misread characters that look similar (e.g. `0` vs `O`, `1` vs `I`).
- The model is trained specifically on Indian plates and may not generalise well to other regions.

---

## 🔮 Future Improvements

- [ ] Try `PaddleOCR` as an alternative to EasyOCR for better accuracy on Indian plates
- [ ] Add a Gradio or Streamlit web interface for easy demos
- [ ] Experiment with larger YOLOv8 variants (`yolov8m`, `yolov8l`) for higher mAP
- [ ] Train on a larger, more diverse dataset including night-time and low-resolution images
- [ ] Add character-level correction using known Indian plate format patterns

---

## 📄 License

This project is for educational purposes. The dataset is sourced from Kaggle and subject to its respective license.