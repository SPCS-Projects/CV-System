# CV System  
**Real-Time Computer Vision System**

---

## Overview
CV System is a lightweight, edge-optimized vision project built for the **Luxonis OAK-D** camera.  
It performs **real-time face detection** and **face re-identification** directly on the device — keeping inference off the host for low-latency, efficient performance.

This version focuses only on **detection** and **re-identification**.  
Demographic classification and testing/logging modules from earlier builds were intentionally removed for clarity and reliability.

---

## Objectives
| # | Goal | Description |
|---|------|-------------|
| 1 | **Real-Time Detection** | Identify when a face enters the frame with minimal latency. |
| 2 | **Re-Identification** | Match detected faces against previous embeddings to determine if they’ve been seen before. |
| 3 | **Edge Execution** | Offload model inference to the OAK-D’s Myriad X VPU. |
| 4 | **Simplicity & Stability** | Keep dependencies light and code easy to maintain. |

---

## Technologies

### Hardware
- **Luxonis OAK-D** — powered by the *Movidius Myriad X VPU*  
  Runs both the detection and embedding networks directly on the device.

### Software Stack
| Component | Purpose |
|------------|----------|
| **Python 3** | Core runtime |
| **OpenCV** | Video capture and visualization |
| **DepthAI** | Communication and model inference on OAK-D |
| **SQLite** | Local database for persistent face records |
| **NumPy** | Embedding operations and similarity matching |

---

## Models Used
| Model | Function | Runs On | File |
|--------|-----------|---------|------|
| **RetinaFace** | Face detection | OAK-D (VPU) | `blobs/Retinaface-720x1280.blob` |
| **FaceNet** | Face embeddings (512-D) | OAK-D (VPU) | `blobs/Facenet.blob` |

Both models are compiled as `.blob` files for the OAK-D device.

---

## Folder Structure
```
CV_System/
├── main.py
├── db_simple.py
├── requirements.txt
├── blobs/
│   ├── Retinaface-720x1280.blob
│   └── Facenet.blob
├── Retinaface/
│   ├── cv_retinaface.py
│   └── cv_imgprocess.py
├── faces/       # auto-created
├── images/      # auto-created
└── database/    # auto-created
```

---

## Setup & Usage
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
Press **Q** to quit the live window.

---

## Database Schema
The system writes detections to an SQLite database (`database/YYYY-MM-DD.db`).

| Column | Description |
|--------|-------------|
| **ID** | Autoincrement primary key |
| **uID** | Unique identifier per person |
| **EMBEDDINGS** | 512-float32 FaceNet vector |
| **IMG** | Cropped face image path |
| **FULL_IMG** | Full frame path |
| **FIRST_SEEN / LAST_SEEN** | UTC timestamps |
| **TIME_RANGES** | JSON list of [start, end] appearances |
| **TOTAL_TIME** | Total time visible in seconds |

---

## Performance
- Full inference (RetinaFace + FaceNet) runs on the **OAK-D**.  
- Host handles:
  - Frame display (`cv2.imshow`)
  - Database writes
  - Label drawing  
- Typical latency: **~30–40 ms per frame** at 1080p.

---

## Requirements
```
opencv-python
numpy
depthai
```

---

## Notes
- Default preview resolution: **1280×720** (matching the RetinaFace blob).  
  If your blob expects the opposite orientation (720×1280), change the preview size in `main.py`.
- If you add more models, drop them in `blobs/` and reference them in the pipeline.

---

**Author:** Sahven Patel  
*(Original CV System concept & implementation)*
