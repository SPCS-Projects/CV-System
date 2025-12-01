# CV System

A lightweight, on-device face detection + re-identification pipeline built around the **Luxonis OAK‑D**.

---

## Overview

- **Device NN:** RetinaFace (detection) and FaceNet (embedding) run on the OAK‑D via DepthAI.
- **Host tasks:** frame capture, simple preprocessing, saving images, and database writes.
- **Database:** per‑day SQLite DB with entries for face embeddings, image paths, and timing info.
- **No demographics:** the Race/Age/Gender path is disabled by default in this build.

Key runtime logic lives in `cv_start.py` (camera + pipeline + main loop) and `cv_db.sql(...)` (DB insert/update).

---

## Project Structure

```
CV_System/
├── cv_start.py           # main entrypoint (camera, NNs, loop, saving, DB calls)
├── cv_db.py              # sqlite insert/update (face_meta table, re-id logic)
├── cv_attributes.py      # A handler file for the attributes model
├── cv_parsedb.py         # Exports and neatly sorts data from the database/saved image captures
├── requirements.txt      # pinned libs (DepthAI, TF, etc.)
├── Retinaface/
│   ├── cv_retinaface.py
│   └── cv_imgprocess.py
├── blobs/
│   ├── Retinaface-720x1280.blob
│   └── Facenet.blob
├── images/               # auto-saved full-frame PNGs (timestamped)
├── faces/                # auto-saved cropped faces (timestamped)
└── database/             # per-day SQLite DB files (YYYY-MM-DD.db)
```

Runtime/loop details are in `cv_start.py`. It builds a DepthAI **pipeline** with:
- `ColorCamera` at **1280×720**,
- device **NeuralNetwork** nodes for RetinaFace + FaceNet,
- **XLinkIn/XLinkOut** for host ↔ device IO

---

## Libraries

From `requirements.txt`
```
opencv-python
depthai
numpy
Pillow
tensorflow
matplotlib
pandas
tqdm
ntplib
```

---

## Setup

```bash
# 1) Create and activate a venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# 2) Install pinned dependencies
pip install -r requirements.txt

# 3) Place model blobs
#   blobs/Retinaface-720x1280.blob
#   blobs/Facenet.blob
```

---

## Running

```bash
python cv_start.py
```

- On start, the app waits briefly for camera/queues, then enters the frame loop.  
- Each iteration:
  1. Grabs a frame from the OAK‑D (`ColorCamera.video`), optionally flips it.
  2. Saves a timestamped PNG to `images/` when someone was detected (currently being done for testing purposes).
  3. Preprocesses and **sends** it to the RetinaFace network via `XLinkIn`. The code uses explicit **layer names** to read tensors back and reorders them to match RetinaFace’s expected shapes.
  4. Runs `Retinaface.get_data(...)` to produce face crops (and currently has framework for an age/gender tensor set that is currently unused).
  5. For each crop, sends a 160×160 input to **FaceNet** and reads an embedding as a NumPy array.
  6. Calls `cv_db.sql(...)` to **insert/update** the `face_meta` table, save the crop into `faces/`, and update timing fields.

Timestamps use NTP (`ntplib`) with local fallback. Crashes are logged to `log.txt` with a UTC/local timestamp and last runtime. Multiprocessing is used to launch `cv_main(...)` and monitor its health.

---

## Database

`cv_db.sql(...)` ensures a per‑day DB exists in `database/YYYY-MM-DD.db`. When empty, it creates `face_meta`:

| Column | Description |
|---|---|
| `ID` | integer PK (manual counter) |
| `uID` | unique identifier (50‑char) |
| `EMBEDDING` | FaceNet vector as `float32` bytes |
| `IMG` | JSON‑string list of face crop paths |
| `FULL_IMG` | JSON‑string list of full image paths |
| `FIRST_SEEN`, `LAST_SEEN` | timestamps (string) |
| `TIME_RANGES` | JSON‑string list of `[start, end]` ranges |
| `TOTAL_TIME` | accumulated seconds |
| `AGE`, `GENDER`, `RACE` | strings (currently `"empty"` in this build) |

Matching uses **Euclidean distance**; new entries are inserted if no existing embedding is within a threshold (≤ 10 by default). When a known face reappears, image paths append, `LAST_SEEN` updates, and **time ranges** cache after inactivity (> 600s), rolling into `TOTAL_TIME`.

Backups of the current daily DB are copied into `db_backup/` after each write.

---

## Configuration Notes

- **Resolution:** Camera video is set to **1280×720**; the RetinaFace blob name indicates expected layout `720×1280` tensors. Keep these consistent with your blob.
- **Flip:** Set `flip=True` in `cv_main(...)` for a vertical flip if required.
- **Layer order:** The detection layer names are listed explicitly and then **reordered** (`idx = [7,1,4,8,0,3,6,2,5]`) before reshaping to the correct 4D tensors. If you swap blobs, update this map.
- **Disabled attributes:** The Race/Age/Gender network currently only has framework set up and is commented out. `stats = ["empty","empty","empty"]` is passed to DB. Needs to be edited after adding attributes properly.

---

## Troubleshooting

- **No detections:** Ensure `blobs/Retinaface-720x1280.blob` matches the 1280×720 video feed. If using a different blob, update the **layer name list** and `idx` ordering.
- **DB not updating:** Check write permissions for `database/` and `db_backup/`, and confirm face crops are being generated; `cv_db.sql(...)` writes a crop per detection before the insert/update.

---

## Run Preview (optional)

If you want a live preview window of what the camera sees, uncomments the following lines inside the main loop in `cv_start.py` just after `image = frame`:

```python
cv2.imshow("CV System - preview", image)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```
(And `cv2.destroyAllWindows()` after the loop.)

---

## License

For internal / educational use.
