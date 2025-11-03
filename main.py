import os
import cv2
import numpy as np
import depthai as dai
from pathlib import Path
from datetime import datetime
from db import insert_face

BASE = Path(".")
(FACES := BASE/"faces").mkdir(exist_ok=True, parents=True)
(IMAGES := BASE/"images").mkdir(exist_ok=True, parents=True)
(DBDIR := BASE/"database").mkdir(exist_ok=True, parents=True)

# ---------------------------
# helpers
# ---------------------------


def planar_from_bgr(img):
    b, g, r = cv2.split(img)
    return b.tobytes() + g.tobytes() + r.tobytes()


def normalize(v):
    v = v.astype("float32")
    n = np.linalg.norm(v) + 1e-8
    return v / n


def _nn_first_fp32(pkt):
    # Try by layer name(s)
    try:
        names = pkt.getAllLayerNames()
    except Exception:
        names = []
    for n in names:
        for getter in ("getLayerFp16", "getLayerFp32", "getLayerInt32"):
            if hasattr(pkt, getter):
                try:
                    arr = np.array(getattr(pkt, getter)(n), dtype=np.float32).ravel()
                    if arr.size:
                        return arr
                except Exception:
                    pass

    # Try using layer infos -> names
    if hasattr(pkt, "getAllLayers"):
        try:
            infos = pkt.getAllLayers()  # TensorInfo list (metadata)
            for info in infos:
                n = getattr(info, "name", None)
                if not n:
                    continue
                for getter in ("getLayerFp16", "getLayerFp32", "getLayerInt32"):
                    if hasattr(pkt, getter):
                        try:
                            arr = np.array(getattr(pkt, getter)(n), dtype=np.float32).ravel()
                            if arr.size:
                                return arr
                        except Exception:
                            pass
        except Exception:
            pass

    # Some bindings expose a dict-like dump
    if hasattr(pkt, "toDict"):
        try:
            d = pkt.toDict()
            for v in d.values():
                arr = np.array(v, dtype=np.float32).ravel()
                if arr.size:
                    return arr
        except Exception:
            pass

    return np.array([], dtype=np.float32)


def parse_openvino_7tuple(pkt, w, h, thr=0.5):
    data = _nn_first_fp32(pkt)
    if data.size == 0 or data.size % 7 != 0:
        return []

    dets = data.reshape(-1, 7)
    boxes = []
    for d in dets:
        conf = float(d[2])
        if conf < thr:
            continue
        x1 = int(max(0, min(1, d[3])) * w)
        y1 = int(max(0, min(1, d[4])) * h)
        x2 = int(max(0, min(1, d[5])) * w)
        y2 = int(max(0, min(1, d[6])) * h)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes

# ---------------------------
# pipeline (DepthAI v3)
# ---------------------------

def build_pipeline():
    p = dai.Pipeline()

    # Build + request an output.
    cam = p.create(dai.node.Camera).build()
    preview = cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p)

    # RetinaFace NN (device-side detection)
    nn_det = p.create(dai.node.NeuralNetwork)
    nn_det.setBlobPath("blobs/Retinaface-720x1280.blob")
    preview.link(nn_det.input)

    # Host crops -> ImageManip (resize) -> Facenet NN (device embeddings)
    manip = p.create(dai.node.ImageManip)
    manip.initialConfig.setOutputSize(160, 160, dai.ImageManipConfig.ResizeMode.STRETCH)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)  # ensure Facenet gets BGR planes

    nn_emb = p.create(dai.node.NeuralNetwork)
    nn_emb.setBlobPath("blobs/Facenet.blob")
    manip.out.link(nn_emb.input)

    # Queues:
    q_rgb   = preview.createOutputQueue()             # frames -> host
    q_det   = nn_det.out.createOutputQueue()          # detections -> host
    q_face  = manip.inputImage.createInputQueue()     # host -> manip (face crops)
    q_emb   = nn_emb.out.createOutputQueue()          # embeddings -> host

    return p, q_rgb, q_det, q_face, q_emb

# ---------------------------
# main()
# ---------------------------

def main():
    # DB path per-day (same convention)
    db_path = str(DBDIR / (datetime.utcnow().strftime("%Y-%m-%d") + ".db"))
    sim_threshold = 0.55

    pipeline, q_rgb, q_det, q_face, q_emb = build_pipeline()

    # Start pipeline
    pipeline.start()

    print("Running on DepthAI v3.x — press 'q' to quit.")
    while pipeline.isRunning():
        # Get a frame
        rgb_pkt = q_rgb.get()  # blocking
        frame = rgb_pkt.getCvFrame()
        h, w = frame.shape[:2]

        # Parse detections if available
        boxes = []
        if q_det.has():
            det_pkt = q_det.get()
            boxes = parse_openvino_7tuple(det_pkt, w, h, thr=0.5)

        # Save full frame
        ts = datetime.utcnow()
        ts_tag = ts.strftime("%Y%m%d_%H%M%S_%f")
        full_rel = f"/images/{ts_tag}.jpg"
        cv2.imwrite(str(BASE) + full_rel, frame)

        # Send crops to Facenet path
        crops_meta = []
        for (x1, y1, x2, y2) in boxes:
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            planar = planar_from_bgr(face)
            img = dai.ImgFrame()
            img.setType(dai.ImgFrame.Type.BGR888p)
            img.setWidth(face.shape[1])
            img.setHeight(face.shape[0])
            img.setData(planar)
            q_face.send(img)
            crops_meta.append((x1, y1, x2, y2, face))

        # Read Facenet embeddings (1 per crop)
        embs = []
        for _ in range(len(crops_meta)):
            pkt = q_emb.get()  # blocking, one out per one in
            vec = _nn_first_fp32(pkt)
            if vec.size == 0:
                embs.append(None)
            else:
                if vec.size != 512:
                    vec = np.pad(vec, (0, max(0, 512 - vec.size)))[:512]
                embs.append(normalize(vec))

        # insert + draw
        for (meta, emb) in zip(crops_meta, embs):
            if emb is None:
                continue
            x1, y1, x2, y2, face = meta
            face_rel = f"/faces/{ts_tag}_{x1}_{y1}.jpg"
            cv2.imwrite(str(BASE) + face_rel, face)
            uid, name, sim = insert_face(db_path, emb, face_rel, full_rel, ts=ts, threshold=sim_threshold)
            label = name if name else "id:" + uid[:6]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({sim:.2f})", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("CV System — RetinaFace + Facenet (DepthAI v3)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
