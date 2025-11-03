import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np
import os

SCHEMA = """
CREATE TABLE IF NOT EXISTS face_meta (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    uID TEXT UNIQUE,
    EMBEDDINGS BLOB,
    IMG TEXT,
    FULL_IMG TEXT,
    FIRST_SEEN TEXT,
    LAST_SEEN TEXT,
    NAME TEXT,
    TIME_RANGES TEXT,
    TOTAL_TIME REAL
);
"""

def _connect(db_path):
    Path(os.path.dirname(db_path)).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(SCHEMA)
    return conn

def _cosine(a, b):
    a = a.astype("float32"); b = b.astype("float32")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def insert_face(db_path, embedding, face_img_rel, full_img_rel, ts=None, threshold=0.55):
    import secrets, json
    ts = ts or datetime.utcnow()
    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT uID, EMBEDDINGS, TIME_RANGES, NAME FROM face_meta")
    rows = cur.fetchall()
    best_uid, best_sim, best_name = None, -1.0, None
    for uid, blob, tr_json, name in rows:
        if blob is None:
            continue
        emb = np.frombuffer(blob, dtype="float32")
        sim = _cosine(embedding, emb)
        if sim > best_sim:
            best_sim, best_uid, best_name = sim, uid, name

    now_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")

    if best_sim >= threshold and best_uid is not None:
        try:
            tr = json.loads(tr_json or "[]")
        except Exception:
            tr = []
        if tr and isinstance(tr[-1], list) and len(tr[-1]) == 2:
            tr[-1][1] = now_str
        else:
            tr.append([now_str, now_str])

        # recompute total time
        total_time = 0.0
        from datetime import datetime as dt
        for start, end in tr:
            try:
                total_time += max(0.0, (dt.strptime(end, "%Y-%m-%d %H:%M:%S.%f") - dt.strptime(start, "%Y-%m-%d %H:%M:%S.%f")).total_seconds())
            except Exception:
                pass

        cur.execute(
            "UPDATE face_meta SET IMG=?, FULL_IMG=?, LAST_SEEN=?, TIME_RANGES=?, TOTAL_TIME=? WHERE uID=?",
            (face_img_rel, full_img_rel, now_str, json.dumps(tr), float(total_time), best_uid)
        )
        conn.commit()
        conn.close()
        return best_uid, best_name, best_sim

    # new face
    uid = secrets.token_urlsafe(24)
    emb_blob = embedding.astype("float32").tobytes()
    tr = [[now_str, now_str]]
    cur.execute(
        "INSERT INTO face_meta (uID, EMBEDDINGS, IMG, FULL_IMG, FIRST_SEEN, LAST_SEEN, NAME, TIME_RANGES, TOTAL_TIME) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (uid, emb_blob, face_img_rel, full_img_rel, now_str, now_str, None, __import__('json').dumps(tr), 0.0)
    )
    conn.commit()
    conn.close()
    return uid, None, 0.0
