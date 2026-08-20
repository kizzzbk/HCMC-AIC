import os
import sys
import json
import sqlite3
import re
from pathlib import Path
from tqdm import tqdm

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from config import METADATA_DB_PATH, METADATA_JSON_PATH

def init_database(force_rebuild: bool = False):
    db_file = Path(METADATA_DB_PATH)
    if db_file.exists() and not force_rebuild:
        # Check if already populated
        conn = sqlite3.connect(METADATA_DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM metadata")
            cnt = cursor.fetchone()[0]
            if cnt > 0:
                print(f"[DB] Database already initialized with {cnt} records at {METADATA_DB_PATH}.")
                conn.close()
                return
        except sqlite3.OperationalError:
            pass
        conn.close()

    print(f"[DB] Building metadata database from {METADATA_JSON_PATH}...")
    if not os.path.exists(METADATA_JSON_PATH):
        raise FileNotFoundError(f"Cannot find metadata JSON file at {METADATA_JSON_PATH}")

    with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    print(f"[DB] Loaded {len(records)} raw records from JSON.")

    if db_file.exists():
        db_file.unlink()

    conn = sqlite3.connect(METADATA_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE metadata (
            global_id INTEGER PRIMARY KEY,
            frame_id TEXT,
            video_id TEXT,
            frame_idx INTEGER,
            timestamp REAL,
            image_path TEXT,
            objects_tag TEXT,
            object_detail TEXT,
            caption TEXT
        );
    """)

    # Track frame_idx per video_id
    video_frame_counters = {}
    rows_to_insert = []
    batch_size = 50000

    for r in tqdm(records, desc="Processing metadata records"):
        # global_id is 1-indexed in FAISS
        raw_id = r.get("id", 0)
        global_id = raw_id + 1
        video_id = r.get("video_id", "")
        img_name = r.get("image_name", "")
        frame_id = img_name.replace(".jpg", "").replace(".png", "")
        rel_path = r.get("relative_path", "")

        # Compute sequential frame index per video
        if video_id not in video_frame_counters:
            video_frame_counters[video_id] = 0
        frame_idx = video_frame_counters[video_id]
        video_frame_counters[video_id] += 1

        # Approximate timestamp
        timestamp = round(frame_idx * 1.5, 2)
        
        # Tags / object extraction placeholder or derived from video name
        objects_tag = r.get("objects_tag", "")
        object_detail = r.get("object_detail", "")
        caption = r.get("caption", "")

        rows_to_insert.append((
            global_id,
            frame_id,
            video_id,
            frame_idx,
            timestamp,
            rel_path,
            objects_tag,
            object_detail,
            caption
        ))

        if len(rows_to_insert) >= batch_size:
            cursor.executemany(
                "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows_to_insert
            )
            conn.commit()
            rows_to_insert = []

    if rows_to_insert:
        cursor.executemany(
            "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows_to_insert
        )
        conn.commit()

    print("[DB] Creating indices on (video_id, frame_idx), (video_id), (frame_id)...")
    cursor.execute("CREATE INDEX idx_video_frame ON metadata(video_id, frame_idx);")
    cursor.execute("CREATE INDEX idx_video_id ON metadata(video_id);")
    cursor.execute("CREATE INDEX idx_frame_id ON metadata(frame_id);")
    cursor.execute("CREATE INDEX idx_objects_tag ON metadata(objects_tag);")
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM metadata")
    total = cursor.fetchone()[0]
    conn.close()
    print(f"[DB] Successfully created metadata.db with {total} records at {METADATA_DB_PATH}!")

if __name__ == "__main__":
    init_database(force_rebuild=True)
