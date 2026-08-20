import sqlite3
import os
from typing import List, Dict, Any, Optional, Tuple

class MetadataDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_tables()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Check if keyframes table exists (standard notebook table with 420k+ frames)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='keyframes'")
            if cursor.fetchone() is not None:
                self.table_name = "keyframes"
                # Ensure fast lookup indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_keyframes_video_frame ON keyframes(video_id, frame_idx);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_keyframes_keyframe_id ON keyframes(keyframe_id);")
            else:
                self.table_name = "metadata"
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
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
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_frame ON metadata(video_id, frame_idx);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_id ON metadata(video_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_frame_id ON metadata(frame_id);")
            conn.commit()

    def get_by_global_id(self, global_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.table_name == "keyframes":
                cursor.execute("""
                    SELECT rowid AS global_id,
                           keyframe_id AS frame_id,
                           video_id,
                           shot_id,
                           ordinal,
                           frame_idx,
                           timestamp,
                           image_path,
                           '' AS objects_tag,
                           '' AS object_detail,
                           '' AS caption
                    FROM keyframes
                    WHERE rowid = ?
                """, (global_id,))
            else:
                cursor.execute("SELECT * FROM metadata WHERE global_id = ?", (global_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_by_global_ids(self, global_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        if not global_ids:
            return {}
        placeholders = ",".join("?" for _ in global_ids)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.table_name == "keyframes":
                query = f"""
                    SELECT rowid AS global_id,
                           keyframe_id AS frame_id,
                           video_id,
                           shot_id,
                           ordinal,
                           frame_idx,
                           timestamp,
                           image_path,
                           '' AS objects_tag,
                           '' AS object_detail,
                           '' AS caption
                    FROM keyframes
                    WHERE rowid IN ({placeholders})
                """
            else:
                query = f"SELECT * FROM metadata WHERE global_id IN ({placeholders})"
            cursor.execute(query, global_ids)
            rows = cursor.fetchall()
            return {row["global_id"]: dict(row) for row in rows}

    def get_neighbors(self, global_id: int, window: int = 5) -> List[Dict[str, Any]]:
        target = self.get_by_global_id(global_id)
        if not target:
            return []
        
        video_id = target["video_id"]
        frame_idx = target["frame_idx"]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.table_name == "keyframes":
                cursor.execute("""
                    SELECT rowid AS global_id, keyframe_id AS frame_id, video_id, shot_id, ordinal, frame_idx, timestamp, image_path, '' AS objects_tag, '' AS object_detail, '' AS caption
                    FROM keyframes
                    WHERE video_id = ? AND frame_idx < ?
                    ORDER BY frame_idx DESC LIMIT ?
                """, (video_id, frame_idx, window))
                before = list(reversed(cursor.fetchall()))

                cursor.execute("""
                    SELECT rowid AS global_id, keyframe_id AS frame_id, video_id, shot_id, ordinal, frame_idx, timestamp, image_path, '' AS objects_tag, '' AS object_detail, '' AS caption
                    FROM keyframes
                    WHERE rowid = ?
                """, (global_id,))
                center = cursor.fetchall()

                cursor.execute("""
                    SELECT rowid AS global_id, keyframe_id AS frame_id, video_id, shot_id, ordinal, frame_idx, timestamp, image_path, '' AS objects_tag, '' AS object_detail, '' AS caption
                    FROM keyframes
                    WHERE video_id = ? AND frame_idx > ?
                    ORDER BY frame_idx ASC LIMIT ?
                """, (video_id, frame_idx, window))
                after = cursor.fetchall()
            else:
                cursor.execute("""
                    SELECT * FROM metadata
                    WHERE video_id = ? AND frame_idx < ?
                    ORDER BY frame_idx DESC LIMIT ?
                """, (video_id, frame_idx, window))
                before = list(reversed(cursor.fetchall()))

                cursor.execute("SELECT * FROM metadata WHERE global_id = ?", (global_id,))
                center = cursor.fetchall()

                cursor.execute("""
                    SELECT * FROM metadata
                    WHERE video_id = ? AND frame_idx > ?
                    ORDER BY frame_idx ASC LIMIT ?
                """, (video_id, frame_idx, window))
                after = cursor.fetchall()

            all_rows = before + center + after
            results = []
            for row in all_rows:
                item = dict(row)
                item["is_center"] = (item["global_id"] == global_id)
                item["image_url"] = f"/media/{item['image_path']}"
                results.append(item)
            return results

    def get_video_summaries(self, dataset: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            tbl = self.table_name
            if dataset:
                cursor.execute(
                    f"SELECT video_id, COUNT(*) as total_frames "
                    f"FROM {tbl} WHERE video_id LIKE ? GROUP BY video_id ORDER BY video_id ASC",
                    (f"{dataset}%",)
                )
            else:
                cursor.execute(
                    f"SELECT video_id, COUNT(*) as total_frames "
                    f"FROM {tbl} WHERE video_id IS NOT NULL GROUP BY video_id ORDER BY video_id ASC"
                )
            summary_rows = cursor.fetchall()

            results = []
            for row in summary_rows:
                vid = row["video_id"]
                total = row["total_frames"]

                if tbl == "keyframes":
                    cursor.execute(
                        "SELECT rowid AS global_id, keyframe_id AS frame_id, frame_idx, timestamp, image_path FROM keyframes "
                        "WHERE video_id = ? ORDER BY frame_idx ASC LIMIT 4",
                        (vid,)
                    )
                else:
                    cursor.execute(
                        "SELECT global_id, frame_id, frame_idx, timestamp, image_path FROM metadata "
                        "WHERE video_id = ? ORDER BY frame_idx ASC LIMIT 4",
                        (vid,)
                    )
                sample_rows = cursor.fetchall()
                samples = []
                for s in sample_rows:
                    samples.append({
                        "global_id": s["global_id"],
                        "frame_id": s["frame_id"],
                        "frame_idx": s["frame_idx"],
                        "timestamp": s["timestamp"],
                        "image_path": s["image_path"],
                        "image_url": f"/media/{s['image_path']}",
                    })

                first_img = samples[0]["image_path"] if samples else ""
                first_url = samples[0]["image_url"] if samples else ""
                ds_prefix = vid.split("_")[0] if "_" in vid else vid

                results.append({
                    "video_id": vid,
                    "dataset": ds_prefix,
                    "total_frames": total,
                    "thumbnail_path": first_img,
                    "image_url": first_url,
                    "sample_frames": samples
                })
            return results

    def get_all_videos(self) -> List[str]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            tbl = self.table_name
            cursor.execute(f"SELECT DISTINCT video_id FROM {tbl} WHERE video_id IS NOT NULL ORDER BY video_id ASC")
            return [row["video_id"] for row in cursor.fetchall()]

    def get_all_objects(self) -> List[str]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            tbl = self.table_name
            if tbl == "metadata":
                cursor.execute("SELECT DISTINCT objects_tag FROM metadata WHERE objects_tag IS NOT NULL AND objects_tag != ''")
                tags_set = set()
                for row in cursor.fetchall():
                    raw_tags = row["objects_tag"]
                    if raw_tags:
                        for t in raw_tags.replace(";", ",").split(","):
                            cleaned = t.strip().lower()
                            if cleaned:
                                tags_set.add(cleaned)
                return sorted(list(tags_set))
            return []

    def count(self) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            tbl = self.table_name
            cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
            return cursor.fetchone()[0]
