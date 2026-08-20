import os
import sys
import unittest
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from config import KEYFRAME_ROOTS, METADATA_DB_PATH
from src.db.metadata_db import MetadataDB
from src.retrieval.service import RetrievalService
from src.api.main import _resolve_image_path

def test_keyframe_flow():
    print("=" * 60)
    print(">>> TESTING KEYFRAME IMAGE PREVIEW & PATH RESOLUTION <<<")
    print("=" * 60)

    db = MetadataDB(METADATA_DB_PATH)
    retrieval_service = RetrievalService.get_instance()

    # 1. Test L21 Path Resolution
    print("\n[TEST 1] Testing L21 Keyframe Resolution...")
    l21_sample = db.get_by_global_id(1)
    assert l21_sample is not None
    l21_path = l21_sample["image_path"]
    resolved_l21 = _resolve_image_path(l21_path)
    print(f"L21 raw: {l21_path} -> resolved: {resolved_l21}")
    assert resolved_l21 is not None and resolved_l21.is_file(), f"Failed to resolve L21 path: {l21_path}"
    print("-> L21 Image Resolution: PASSED")

    # 2. Test L22 Path Resolution
    print("\n[TEST 2] Testing L22 Keyframe Resolution...")
    # Find first L22 global_id
    with db.get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT global_id, image_path FROM metadata WHERE video_id LIKE 'L22%' LIMIT 1")
        l22_row = c.fetchone()
    assert l22_row is not None
    l22_path = l22_row["image_path"]
    resolved_l22 = _resolve_image_path(l22_path)
    print(f"L22 raw: {l22_path} -> resolved: {resolved_l22}")
    assert resolved_l22 is not None and resolved_l22.is_file(), f"Failed to resolve L22 path: {l22_path}"
    print("-> L22 Image Resolution: PASSED")

    # 3. Test L23 Path Resolution
    print("\n[TEST 3] Testing L23 Keyframe Resolution...")
    # Find first L23 global_id
    with db.get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT global_id, image_path FROM metadata WHERE video_id LIKE 'L23%' LIMIT 1")
        l23_row = c.fetchone()
    assert l23_row is not None
    l23_path = l23_row["image_path"]
    resolved_l23 = _resolve_image_path(l23_path)
    print(f"L23 raw: {l23_path} -> resolved: {resolved_l23}")
    assert resolved_l23 is not None and resolved_l23.is_file(), f"Failed to resolve L23 path: {l23_path}"
    print("-> L23 Image Resolution: PASSED")

    # 4. Test Search Result image_url
    print("\n[TEST 4] Testing Search Results image_url attachment...")
    search_res = retrieval_service.search("car on the street", top_k=10)
    assert len(search_res["results"]) > 0
    for r in search_res["results"]:
        assert "image_url" in r, f"Missing image_url in search result item: {r}"
        assert r["image_url"].startswith("/media/"), f"Invalid image_url format: {r['image_url']}"
    print(f"-> Search result sample image_url: {search_res['results'][0]['image_url']}")
    print("-> Search Results image_url: PASSED")

    # 5. Test Neighbor Frames image_url
    print("\n[TEST 5] Testing Neighbor Frames image_url attachment...")
    neighbors = db.get_neighbors(1, window=5)
    assert len(neighbors) == 6 # frame_idx 0 to 5
    for n in neighbors:
        assert "image_url" in n, f"Missing image_url in neighbor item: {n}"
        assert n["image_url"].startswith("/media/"), f"Invalid image_url format: {n['image_url']}"
    print(f"-> Neighbor sample image_url: {neighbors[0]['image_url']}")
    print("-> Neighbor Frames image_url: PASSED")

    # 6. Test Detailed Video Summaries with Thumbnails
    print("\n[TEST 6] Testing Video Summaries & Thumbnails...")
    summaries = db.get_video_summaries()
    assert len(summaries) >= 85, f"Expected >= 85 videos, got {len(summaries)}"
    l21_summaries = [s for s in summaries if s["dataset"] == "L21"]
    l22_summaries = [s for s in summaries if s["dataset"] == "L22"]
    l23_summaries = [s for s in summaries if s["dataset"] == "L23"]
    assert len(l21_summaries) == 29, f"L21 video count mismatch: {len(l21_summaries)}"
    assert len(l22_summaries) == 31, f"L22 video count mismatch: {len(l22_summaries)}"
    assert len(l23_summaries) == 25, f"L23 video count mismatch: {len(l23_summaries)}"
    
    # Check that each video has a valid thumbnail
    for s in summaries:
        assert s["image_url"].startswith("/media/"), f"Invalid video thumbnail url: {s['image_url']}"
        assert len(s["sample_frames"]) > 0, f"No sample frames for video: {s['video_id']}"
    print(f"-> Verified 85 videos (L21: {len(l21_summaries)}, L22: {len(l22_summaries)}, L23: {len(l23_summaries)}) with valid thumbnails.")
    print("-> Video Summaries & Thumbnails: PASSED")

    # 7. Test Missing Image Fallback
    print("\n[TEST 7] Testing Missing Image Path...")
    missing_resolved = _resolve_image_path("L21_V999/non_existent_frame.jpg")
    assert missing_resolved is None, "Missing path should resolve to None"
    print("-> Missing image correctly resolved to None (triggers SVG fallback)")
    print("-> Missing Image Fallback: PASSED")

    # 8. Test Security Path Traversal Protection
    print("\n[TEST 8] Testing Path Traversal Protection...")
    traversal_paths = [
        "../../config.py",
        "../metadata.db",
        "L21/../../../etc/passwd",
        "/absolute/path/test.jpg",
        "L21_V001/../../config.py"
    ]
    for tp in traversal_paths:
        res = _resolve_image_path(tp)
        assert res is None, f"Security vulnerability: path traversal not blocked for {tp}"
    print("-> All path traversal attempts successfully rejected.")
    print("-> Security Protection: PASSED")

    print("\n" + "=" * 60)
    print(">>> ALL KEYFRAME PREVIEW TESTS PASSED WITH 100% SUCCESS! <<<")
    print("=" * 60)

if __name__ == "__main__":
    test_keyframe_flow()
