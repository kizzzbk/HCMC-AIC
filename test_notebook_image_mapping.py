"""
Test Notebook Image Mapping Suite
Validates that keyframe image resolution adheres 100% to Download_Keyframe.ipynb.
"""
import sys
import sqlite3
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.retrieval.keyframe_resolver import get_folder_name, resolve_keyframe_image_path
from src.retrieval.service import RetrievalService
from src.db.metadata_db import MetadataDB
from config import METADATA_DB_PATH

def run_tests():
    print("=" * 60)
    print(">>> TESTING NOTEBOOK IMAGE MAPPING & KEYFRAME RESOLUTION <<<")
    print("=" * 60)

    # 1. Test get_folder_name derived from notebook Cell 4
    test_cases = [
        ('L21_v030_transnetv2_0093_06', 'L21_V030'),
        ('L22_v001_transnetv2_0063_01', 'L22_V001'),
        ('L26_v118_transnetv2_0014_07', 'L26_V118'),
        ('L23_V001_transnetv2_0001_00', 'L23_V001'),
        ('L21_V001_transnetv2_0001_00.jpg', 'L21_V001'),
    ]

    print("\n[TEST 1] Testing get_folder_name (from notebook Cell 4)...")
    for img_id, expected_folder in test_cases:
        derived = get_folder_name(img_id)
        assert derived == expected_folder, f"Expected {expected_folder}, got {derived}"
        print(f"  '{img_id}' -> '{derived}' (MATCH)")
    print("-> get_folder_name: PASSED")

    # 2. Test notebook sample images exist on local disk
    print("\n[TEST 2] Testing notebook sample keyframes on local disk (L21, L22, L23)...")
    notebook_samples = [
        'L21_v030_transnetv2_0093_06',
        'L22_v001_transnetv2_0063_01',
        'L21_V001_transnetv2_0001_00',
        'L23_V001_transnetv2_0001_00'
    ]
    for s in notebook_samples:
        resolved = resolve_keyframe_image_path(s)
        assert resolved is not None and resolved.is_file(), f"Sample {s} failed to resolve!"
        print(f"  {s} -> {resolved} (EXISTS: {resolved.exists()})")
    print("-> Notebook sample keyframes: PASSED")

    # 3. Test frames for all available datasets (L21, L22, L23, L24, L25, L27) from database
    print("\n[TEST 3] Testing random frames per dataset (L21, L22, L23, L24, L25, L27) from database...")
    db = MetadataDB(METADATA_DB_PATH)

    for ds in ['L21', 'L22', 'L23', 'L24', 'L25', 'L27']:
        summaries = db.get_video_summaries(dataset=ds)
        if not summaries:
            print(f"  [SKIPPED] No video summaries for {ds}")
            continue
        print(f"  Testing dataset {ds} ({len(summaries)} videos found):")
        tested = 0
        for vid_summary in summaries[:3]:
            for frame in vid_summary['sample_frames']:
                res_by_path = resolve_keyframe_image_path(frame['image_path'])
                res_by_fid = resolve_keyframe_image_path(frame['frame_id'])
                assert res_by_path is not None and res_by_path.is_file(), f"Failed for {frame['image_path']}"
                assert res_by_fid is not None and res_by_fid.is_file(), f"Failed for {frame['frame_id']}"
                tested += 1
        print(f"    Verified {tested} frames on disk -> OK")
    print("-> Dataset frames check (L21, L22, L23, L24, L25, L27): PASSED")

    # 4. Test Search Result image_url
    print("\n[TEST 4] Testing RetrievalService search result image_url...")
    srv = RetrievalService.get_instance()
    res = srv.search(query='car', top_k=5)
    assert len(res['results']) > 0, "No search results returned!"
    for item in res['results']:
        assert 'image_url' in item, "Missing image_url in search result!"
        assert 'global_id' in item, "Missing global_id in search result!"
        assert item['video_id'] != 'unknown', "video_id is unknown!"
        print(f"  Result: Global ID #{item['global_id']} -> Video: {item['video_id']} ({item['frame_id']}) -> URL: {item['image_url']}")
    print("-> Search Results image_url: PASSED")

    # 5. Test Neighbor Frames image_url
    print("\n[TEST 5] Testing MetadataDB get_neighbors image_url...")
    neighbors = db.get_neighbors(global_id=1, window=5)
    assert len(neighbors) > 0, "No neighbors returned!"
    for n in neighbors:
        assert 'image_url' in n, "Missing image_url in neighbor!"
        print(f"  Neighbor: Global ID #{n['global_id']} (frame_idx:{n['frame_idx']}) -> URL: {n['image_url']}")
    print("-> Neighbor Frames image_url: PASSED")

    # 6. Test Security (Path Traversal)
    print("\n[TEST 6] Testing Security Path Traversal...")
    assert resolve_keyframe_image_path("../../../etc/passwd") is None
    assert resolve_keyframe_image_path("..\\..\\Windows\\System32") is None
    assert resolve_keyframe_image_path("L21/../../secret.txt") is None
    print("-> Path Traversal Protection: PASSED")

    print("\n" + "=" * 60)
    print(">>> ALL NOTEBOOK MAPPING TESTS PASSED WITH 100% SUCCESS! <<<")
    print("=" * 60)

if __name__ == '__main__':
    run_tests()
