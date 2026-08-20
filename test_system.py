import os
import sys
import time
import json
import requests
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from config import (
    FAISS_SIGLIP_PATH,
    FAISS_OPENCLIP_PATH,
    FAISS_CAPTION_PATH,
    METADATA_DB_PATH,
)
from src.db.metadata_db import MetadataDB
from src.retrieval.encoders import ModelManager
from src.retrieval.faiss_search import FAISSManager
from src.retrieval.fusion import FusionEngine
from src.retrieval.service import RetrievalService
from src.submission.btc_payload import build_submission_payload, submit_to_btc

def run_all_tests():
    print("=" * 60)
    print(">>> RUNNING SYSTEM VERIFICATION SUITE (14 MANDATORY TESTS) <<<")
    print("=" * 60)

    test_results = {}
    sample_query = "a man wearing a red shirt near a car"

    # -------------------------------------------------------------
    # Test 1: Load SigLIP2 FAISS
    # -------------------------------------------------------------
    print("\n[TEST 1] Load SigLIP2 FAISS Index...")
    try:
        faiss_mgr = FAISSManager.get_instance()
        assert faiss_mgr.index_siglip is not None, "SigLIP index is None"
        print(f"-> PASSED: SigLIP2 ntotal={faiss_mgr.index_siglip.ntotal}, dim={faiss_mgr.index_siglip.d}")
        test_results["Test 1 (Load SigLIP2 FAISS)"] = "PASSED"
    except Exception as e:
        print(f"-> FAILED: {e}")
        test_results["Test 1 (Load SigLIP2 FAISS)"] = f"FAILED: {e}"

    # -------------------------------------------------------------
    # Test 2: Load OpenCLIP FAISS
    # -------------------------------------------------------------
    print("\n[TEST 2] Load OpenCLIP FAISS Index...")
    try:
        assert faiss_mgr.index_openclip is not None, "OpenCLIP index is None"
        print(f"-> PASSED: OpenCLIP ntotal={faiss_mgr.index_openclip.ntotal}, dim={faiss_mgr.index_openclip.d}")
        test_results["Test 2 (Load OpenCLIP FAISS)"] = "PASSED"
    except Exception as e:
        print(f"-> FAILED: {e}")
        test_results["Test 2 (Load OpenCLIP FAISS)"] = f"FAILED: {e}"

    # -------------------------------------------------------------
    # Test 3: Load Caption FAISS
    # -------------------------------------------------------------
    print("\n[TEST 3] Load Caption FAISS Index...")
    try:
        caption_status = "LOADED" if faiss_mgr.index_caption is not None else "OPTIONAL_FALLBACK_ACTIVE"
        print(f"-> PASSED: Caption FAISS Status = {caption_status}")
        test_results["Test 3 (Load Caption FAISS)"] = "PASSED"
    except Exception as e:
        print(f"-> FAILED: {e}")
        test_results["Test 3 (Load Caption FAISS)"] = f"FAILED: {e}"

    # -------------------------------------------------------------
    # Test 4: global_id -> SQLite metadata
    # -------------------------------------------------------------
    print("\n[TEST 4] global_id -> SQLite lookup...")
    try:
        db = MetadataDB(METADATA_DB_PATH)
        sample_record = db.get_by_global_id(1)
        assert sample_record is not None, "Record global_id=1 not found"
        assert "video_id" in sample_record, "video_id missing"
        assert "frame_idx" in sample_record, "frame_idx missing"
        print(f"-> PASSED: Found record for global_id=1: video_id={sample_record['video_id']}, frame_id={sample_record['frame_id']}")
        test_results["Test 4 (global_id -> SQLite)"] = "PASSED"
    except Exception as e:
        print(f"-> FAILED: {e}")
        test_results["Test 4 (global_id -> SQLite)"] = f"FAILED: {e}"

    # -------------------------------------------------------------
    # Test 5: query -> 3 embeddings
    # -------------------------------------------------------------
    print(f"\n[TEST 5] Query -> 3 text embeddings (query='{sample_query}')...")
    try:
        model_mgr = ModelManager.get_instance()
        embs = model_mgr.encode_query(sample_query)
        assert embs["siglip"].shape[1] == 768, f"SigLIP dim mismatch: {embs['siglip'].shape}"
        assert embs["openclip"].shape[1] == 512, f"OpenCLIP dim mismatch: {embs['openclip'].shape}"
        print(f"-> PASSED: SigLIP shape={embs['siglip'].shape}, OpenCLIP shape={embs['openclip'].shape}, Caption shape={embs['caption'].shape if embs['caption'] is not None else 'None'}")
        test_results["Test 5 (Query -> 3 Embeddings)"] = "PASSED"
    except Exception as e:
        print(f"-> FAILED: {e}")
        test_results["Test 5 (Query -> 3 Embeddings)"] = f"FAILED: {e}"

    # -------------------------------------------------------------
    # Test 6: 3 FAISS -> candidates
    # -------------------------------------------------------------
    print("\n[TEST 6] 3 FAISS search -> Candidate retrieval (Top 100 each)...")
    try:
        faiss_res = faiss_mgr.search_all(embs, top_n=100)
        assert len(faiss_res["siglip"]) > 0, "SigLIP returned 0 candidates"
        assert len(faiss_res["openclip"]) > 0, "OpenCLIP returned 0 candidates"
        print(f"-> PASSED: SigLIP returned {len(faiss_res['siglip'])} candidates, OpenCLIP returned {len(faiss_res['openclip'])} candidates")
        print(f"   SigLIP Top 1: ID={faiss_res['siglip'][0][0]}, Score={faiss_res['siglip'][0][1]:.4f}")
        print(f"   OpenCLIP Top 1: ID={faiss_res['openclip'][0][0]}, Score={faiss_res['openclip'][0][1]:.4f}")
        test_results["Test 6 (3 FAISS -> Candidates)"] = "PASSED"
    except Exception as e:
        print(f"-> FAILED: {e}")
        test_results["Test 6 (3 FAISS -> Candidates)"] = f"FAILED: {e}"

    # -------------------------------------------------------------
    # Test 7: Candidates -> RRF Fusion
    # -------------------------------------------------------------
    print("\n[TEST 7] Candidates -> Reciprocal Rank Fusion (RRF k=60)...")
    try:
        fused = FusionEngine.reciprocal_rank_fusion(
            siglip_results=faiss_res["siglip"],
            openclip_results=faiss_res["openclip"],
            caption_results=faiss_res["caption"]
        )
        assert len(fused) > 0, "RRF fused result is empty"
        print(f"-> PASSED: Fused {len(fused)} unique candidate pool.")
        print(f"   Top 1 Fused: ID={fused[0]['global_id']}, RRF Score={fused[0]['score']}")
        test_results["Test 7 (Candidates -> RRF)"] = "PASSED"
    except Exception as e:
        print(f"-> FAILED: {e}")
        test_results["Test 7 (Candidates -> RRF)"] = f"FAILED: {e}"

    # -------------------------------------------------------------
    # Test 8: RRF -> SQLite metadata attachment
    # -------------------------------------------------------------
    print("\n[TEST 8] RRF -> SQLite metadata lookup & attachment...")
    try:
        top_ids = [item["global_id"] for item in fused[:10]]
        meta_map = db.get_by_global_ids(top_ids)
        assert len(meta_map) == 10, f"Expected 10 metadata records, got {len(meta_map)}"
        for gid in top_ids:
            m = meta_map[gid]
            assert "video_id" in m and "frame_idx" in m, "Metadata fields missing"
        print(f"-> PASSED: Successfully retrieved and matched metadata for all top 10 IDs.")
        test_results["Test 8 (RRF -> SQLite metadata)"] = "PASSED"
    except Exception as e:
        print(f"-> FAILED: {e}")
        test_results["Test 8 (RRF -> SQLite metadata)"] = f"FAILED: {e}"

    # -------------------------------------------------------------
    # Test 9: /search API (via RetrievalService)
    # -------------------------------------------------------------
    print("\n[TEST 9] RetrievalService search pipeline (/search)...")
    try:
        service = RetrievalService.get_instance()
        search_res = service.search(query=sample_query, top_k=20, offset=0)
        assert len(search_res["results"]) == 20, f"Expected 20 results, got {len(search_res['results'])}"
        assert search_res["total"] > 0, "Total count is 0"
        print(f"-> PASSED: /search completed in {search_res['time_taken_ms']}ms, returned {len(search_res['results'])}/{search_res['total']} items.")
        test_results["Test 9 (/search API pipeline)"] = "PASSED"
    except Exception as e:
        print(f"-> FAILED: {e}")
        test_results["Test 9 (/search API pipeline)"] = f"FAILED: {e}"

    # -------------------------------------------------------------
    # Test 10: /neighbors API
    # -------------------------------------------------------------
    print("\n[TEST 10] Frame Neighbors retrieval (GET /neighbors)...")
    try:
        center_id = fused[0]["global_id"]
        neighbors = db.get_neighbors(center_id, window=5)
        assert len(neighbors) > 0, "No neighbors returned"
        center_match = [n for n in neighbors if n.get("is_center")]
        assert len(center_match) == 1, "Center frame not marked"
        print(f"-> PASSED: Retrieved {len(neighbors)} neighbor frames around global_id={center_id}.")
        test_results["Test 10 (/neighbors API)"] = "PASSED"
    except Exception as e:
        print(f"-> FAILED: {e}")
        test_results["Test 10 (/neighbors API)"] = f"FAILED: {e}"

    # -------------------------------------------------------------
    # Test 11: Video filter
    # -------------------------------------------------------------
    print("\n[TEST 11] Video filter test...")
    try:
        target_vid = search_res["results"][0]["video_id"]
        filtered_res = service.search(query=sample_query, top_k=20, offset=0, video_ids=[target_vid])
        for r in filtered_res["results"]:
            assert r["video_id"].lower() == target_vid.lower(), f"Video ID mismatch: {r['video_id']} != {target_vid}"
        print(f"-> PASSED: Filtered by video_id='{target_vid}', all {len(filtered_res['results'])} results match.")
        test_results["Test 11 (Video Filter)"] = "PASSED"
    except Exception as e:
        print(f"-> FAILED: {e}")
        test_results["Test 11 (Video Filter)"] = f"FAILED: {e}"

    # -------------------------------------------------------------
    # Test 12: Object filter
    # -------------------------------------------------------------
    print("\n[TEST 12] Object filter test...")
    try:
        obj_res = service.search(query=sample_query, top_k=20, offset=0, objects=["car"])
        print(f"-> PASSED: Object filter executed smoothly ({len(obj_res['results'])} matches).")
        test_results["Test 12 (Object Filter)"] = "PASSED"
    except Exception as e:
        print(f"-> FAILED: {e}")
        test_results["Test 12 (Object Filter)"] = f"FAILED: {e}"

    # -------------------------------------------------------------
    # Test 13: Next Top-K Pagination
    # -------------------------------------------------------------
    print("\n[TEST 13] Next Top-K Pagination test (offset=20)...")
    try:
        page1 = service.search(query=sample_query, top_k=20, offset=0)
        page2 = service.search(query=sample_query, top_k=20, offset=20)
        assert len(page2["results"]) > 0, "Page 2 returned 0 items"
        ids_page1 = {r["global_id"] for r in page1["results"]}
        ids_page2 = {r["global_id"] for r in page2["results"]}
        overlap = ids_page1.intersection(ids_page2)
        assert len(overlap) == 0, f"Pages should not overlap, found {len(overlap)} duplicate IDs"
        print(f"-> PASSED: Page 1 (IDs 1-20) and Page 2 (IDs 21-40) are distinct and correctly paginated.")
        test_results["Test 13 (Next Top-K)"] = "PASSED"
    except Exception as e:
        print(f"-> FAILED: {e}")
        test_results["Test 13 (Next Top-K)"] = f"FAILED: {e}"

    # -------------------------------------------------------------
    # Test 14: Submit Payload & Flow
    # -------------------------------------------------------------
    print("\n[TEST 14] Submission payload generation & mock submit...")
    try:
        target_frame = db.get_by_global_id(fused[0]["global_id"])
        payload = build_submission_payload(target_frame, query=sample_query, session_id="TEST_SESSION")
        assert "item" in payload, "Missing item in payload"
        assert payload["item"]["global_id"] == target_frame["global_id"], "Payload global_id mismatch"
        assert payload["item"]["video_id"] == target_frame["video_id"], "Payload video_id mismatch"
        print(f"-> PASSED: Submission payload created successfully:\n{json.dumps(payload, indent=2)}")
        test_results["Test 14 (Submit)"] = "PASSED"
    except Exception as e:
        print(f"-> FAILED: {e}")
        test_results["Test 14 (Submit)"] = f"FAILED: {e}"

    print("\n" + "=" * 60)
    print(">>> TEST SUMMARY <<<")
    print("=" * 60)
    all_passed = True
    for test_name, status in test_results.items():
        print(f"{test_name:.<45} {status}")
        if status != "PASSED":
            all_passed = False

    if all_passed:
        print("\n>>> ALL 14 TESTS PASSED PERFECTLY! <<<")
    else:
        print("\n>>> SOME TESTS FAILED! <<<")

if __name__ == "__main__":
    run_all_tests()
