import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import (
    KEYFRAMES_DIR,
    KEYFRAME_ROOTS,
    STATIC_DIR,
    METADATA_DB_PATH,
    DEFAULT_TOP_K,
    DEFAULT_TOP_N,
)
from src.retrieval.service import RetrievalService
from src.db.metadata_db import MetadataDB
from src.submission.btc_payload import build_submission_payload, submit_to_btc

app = FastAPI(
    title="HCMC AI Video Keyframe Retrieval API",
    description="Multi-modal Hybrid Video Keyframe Retrieval System powered by SigLIP2, OpenCLIP, and FAISS",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class SearchRequest(BaseModel):
    query: str = Field(..., description="Text search query in Vietnamese or English")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=200, description="Number of results per page")
    offset: int = Field(default=0, ge=0, description="Pagination offset for Next Top-K")
    video_ids: Optional[List[str]] = Field(default=None, description="Optional list of video_ids to filter")
    objects: Optional[List[str]] = Field(default=None, description="Optional list of object tags to filter")
    top_n_candidates: Optional[int] = Field(default=DEFAULT_TOP_N, description="FAISS candidate pool size per index")

class SubmitRequest(BaseModel):
    global_id: int = Field(..., description="Canonical global_id of selected keyframe")
    query: Optional[str] = Field(default=None, description="Original query associated with submission")
    session_id: Optional[str] = Field(default=None, description="Optional session ID")

# Singletons (lazy/startup init)
retrieval_service: Optional[RetrievalService] = None
db: Optional[MetadataDB] = None

@app.on_event("startup")
async def startup_event():
    global retrieval_service, db
    print("[API] Initializing Database connection...")
    db = MetadataDB(METADATA_DB_PATH)
    print(f"[API] Connected to SQLite DB ({db.count()} frames).")
    print("[API] Initializing RetrievalService and Models...")
    retrieval_service = RetrievalService.get_instance()
    print("[API] System ready to serve requests!")


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "total_frames": db.count() if db else 0,
        "models": {
            "siglip": retrieval_service.faiss_manager.index_siglip is not None if retrieval_service else False,
            "openclip": retrieval_service.faiss_manager.index_openclip is not None if retrieval_service else False,
            "caption": retrieval_service.faiss_manager.index_caption is not None if retrieval_service else False,
        }
    }


@app.post("/api/search")
def search_keyframes(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")
    
    try:
        results = retrieval_service.search(
            query=req.query,
            top_k=req.top_k,
            offset=req.offset,
            video_ids=req.video_ids,
            objects=req.objects,
            top_n_candidates=req.top_n_candidates or DEFAULT_TOP_N
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/api/frames/{global_id}/neighbors")
def get_frame_neighbors(global_id: int, window: int = Query(default=5, ge=1, le=20)):
    target = db.get_by_global_id(global_id)
    if not target:
        raise HTTPException(status_code=404, detail=f"Frame with global_id {global_id} not found")
    
    neighbors = db.get_neighbors(global_id, window=window)
    return {
        "center_id": global_id,
        "video_id": target["video_id"],
        "center_frame_idx": target["frame_idx"],
        "window": window,
        "total_neighbors": len(neighbors),
        "neighbors": neighbors
    }


@app.get("/api/videos")
def list_videos(detailed: bool = Query(default=False), dataset: Optional[str] = None):
    try:
        if detailed:
            videos = db.get_video_summaries(dataset=dataset)
            return {"videos": videos, "total": len(videos)}
        videos = db.get_all_videos()
        return {"videos": videos, "total": len(videos)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/objects")
def list_objects():
    try:
        objs = db.get_all_objects()
        return {"objects": objs, "total": len(objs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/submit")
def submit_frame(req: SubmitRequest):
    target = db.get_by_global_id(req.global_id)
    if not target:
        raise HTTPException(status_code=404, detail=f"Cannot submit: global_id {req.global_id} not found in database")

    payload = build_submission_payload(
        metadata=target,
        query=req.query,
        session_id=req.session_id
    )

    # Attempt submission
    sub_res = submit_to_btc(payload)
    return {
        "success": True,
        "global_id": req.global_id,
        "payload": payload,
        "btc_submission_result": sub_res
    }


@app.post("/api/mock_btc_submit")
def mock_btc_submit(payload: Dict[str, Any]):
    """
    Mock BTC evaluation server endpoint for testing submission flow offline.
    """
    return {
        "status": "ACCEPTED",
        "message": "Submission successfully received and verified.",
        "received_payload": payload
    }


from src.retrieval.keyframe_resolver import resolve_keyframe_image_path


def _resolve_image_path(raw_path: str) -> Optional[Path]:
    """
    Safely resolves a relative image path or image_id to an existing filesystem location.
    Uses exact logic derived from Download_Keyframe.ipynb.
    """
    # 1. Use the notebook-derived resolver
    resolved = resolve_keyframe_image_path(raw_path)
    if resolved and resolved.is_file():
        return resolved

    # 2. Fallback check within root_dir
    cleaned = raw_path.replace("\\", "/").strip().lstrip("/")
    if ".." in cleaned or cleaned.startswith("/"):
        return None

    # Check in KEYFRAMES_DIR
    target_kf = (Path(KEYFRAMES_DIR) / cleaned).resolve()
    try:
        if target_kf.is_file():
            return target_kf
    except (ValueError, RuntimeError):
        pass

    # Direct path within root_dir
    direct_path = (Path(root_dir) / cleaned).resolve()
    try:
        if direct_path.is_relative_to(Path(root_dir).resolve()) and direct_path.is_file():
            return direct_path
    except (ValueError, RuntimeError):
        pass

    return None


@app.get("/media/{path:path}")
def serve_media(path: str):
    resolved = _resolve_image_path(path)
    if resolved and resolved.is_file():
        return FileResponse(
            str(resolved),
            headers={
                "Cache-Control": "public, max-age=86400",
            }
        )

    parts = Path(path).stem.split("_")
    video_id = parts[0] if parts else "UNKNOWN"
    frame_name = Path(path).name
    raw_root = KEYFRAME_ROOTS.get(video_id.split("_")[0], KEYFRAMES_DIR) if video_id else KEYFRAMES_DIR
    expected_root = raw_root[0] if isinstance(raw_root, list) else raw_root
    expected_path = f"{expected_root}\\{path}"

    print(f"[Missing Keyframe] video_id={video_id}, frame_name={frame_name}, expected_path={expected_path}")

    svg = f"""<svg width="400" height="225" xmlns="http://www.w3.org/2000/svg" style="background:#1e293b; font-family:sans-serif;">
        <rect width="100%" height="100%" fill="#0f172a"/>
        <circle cx="200" cy="90" r="35" fill="#334155"/>
        <polygon points="190,75 215,90 190,105" fill="#38bdf8"/>
        <text x="200" y="150" fill="#f8fafc" font-size="14" font-weight="bold" text-anchor="middle">{frame_name}</text>
        <text x="200" y="175" fill="#94a3b8" font-size="12" text-anchor="middle">Video: {video_id}</text>
        <text x="200" y="198" fill="#f43f5e" font-size="11" text-anchor="middle">Image unavailable</text>
    </svg>"""
    return Response(content=svg, media_type="image/svg+xml")


# Mount static directory for frontend
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = Path(STATIC_DIR) / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "HCMC AI Video Retrieval API is running. Access /docs for Swagger."})
