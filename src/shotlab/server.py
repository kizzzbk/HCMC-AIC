import os
import csv
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, AutoModel
import time
import httpx

app = FastAPI(title="Vector Search & Temporal Navigation API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model, processor, metadata, and embeddings
processor = None
model = None
device = "cpu"

keyframes_list = []
embeddings_matrix = None

# Paths
OUTPUT_ROOT = Path("output")
CSV_PATH = OUTPUT_ROOT / "keyframe_mapping.csv"
EMBEDDINGS_DIR = OUTPUT_ROOT / "embeddings"
FRONTEND_DIR = Path("frontend")

@app.on_event("startup")
def startup_event():
    global processor, model, device, keyframes_list, embeddings_matrix
    
    # 1. Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Startup] Backend server using device: {device}")
    
    # 2. Load SigLIP Model
    model_name = "google/siglip-base-patch16-224"
    print(f"[Startup] Loading SigLIP model '{model_name}'...")
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        print("[Startup] SigLIP model loaded successfully.")
    except Exception as e:
        print(f"[Startup] Error loading SigLIP model: {e}")
    
    # 3. Load DB Metadata
    if not CSV_PATH.exists():
        print(f"[Startup] Warning: Metadata CSV {CSV_PATH} not found. Please run the pipeline first.")
        return
        
    keyframes_list = []
    try:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                keyframes_list.append({
                    "keyframe_id": row["keyframe_id"],
                    "video_id": row["video_id"],
                    "shot_id": int(row["shot_id"]),
                    "ordinal": int(row["ordinal"]),
                    "frame_idx": int(row["frame_idx"]),
                    "timestamp": float(row["timestamp"]),
                    "image_path": row["image_path"]
                })
        print(f"[Startup] Loaded {len(keyframes_list)} keyframes from metadata.")
    except Exception as e:
        print(f"[Startup] Error reading metadata CSV: {e}")
        return
    
    # 4. Load Embeddings and align them
    try:
        # Group keyframes by video_id
        by_video = {}
        for kf in keyframes_list:
            by_video.setdefault(kf["video_id"], []).append(kf)
            
        all_vectors = []
        ordered_keyframes = []
        
        for video_id, video_kfs in by_video.items():
            # Sort by frame_idx to match the exact order processed in embed.py
            video_kfs.sort(key=lambda x: x["frame_idx"])
            
            npy_path = EMBEDDINGS_DIR / f"{video_id}.npy"
            if not npy_path.exists():
                print(f"[Startup] Warning: Embedding file not found for video {video_id}: {npy_path}")
                continue
                
            video_matrix = np.load(str(npy_path))
            if len(video_kfs) != video_matrix.shape[0]:
                print(f"[Startup] Warning: Keyframe count mismatch for {video_id}. Mapping has {len(video_kfs)}, matrix has {video_matrix.shape[0]}. Aligning to matrix size.")
                # Align elements
                limit = min(len(video_kfs), video_matrix.shape[0])
                for i in range(limit):
                    ordered_keyframes.append(video_kfs[i])
                    all_vectors.append(video_matrix[i])
            else:
                for i, kf in enumerate(video_kfs):
                    ordered_keyframes.append(kf)
                    all_vectors.append(video_matrix[i])
                
        if all_vectors:
            embeddings_matrix = np.vstack(all_vectors)
            keyframes_list = ordered_keyframes
            print(f"[Startup] Aligned and loaded embeddings matrix of shape: {embeddings_matrix.shape}")
        else:
            print("[Startup] Warning: No embeddings were loaded.")
    except Exception as e:
        print(f"[Startup] Error loading embeddings: {e}")

class SearchRequest(BaseModel):
    query: str
    video_filter: list[str] = None
    limit: int = 100
    threshold: float = 0.0

@app.post("/api/search")
def search_keyframes(req: SearchRequest):
    global processor, model, device, keyframes_list, embeddings_matrix
    
    if not keyframes_list or embeddings_matrix is None:
        raise HTTPException(status_code=500, detail="Database is not loaded yet or contains no embeddings.")
        
    if not req.query.strip():
        return {"results": [], "query": req.query, "results_count": 0, "search_time_ms": 0.0}
        
    start_time = time.perf_counter()
    
    try:
        # 1. Encode query
        inputs = processor(text=[req.query], padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            query_vector = text_features.cpu().numpy()[0]
            
        # 2. Compute similarities (dot product since vectors are L2-normalized)
        scores = np.dot(embeddings_matrix, query_vector)
        
        # 3. Filter and rank results
        results = []
        for idx, kf in enumerate(keyframes_list):
            score = float(scores[idx])
            score_percent = score * 100.0
            
            # Apply video filter if specified and not empty
            if req.video_filter and len(req.video_filter) > 0 and kf["video_id"] not in req.video_filter:
                continue
                
            # Apply threshold similarity filter
            if score_percent < req.threshold:
                continue
                
            results.append({
                **kf,
                "similarity_score": round(score, 6),
                "similarity_percentage": round(score_percent, 2)
            })
            
        # Sort by score descending
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_results = results[:req.limit]
        
        search_time = (time.perf_counter() - start_time) * 1000.0
        
        return {
            "query": req.query,
            "results_count": len(results),
            "search_time_ms": round(search_time, 2),
            "results": top_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/keyframes/neighbors")
def get_neighbors(keyframe_id: str):
    global keyframes_list
    
    # Find target keyframe
    target_kf = None
    for kf in keyframes_list:
        if kf["keyframe_id"] == keyframe_id:
            target_kf = kf
            break
            
    if not target_kf:
        raise HTTPException(status_code=404, detail=f"Keyframe {keyframe_id} not found.")
        
    video_id = target_kf["video_id"]
    target_time = target_kf["timestamp"]
    
    # Find neighbors in the same video within 10 seconds before and after
    neighbors = []
    for kf in keyframes_list:
        if kf["video_id"] == video_id:
            time_diff = kf["timestamp"] - target_time
            if -10.0 <= time_diff <= 10.0:
                neighbors.append({
                    **kf,
                    "time_difference": round(time_diff, 2)
                })
                
    # Sort neighbors by frame_idx (temporal order)
    neighbors.sort(key=lambda x: x["frame_idx"])
    
    return {
        "target_keyframe_id": keyframe_id,
        "video_id": video_id,
        "target_timestamp": target_time,
        "neighbors": neighbors
    }

class ProxyRequest(BaseModel):
    url: str
    token: str = ""
    payload: dict = None
    method: str = "POST"

@app.post("/api/submit_proxy")
async def submit_proxy(req: ProxyRequest):
    headers = {}
    if req.token:
        headers["Authorization"] = f"Bearer {req.token}"
        headers["X-API-Token"] = req.token
        
    try:
        async with httpx.AsyncClient() as client:
            if req.method.upper() == "POST":
                response = await client.post(req.url, json=req.payload, headers=headers, timeout=10.0)
            else:
                # GET request
                response = await client.get(req.url, params=req.payload, headers=headers, timeout=10.0)
                
            return {
                "status_code": response.status_code,
                "content": response.text,
                "headers": dict(response.headers)
            }
    except Exception as e:
        return {
            "status_code": 500,
            "content": f"Submission failed to connect: {str(e)}",
            "headers": {}
        }

@app.get("/api/videos")
def get_videos():
    """Retrieve distinct video_ids for sidebar filtering"""
    global keyframes_list
    videos = sorted(list(set(kf["video_id"] for kf in keyframes_list)))
    return {"videos": videos}

# Serve the static keyframe files at /static/
if OUTPUT_ROOT.exists():
    app.mount("/static", StaticFiles(directory=str(OUTPUT_ROOT)), name="static")

# Serve the frontend files
@app.get("/")
def get_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return "Frontend index.html not found. Please place index.html in the frontend directory."

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
