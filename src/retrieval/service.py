import time
import hashlib
from typing import List, Dict, Any, Optional
from config import METADATA_DB_PATH, DEFAULT_TOP_N, DEFAULT_TOP_K
from src.retrieval.encoders import ModelManager
from src.retrieval.faiss_search import FAISSManager
from src.retrieval.fusion import FusionEngine, rerank
from src.db.metadata_db import MetadataDB

class RetrievalService:
    _instance: Optional["RetrievalService"] = None

    def __init__(self):
        self.model_manager = ModelManager.get_instance()
        self.faiss_manager = FAISSManager.get_instance()
        self.db = MetadataDB(METADATA_DB_PATH)
        # In-memory query pool cache for instant Next Top-K pagination
        # key: query_hash, value: (timestamp, List[Dict])
        self._query_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
        self._cache_ttl = 300  # 5 minutes

    @classmethod
    def get_instance(cls) -> "RetrievalService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_cache_key(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()

    def _cleanup_cache(self):
        now = time.time()
        expired_keys = [k for k, (ts, _) in self._query_cache.items() if now - ts > self._cache_ttl]
        for k in expired_keys:
            del self._query_cache[k]

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        offset: int = 0,
        video_ids: Optional[List[str]] = None,
        objects: Optional[List[str]] = None,
        top_n_candidates: int = DEFAULT_TOP_N,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        start_time = time.time()
        query_str = query.strip()
        if not query_str:
            return {
                "results": [],
                "total": 0,
                "offset": offset,
                "top_k": top_k,
                "query": query,
                "time_taken_ms": 0.0
            }

        cache_key = self._get_cache_key(query_str)
        cached_candidates = None

        if use_cache and cache_key in self._query_cache:
            ts, candidates = self._query_cache[cache_key]
            if time.time() - ts <= self._cache_ttl:
                cached_candidates = candidates

        if cached_candidates is None:
            # 1. Query Embeddings
            embeddings = self.model_manager.encode_query(query_str)

            # 2. 3-Way FAISS Search (Top-N for each)
            faiss_results = self.faiss_manager.search_all(embeddings, top_n=top_n_candidates)

            # 3. Reciprocal Rank Fusion (RRF)
            fused_candidates = FusionEngine.reciprocal_rank_fusion(
                siglip_results=faiss_results["siglip"],
                openclip_results=faiss_results["openclip"],
                caption_results=faiss_results["caption"]
            )

            # 4. Optional Reranking
            candidates = rerank(fused_candidates, query_str)

            # 5. Save to cache
            self._cleanup_cache()
            self._query_cache[cache_key] = (time.time(), candidates)
        else:
            candidates = cached_candidates

        # 6. Apply Filters (video_ids, objects) if provided
        filtered_candidates = candidates
        if video_ids or objects:
            # Fetch all candidate metadata for filtering
            cand_ids = [c["global_id"] for c in candidates]
            meta_map = self.db.get_by_global_ids(cand_ids)
            
            video_set = set(v.strip().lower() for v in video_ids) if video_ids else None
            obj_set = set(o.strip().lower() for o in objects) if objects else None

            temp = []
            for c in candidates:
                gid = c["global_id"]
                m = meta_map.get(gid)
                if not m:
                    continue
                if video_set and m.get("video_id", "").lower() not in video_set:
                    continue
                if obj_set:
                    m_tags = (m.get("objects_tag") or "").lower()
                    if not any(o in m_tags for o in obj_set):
                        continue
                temp.append(c)
            filtered_candidates = temp

        total_matches = len(filtered_candidates)

        # 7. Pagination (offset & limit)
        page_slice = filtered_candidates[offset : offset + top_k]
        target_ids = [item["global_id"] for item in page_slice]

        # 8. Fetch detailed metadata from SQLite
        metadata_map = self.db.get_by_global_ids(target_ids)

        # 9. Build final response objects
        results = []
        for item in page_slice:
            gid = item["global_id"]
            meta = metadata_map.get(gid, {})
            results.append({
                "global_id": gid,
                "frame_id": meta.get("frame_id", f"frame_{gid}"),
                "video_id": meta.get("video_id", "unknown"),
                "frame_idx": meta.get("frame_idx", 0),
                "timestamp": meta.get("timestamp", 0.0),
                "image_path": meta.get("image_path", ""),
                "image_url": f"/media/{meta.get('image_path', '')}",
                "objects_tag": meta.get("objects_tag", ""),
                "object_detail": meta.get("object_detail", ""),
                "caption": meta.get("caption", ""),
                "score": item["score"],
                "ranks": item.get("ranks", {}),
                "raw_scores": item.get("raw_scores", {}),
            })

        time_taken_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "results": results,
            "total": total_matches,
            "offset": offset,
            "top_k": top_k,
            "query": query,
            "time_taken_ms": time_taken_ms
        }
