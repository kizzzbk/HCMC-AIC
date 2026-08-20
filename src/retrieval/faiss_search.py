import os
import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict
from config import FAISS_SIGLIP_PATH, FAISS_OPENCLIP_PATH, FAISS_CAPTION_PATH

class FAISSManager:
    _instance: Optional["FAISSManager"] = None

    def __init__(self):
        self.index_siglip = None
        self.index_openclip = None
        self.index_caption = None
        self._load_indices()

    def _load_indices(self):
        # Load SigLIP Index
        if os.path.exists(FAISS_SIGLIP_PATH):
            print(f"[FAISS] Loading SigLIP2 index from {FAISS_SIGLIP_PATH}...")
            self.index_siglip = faiss.read_index(FAISS_SIGLIP_PATH)
            print(f"[FAISS] SigLIP2 loaded: {self.index_siglip.ntotal} vectors, d={self.index_siglip.d}")
        else:
            print(f"[FAISS] Warning: SigLIP2 index not found at {FAISS_SIGLIP_PATH}")

        # Load OpenCLIP Index
        if os.path.exists(FAISS_OPENCLIP_PATH):
            print(f"[FAISS] Loading OpenCLIP index from {FAISS_OPENCLIP_PATH}...")
            self.index_openclip = faiss.read_index(FAISS_OPENCLIP_PATH)
            print(f"[FAISS] OpenCLIP loaded: {self.index_openclip.ntotal} vectors, d={self.index_openclip.d}")
        else:
            print(f"[FAISS] Warning: OpenCLIP index not found at {FAISS_OPENCLIP_PATH}")

        # Load Caption Index
        if os.path.exists(FAISS_CAPTION_PATH):
            print(f"[FAISS] Loading Caption index from {FAISS_CAPTION_PATH}...")
            self.index_caption = faiss.read_index(FAISS_CAPTION_PATH)
            print(f"[FAISS] Caption index loaded: {self.index_caption.ntotal} vectors, d={self.index_caption.d}")
        else:
            print(f"[FAISS] Info: Caption index not found at {FAISS_CAPTION_PATH} (optional)")

    @classmethod
    def get_instance(cls) -> "FAISSManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def search_index(self, index: Optional[faiss.Index], query_vec: Optional[np.ndarray], top_n: int = 100) -> List[Tuple[int, float]]:
        if index is None or query_vec is None:
            return []
        
        if query_vec.shape[1] != index.d:
            raise ValueError(f"Query vector dimension ({query_vec.shape[1]}) does not match FAISS index dimension ({index.d})")

        scores, ids = index.search(query_vec, top_n)
        results = []
        for i in range(len(ids[0])):
            cand_id = int(ids[0][i])
            if cand_id != -1:
                results.append((cand_id, float(scores[0][i])))
        return results

    def search_siglip(self, query_vec: np.ndarray, top_n: int = 100) -> List[Tuple[int, float]]:
        return self.search_index(self.index_siglip, query_vec, top_n)

    def search_openclip(self, query_vec: np.ndarray, top_n: int = 100) -> List[Tuple[int, float]]:
        return self.search_index(self.index_openclip, query_vec, top_n)

    def search_caption(self, query_vec: np.ndarray, top_n: int = 100) -> List[Tuple[int, float]]:
        return self.search_index(self.index_caption, query_vec, top_n)

    def search_all(self, query_vectors: Dict[str, Optional[np.ndarray]], top_n: int = 100) -> Dict[str, List[Tuple[int, float]]]:
        return {
            "siglip": self.search_siglip(query_vectors["siglip"], top_n) if query_vectors.get("siglip") is not None else [],
            "openclip": self.search_openclip(query_vectors["openclip"], top_n) if query_vectors.get("openclip") is not None else [],
            "caption": self.search_caption(query_vectors["caption"], top_n) if query_vectors.get("caption") is not None else [],
        }
