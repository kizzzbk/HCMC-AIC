from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from config import RRF_K, WEIGHT_SIGLIP, WEIGHT_OPENCLIP, WEIGHT_CAPTION

class FusionEngine:
    @staticmethod
    def reciprocal_rank_fusion(
        siglip_results: List[Tuple[int, float]],
        openclip_results: List[Tuple[int, float]],
        caption_results: Optional[List[Tuple[int, float]]] = None,
        k: int = RRF_K,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF):
        RRF(d) = sum_{m} (w_m / (k + rank_m(d)))
        """
        if weights is None:
            weights = {
                "siglip": WEIGHT_SIGLIP,
                "openclip": WEIGHT_OPENCLIP,
                "caption": WEIGHT_CAPTION,
            }

        scores = defaultdict(float)
        ranks_breakdown = defaultdict(dict)
        raw_scores = defaultdict(dict)

        # 1. SigLIP rank contribution
        for rank, (cand_id, raw_score) in enumerate(siglip_results, start=1):
            contrib = weights.get("siglip", 1.0) / (k + rank)
            scores[cand_id] += contrib
            ranks_breakdown[cand_id]["siglip_rank"] = rank
            raw_scores[cand_id]["siglip_score"] = raw_score

        # 2. OpenCLIP rank contribution
        for rank, (cand_id, raw_score) in enumerate(openclip_results, start=1):
            contrib = weights.get("openclip", 1.0) / (k + rank)
            scores[cand_id] += contrib
            ranks_breakdown[cand_id]["openclip_rank"] = rank
            raw_scores[cand_id]["openclip_score"] = raw_score

        # 3. Caption rank contribution (if available)
        if caption_results:
            for rank, (cand_id, raw_score) in enumerate(caption_results, start=1):
                contrib = weights.get("caption", 1.0) / (k + rank)
                scores[cand_id] += contrib
                ranks_breakdown[cand_id]["caption_rank"] = rank
                raw_scores[cand_id]["caption_score"] = raw_score

        # Sort descending by fused RRF score
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for cand_id, score in sorted_candidates:
            results.append({
                "global_id": cand_id,
                "score": round(score, 6),
                "ranks": ranks_breakdown[cand_id],
                "raw_scores": raw_scores[cand_id],
            })

        return results

    @staticmethod
    def weighted_score_fusion(
        siglip_results: List[Tuple[int, float]],
        openclip_results: List[Tuple[int, float]],
        caption_results: Optional[List[Tuple[int, float]]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Alternative: Min-Max normalized score fusion
        """
        if weights is None:
            weights = {"siglip": 0.45, "openclip": 0.45, "caption": 0.1}

        def min_max_norm(res_list: List[Tuple[int, float]]) -> Dict[int, float]:
            if not res_list:
                return {}
            vals = [s for _, s in res_list]
            min_v, max_v = min(vals), max(vals)
            diff = max_v - min_v if max_v != min_v else 1.0
            return {cid: (s - min_v) / diff for cid, s in res_list}

        norm_siglip = min_max_norm(siglip_results)
        norm_openclip = min_max_norm(openclip_results)
        norm_caption = min_max_norm(caption_results) if caption_results else {}

        all_ids = set(norm_siglip.keys()) | set(norm_openclip.keys()) | set(norm_caption.keys())
        fused = []

        for cid in all_ids:
            score = (
                norm_siglip.get(cid, 0.0) * weights.get("siglip", 0.45) +
                norm_openclip.get(cid, 0.0) * weights.get("openclip", 0.45) +
                norm_caption.get(cid, 0.0) * weights.get("caption", 0.1)
            )
            fused.append({"global_id": cid, "score": round(score, 6)})

        return sorted(fused, key=lambda x: x["score"], reverse=True)


def rerank(candidates: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Reranking abstraction for future expansion (e.g. Cross-Encoder / LLM reranker).
    Default implementation keeps the RRF score ordering.
    """
    return candidates
