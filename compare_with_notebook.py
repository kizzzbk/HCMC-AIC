import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch, faiss, os, sqlite3
from transformers import AutoProcessor, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np

from src.retrieval.service import RetrievalService

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Exact Notebook Code (from Download_Keyframe.ipynb Cells 9, 10, 12, 14, 16)
siglip2_model_name = "google/siglip2-base-patch16-naflex"
siglip2_processor = AutoProcessor.from_pretrained(siglip2_model_name)
siglip2_model = AutoModel.from_pretrained(siglip2_model_name).to(device)

openclip_model_name = "clip-ViT-B-32"
openclip_model = SentenceTransformer(openclip_model_name).to(device)

index_siglip2 = faiss.read_index("Faiss/faiss_siglip2.bin")
index_openclip = faiss.read_index("Faiss/faiss_openclip.bin")

def nb_embed_siglip2(text_query):
    inputs = siglip2_processor(text=text_query, return_tensors="pt").to(siglip2_model.device)
    text_inputs = {k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
    with torch.no_grad():
        text_encoder_output = siglip2_model.text_model(**text_inputs)
        text_features = text_encoder_output.pooler_output
    return text_features.cpu().numpy().astype('float32')

def nb_embed_openclip(text_query):
    return openclip_model.encode(text_query, convert_to_numpy=True).astype('float32')

def nb_reciprocal_rank_fusion(results_lists, k=60):
    fused_scores = {}
    for results_list in results_lists:
        for rank, item_id in enumerate(results_list):
            score = 1.0 / (k + rank + 1)
            fused_scores[item_id] = fused_scores.get(item_id, 0.0) + score
    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, score in sorted_items]

def nb_perform_multimodal_search(query_text, num_results=10):
    vec1 = nb_embed_siglip2(query_text).reshape(1, -1)
    vec2 = nb_embed_openclip(query_text).reshape(1, -1)

    D1, I1 = index_siglip2.search(vec1, num_results)
    results_siglip2 = [str(idx) for idx in I1[0] if idx != -1]

    D2, I2 = index_openclip.search(vec2, num_results)
    results_openclip = [str(idx) for idx in I2[0] if idx != -1]

    all_results_lists = []
    if results_siglip2: all_results_lists.append(results_siglip2)
    if results_openclip: all_results_lists.append(results_openclip)
    return nb_reciprocal_rank_fusion(all_results_lists, k=60)

# Compare with RetrievalService
srv = RetrievalService.get_instance()

test_query = """(two women and one man) sitting side-by-side, focused on playing Handpan. One person wearing a white shirt is seated between two others wearing black shirts. The background features a multi-compartment bookshelf filled with colorful books."""

nb_results = nb_perform_multimodal_search(test_query, num_results=10)
srv_results = srv.search(query=test_query, top_k=10, top_n_candidates=10, use_cache=False)

print("="*60)
print("NOTEBOOK EXACT RESULTS (IDs):")
print(nb_results[:10])
print("\nPROJECT SERVICE RESULTS (Global IDs):")
print([r['global_id'] for r in srv_results['results'][:10]])
print("="*60)

match = [int(x) for x in nb_results[:10]] == [r['global_id'] for r in srv_results['results'][:10]]
print(f"ARE RESULTS 100% IDENTICAL? -> {match}")
