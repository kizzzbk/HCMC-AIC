import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch, faiss, os, sqlite3
from transformers import AutoProcessor, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np

from src.retrieval.service import RetrievalService

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test Queries from Download_Keyframe.ipynb (Cell 16 & Cell 18)
queries = [
    # 1. Cell 16 query
    """(two women and one man) sitting side-by-side, focused on playing Handpan. One person wearing a white shirt is seated between two others wearing black shirts. The background features a multi-compartment bookshelf filled with colorful books.""",
    
    # 2. Cell 18 query 1
    """The clip we're looking for shows two women feeding goats: one wearing a white t-shirt with a red shawl draped over her shoulders, the other a traditional purple striped long-sleeved shirt. Both are smiling, appearing to be enjoying themselves.""",
    
    # 3. Cell 18 query 2
    """The clip we're looking for begins with a chef arranging vegetarian spring rolls on a plate. The filling consists of rolled green vegetables and tofu, wrapped in yellow and purple rice paper. The plate is decorated with green leaves and purple-yellow pansies, creating a refreshing and delicate feel.""",
    
    # 4. Cell 18 query 3 (Spacecraft)
    """Đây là phần giới thiệu việc phóng tàu vũ trụ tư nhân. Đoạn clip bắt đầu với hình ảnh 4 phi hành gia mặc áo đen. Một trong những nhiệm vụ dự kiến của tàu vũ trụ là nghiên cứu ánh sáng cực quang ở vùng cực""",

    # 5. Cell 18 query 4 (Pineapple harvest)
    """Đoạn clip là cảnh thu hoạch dứa ở miền Tây: một bà cụ ngồi bên giỏ dứa trò chuyện với cô gái mặc áo hồng quàng khăn rằn; xung quanh chất đầy dứa, phía sau có người phụ nữ đội nón lá cầm trái dứa và một chiếc ghe xanh đậu cạnh bờ, tạo không khí nông thôn mộc mạc, yên bình."""
]

srv = RetrievalService.get_instance()

for i, q in enumerate(queries, 1):
    print(f"\n{'='*70}")
    print(f"QUERY {i}:\n{q.strip()[:120]}...")
    print(f"{'-'*70}")
    
    res = srv.search(query=q, top_k=5)
    print(f"Top 5 Search Results (Time: {res['time_taken_ms']:.1f}ms):")
    for r in res['results'][:5]:
        gid = r['global_id']
        vid = r['video_id']
        fid = r['frame_id']
        score = r['score']
        url = r['image_url']
        print(f"  [Rank] Global ID #{gid} -> Video: {vid} | Frame: {fid} | Score: {score:.5f} | URL: {url}")
