import sys
import urllib.request
import json
import urllib.parse
import sqlite3

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.retrieval.encoders import ModelManager
from src.retrieval.faiss_search import FAISSManager
from src.retrieval.fusion import FusionEngine

def translate_vi_to_en(text):
    try:
        url = 'https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=' + urllib.parse.quote(text)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4) as response:
            res = json.loads(response.read().decode('utf-8'))
            return ''.join([part[0] for part in res[0] if part[0]])
    except Exception as e:
        return text

mm = ModelManager.get_instance()
fm = FAISSManager.get_instance()
conn = sqlite3.connect('metadata.db')
c = conn.cursor()

queries = [
    'xe cảnh sát màu trắng trên đường phố',
    'trận đấu bóng đá sân cỏ các cầu thủ tranh bóng',
    'phòng khách có ghế sofa và tivi',
    'người phụ nữ mặc áo đầm đi dạo trong công viên',
    'Đây là phần giới thiệu việc phóng tàu vũ trụ tư nhân. Đoạn clip bắt đầu với hình ảnh 4 phi hành gia mặc áo đen.'
]

for q in queries:
    trans_q = translate_vi_to_en(q)
    print(f"\n==================================================")
    print(f"VI: \"{q}\"")
    print(f"EN: \"{trans_q}\"")
    print(f"--------------------------------------------------")
    
    # 1. Search with translated text in OpenCLIP + SigLIP
    embs = mm.encode_query(trans_q)
    res = fm.search_all(embs, top_n=5)
    fused = FusionEngine.reciprocal_rank_fusion(res['siglip'], res['openclip'], res['caption'])
    print("Top 3 matches with translated query:")
    for item in fused[:3]:
        gid = item['global_id']
        c.execute('SELECT global_id, video_id, frame_id, image_path FROM metadata WHERE global_id = ?', (gid,))
        row = c.fetchone()
        sig = item['raw_scores'].get('siglip_score', 0)
        clip = item['raw_scores'].get('openclip_score', 0)
        print(f"  GID={gid} Score={item['score']:.5f} (SigLIP={sig:.3f}, CLIP={clip:.3f}): {row[1]} | {row[2]}")

conn.close()
