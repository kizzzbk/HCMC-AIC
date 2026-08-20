import sys
import sqlite3
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from src.retrieval.encoders import ModelManager
from src.retrieval.faiss_search import FAISSManager
from src.retrieval.fusion import FusionEngine

mm = ModelManager.get_instance()
fm = FAISSManager.get_instance()
conn = sqlite3.connect('metadata.db')
c = conn.cursor()

def run_test(q_label, q_text):
    print(f"\n=== Query ({q_label}): '{q_text}' ===")
    embs = mm.encode_query(q_text)
    res = fm.search_all(embs, top_n=5)
    fused = FusionEngine.reciprocal_rank_fusion(res['siglip'], res['openclip'], res['caption'])
    for item in fused[:3]:
        gid = item['global_id']
        c.execute('SELECT global_id, video_id, frame_id, image_path FROM metadata WHERE global_id = ?', (gid,))
        row = c.fetchone()
        sig_score = item['raw_scores'].get('siglip_score', 0)
        clip_score = item['raw_scores'].get('openclip_score', 0)
        print(f"  GID={gid} Score={item['score']:.5f} (SigLIP={sig_score:.3f}, CLIP={clip_score:.3f}): {row}")

run_test('Vietnamese', 'xe cảnh sát màu trắng trên đường phố')
run_test('English', 'white police car on the street')

run_test('Vietnamese', 'trận đấu bóng đá sân cỏ')
run_test('English', 'soccer football match on grass pitch')

run_test('Vietnamese', '4 phi hành gia mặc áo đen phóng tàu vũ trụ')
run_test('English', 'four astronauts wearing black suits private spacecraft launch')

conn.close()
