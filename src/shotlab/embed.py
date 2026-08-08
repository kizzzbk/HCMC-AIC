import csv
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from collections import defaultdict


def embed_keyframes(
    output_root: Path,
    model_name: str = "google/siglip-base-patch16-224",
    batch_size: int = 32,
    device: str = "auto"
) -> None:
    # 1. Xác định thiết bị chạy (CPU hay GPU CUDA)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị tính toán: {device}")

    # 2. Đọc file mapping sạch (keyframe_mapping_clean.csv) hoặc gốc (keyframe_mapping.csv)
    csv_path = output_root / "keyframe_mapping_clean.csv"
    if not csv_path.exists():
        csv_path = output_root / "keyframe_mapping.csv"
        
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file mapping: {csv_path}. Hãy chạy pipeline trích xuất keyframe trước."
        )

    keyframes_to_process = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keyframes_to_process.append(row)
    
    if not keyframes_to_process:
        print("Không tìm thấy keyframe nào trong file mapping để xử lý.")
        return

    # 3. Tải mô hình SigLIP từ Hugging Face
    print(f"Đang tải mô hình SigLIP: {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # 4. Phân nhóm keyframes theo video_id
    by_video = defaultdict(list)
    for row in keyframes_to_process:
        by_video[row["video_id"]].append(row)

    embeddings_dir = output_root / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    print(f"Đang trích xuất đặc trưng cho {len(keyframes_to_process)} ảnh Keyframe theo từng video...")

    # 5. Duyệt và trích xuất đặc trưng theo từng video
    for video_id, video_rows in by_video.items():
        # Đảm bảo sắp xếp đúng thứ tự để đồng nhất
        video_rows.sort(key=lambda x: int(x["frame_idx"]))
        
        print(f"-> Video {video_id}: có {len(video_rows)} keyframes cần xử lý.")
        video_embeddings = []
        
        for i in tqdm(range(0, len(video_rows), batch_size)):
            batch = video_rows[i:i + batch_size]
            batch_images = []

            for item in batch:
                img_path = output_root / item["image_path"]
                if not img_path.exists():
                    raise FileNotFoundError(f"Không tìm thấy ảnh keyframe: {img_path}")
                
                # Load ảnh bằng Pillow và chuyển về hệ màu RGB để đưa vào ViT
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)

            # Chạy qua mô hình SigLIP
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            with torch.no_grad():
                # Lấy đặc trưng ảnh từ mô hình
                features = model.get_image_features(**inputs)
                if hasattr(features, "pooler_output") and features.pooler_output is not None:
                    features = features.pooler_output
                
                # Chuẩn hóa L2 cho các vector đặc trưng để tính Cosine Similarity nhanh bằng phép nhân ma trận
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                
                # Đẩy kết quả về CPU và lưu dưới dạng numpy array
                video_embeddings.append(features.cpu().numpy())

        # Ghép các lô của video này lại và lưu
        if video_embeddings:
            video_matrix = np.vstack(video_embeddings)
            npy_path = embeddings_dir / f"{video_id}.npy"
            np.save(str(npy_path), video_matrix)
            print(f"   Đã lưu vector đặc trưng video {video_id} ({video_matrix.shape}) vào: {npy_path}")

    print("Hoàn thành trích xuất đặc trưng cho toàn bộ keyframes!")

