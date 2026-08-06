import csv
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


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

    # 2. Đọc file keyframe_mapping.csv để lấy danh sách ảnh
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

    all_embeddings = []

    # 4. Trích xuất đặc trưng theo từng lô (Batch)
    print(f"Đang trích xuất đặc trưng cho {len(keyframes_to_process)} ảnh Keyframe...")
    for i in tqdm(range(0, len(keyframes_to_process), batch_size)):
        batch = keyframes_to_process[i:i + batch_size]
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
            
            # Chuẩn hóa L2 cho các vector đặc trưng để tính Cosine Similarity nhanh bằng phép nhân ma trận
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            # Đẩy kết quả về CPU và lưu dưới dạng numpy array
            all_embeddings.append(features.cpu().numpy())

    # 5. Gom tất cả các lô lại thành 1 ma trận duy nhất và lưu lại
    embeddings_matrix = np.vstack(all_embeddings)
    print(f"Kích thước ma trận vector đặc trưng: {embeddings_matrix.shape}")

    npy_output_path = output_root / "embeddings.npy"
    np.save(str(npy_output_path), embeddings_matrix)
    print(f"Đã lưu thành công ma trận đặc trưng vào: {npy_output_path}")
