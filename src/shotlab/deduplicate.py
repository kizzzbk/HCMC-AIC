import csv
import cv2
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def compare_histograms(img_path1: Path, img_path2: Path) -> float:
    img1 = cv2.imread(str(img_path1))
    img2 = cv2.imread(str(img_path2))
    
    if img1 is None or img2 is None:
        # Nếu không đọc được ảnh, trả về độ tương đồng bằng 0
        return 0.0
    
    # Chuyển sang không gian màu HSV để phân tích màu sắc tốt hơn
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    # Tính toán Histogram cho 2 kênh H và S
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
    # Chuẩn hóa
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # So sánh tương quan (correlation), giá trị từ -1 đến 1. 1 nghĩa là giống hệt.
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity


def deduplicate_by_histogram(output_root: Path, threshold: float = 0.95) -> None:
    csv_path = output_root / "keyframe_mapping.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file mapping: {csv_path}. Hãy chạy pipeline trích xuất keyframe trước."
        )

    # 1. Đọc danh sách keyframe gốc
    keyframes = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            # Thêm cột flag mặc định
            row["is_duplicate_hist"] = "0"
            keyframes.append(row)

    if "is_duplicate_hist" not in fieldnames:
        fieldnames.append("is_duplicate_hist")

    # 2. Phân nhóm theo video_id
    by_video = defaultdict(list)
    for row in keyframes:
        by_video[row["video_id"]].append(row)

    deleted_count = 0

    # 3. Lọc trùng lặp cuốn chiếu cho từng video
    for video_id, video_rows in by_video.items():
        # Sắp xếp theo frame_idx để đảm bảo so sánh tuần tự thời gian
        video_rows.sort(key=lambda x: int(x["frame_idx"]))
        
        if not video_rows:
            continue
            
        anchor_idx = 0
        curr_idx = 1
        
        while curr_idx < len(video_rows):
            anchor_row = video_rows[anchor_idx]
            curr_row = video_rows[curr_idx]
            
            img_path1 = output_root / anchor_row["image_path"]
            img_path2 = output_root / curr_row["image_path"]
            
            # Tính độ tương đồng màu sắc
            similarity = compare_histograms(img_path1, img_path2)
            
            if similarity >= threshold:
                # Đánh dấu trùng lặp
                curr_row["is_duplicate_hist"] = "1"
                # Xóa file vật lý
                img_path2.unlink(missing_ok=True)
                deleted_count += 1
            else:
                # Cập nhật anchor mới
                anchor_idx = curr_idx
                
            curr_idx += 1

    # 4. Ghi đè lại file keyframe_mapping.csv gốc (chứa toàn bộ thông tin và flag)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(keyframes)
    print(f"[Histogram] Đã dọn dẹp và xóa {deleted_count} ảnh trùng lặp thô.")

    # 5. Tạo file keyframe_mapping_clean.csv chỉ chứa các ảnh sạch
    clean_keyframes = [row for row in keyframes if row["is_duplicate_hist"] == "0"]
    
    # Chuẩn hóa lại ordinal và rename ảnh trên đĩa để tránh khoảng trống (gaps)
    clean_by_video = defaultdict(list)
    for row in clean_keyframes:
        clean_by_video[row["video_id"]].append(row)
        
    for video_id, video_rows in clean_by_video.items():
        # Phân nhóm theo shot_id để reset ordinal
        by_shot = defaultdict(list)
        for row in video_rows:
            by_shot[int(row["shot_id"])].append(row)
            
        for shot_id, shot_rows in by_shot.items():
            shot_rows.sort(key=lambda x: int(x["frame_idx"]))
            for new_ordinal, row in enumerate(shot_rows):
                old_keyframe_id = row["keyframe_id"]
                new_keyframe_id = f"{old_keyframe_id[:-2]}{new_ordinal:02d}"
                
                old_image_path = Path(row["image_path"])
                new_image_path = Path(f"keyframes/{video_id}/{new_keyframe_id}.jpg")
                
                # Thực hiện đổi tên file vật lý nếu đường dẫn thay đổi
                if old_image_path != new_image_path:
                    old_file = output_root / old_image_path
                    new_file = output_root / new_image_path
                    if old_file.exists():
                        new_file.parent.mkdir(parents=True, exist_ok=True)
                        old_file.rename(new_file)
                
                # Cập nhật lại trong metadata dict
                row["ordinal"] = str(new_ordinal)
                row["keyframe_id"] = new_keyframe_id
                row["image_path"] = str(new_image_path).replace("\\", "/")

    # 6. Ghi file clean mapping mới
    clean_csv_path = output_root / "keyframe_mapping_clean.csv"
    # Cột của clean csv không cần lưu cột flag 'is_duplicate_hist'
    clean_fieldnames = [c for c in fieldnames if c != "is_duplicate_hist"]
    
    with clean_csv_path.open("w", encoding="utf-8", newline="") as f:
        # Lọc các trường trong dict để chỉ lưu clean_fieldnames
        writer = csv.DictWriter(f, fieldnames=clean_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean_keyframes)
        
    print(f"[Histogram] Đã lưu mapping sạch tại: {clean_csv_path}")

    # 7. Đồng bộ hóa với file shots.json nếu có
    shots_json_path = output_root / "shots.json"
    if shots_json_path.exists():
        try:
            with shots_json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            clean_map = {(row["video_id"], int(row["shot_id"]), int(row["frame_idx"])): row for row in clean_keyframes}
            new_keyframes_json = []
            
            for kf in data.get("keyframes", []):
                key = (kf["video_id"], int(kf["shot_id"]), int(kf["frame_idx"]))
                if key in clean_map:
                    row = clean_map[key]
                    kf["ordinal"] = int(row["ordinal"])
                    kf["keyframe_id"] = row["keyframe_id"]
                    kf["image_path"] = row["image_path"]
                    new_keyframes_json.append(kf)
            
            data["keyframes"] = new_keyframes_json
            with shots_json_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("[Histogram] Đã đồng bộ thành công metadata keyframes trong shots.json.")
        except Exception as e:
            print(f"[Histogram] Cảnh báo: Lỗi đồng bộ hóa với shots.json: {e}")



def deduplicate_by_cosine(output_root: Path, threshold: float = 0.90) -> None:
    clean_csv_path = output_root / "keyframe_mapping_clean.csv"
    if not clean_csv_path.exists():
        print("Không tìm thấy file clean mapping. Hãy chạy bước lọc Histogram và Embed trước.")
        return

    # 1. Đọc clean keyframes
    keyframes = []
    with clean_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            keyframes.append(row)

    if not keyframes:
        print("Không có keyframe nào để lọc Cosine.")
        return

    # 2. Phân nhóm keyframe theo video_id
    by_video = defaultdict(list)
    for idx, row in enumerate(keyframes):
        row["global_index"] = idx  # Để theo dõi vị trí tương đối
        by_video[row["video_id"]].append(row)

    deleted_count = 0
    kept_rows = []

    # 3. Lọc trùng lặp cuốn chiếu bằng Cosine Similarity
    for video_id, video_rows in by_video.items():
        # Sắp xếp tăng dần theo frame_idx
        video_rows.sort(key=lambda x: int(x["frame_idx"]))
        
        # Load file embedding của video tương ứng
        npy_path = output_root / "embeddings" / f"{video_id}.npy"
        if not npy_path.exists():
            print(f"Cảnh báo: Không tìm thấy file embedding cho video {video_id} tại {npy_path}. Bỏ qua video này.")
            # Giữ lại toàn bộ hàng của video này
            for row in video_rows:
                kept_rows.append(row)
            continue
            
        embeddings = np.load(str(npy_path))
        
        # Đảm bảo số lượng hàng trong npy khớp với số lượng keyframes của video này
        if len(embeddings) != len(video_rows):
            print(f"Cảnh báo: Số lượng embedding ({len(embeddings)}) khác số lượng keyframe ({len(video_rows)}) của video {video_id}. Bỏ qua lọc Cosine.")
            for row in video_rows:
                kept_rows.append(row)
            continue

        anchor_idx = 0
        curr_idx = 1
        video_kept_indices = [0]
        
        while curr_idx < len(video_rows):
            anchor_row = video_rows[anchor_idx]
            curr_row = video_rows[curr_idx]
            
            # Tính Cosine Similarity bằng tích vô hướng (do vector đã được chuẩn hóa L2)
            similarity = np.dot(embeddings[anchor_idx], embeddings[curr_idx])
            
            if similarity >= threshold:
                # Đánh dấu trùng lặp ngữ nghĩa -> Xóa file ảnh vật lý
                img_path = output_root / curr_row["image_path"]
                img_path.unlink(missing_ok=True)
                deleted_count += 1
            else:
                # Giữ lại
                video_kept_indices.append(curr_idx)
                anchor_idx = curr_idx
                
            curr_idx += 1
            
        # Lọc lại ma trận embedding và ghi đè
        cleaned_embeddings = embeddings[video_kept_indices]
        np.save(str(npy_path), cleaned_embeddings)
        
        # Thêm các hàng được giữ lại vào danh sách kết quả chung
        for idx in video_kept_indices:
            kept_rows.append(video_rows[idx])

    # 4. Sắp xếp lại danh sách keyframes mới
    # Chuẩn hóa lại ordinal và rename ảnh trên đĩa để tránh gaps
    kept_by_video = defaultdict(list)
    for row in kept_rows:
        kept_by_video[row["video_id"]].append(row)

    final_keyframes = []
    for video_id, video_rows in kept_by_video.items():
        by_shot = defaultdict(list)
        for row in video_rows:
            by_shot[int(row["shot_id"])].append(row)
            
        for shot_id, shot_rows in by_shot.items():
            shot_rows.sort(key=lambda x: int(x["frame_idx"]))
            for new_ordinal, row in enumerate(shot_rows):
                old_keyframe_id = row["keyframe_id"]
                new_keyframe_id = f"{old_keyframe_id[:-2]}{new_ordinal:02d}"
                
                old_image_path = Path(row["image_path"])
                new_image_path = Path(f"keyframes/{video_id}/{new_keyframe_id}.jpg")
                
                if old_image_path != new_image_path:
                    old_file = output_root / old_image_path
                    new_file = output_root / new_image_path
                    if old_file.exists():
                        new_file.parent.mkdir(parents=True, exist_ok=True)
                        old_file.rename(new_file)
                        
                row["ordinal"] = str(new_ordinal)
                row["keyframe_id"] = new_keyframe_id
                row["image_path"] = str(new_image_path).replace("\\", "/")
                
            final_keyframes.extend(shot_rows)

    # 5. Ghi đè lại file keyframe_mapping_clean.csv
    # Sắp xếp lại theo thứ tự ban đầu để đảm bảo tính tuần tự
    final_keyframes.sort(key=lambda x: (x["video_id"], int(x["shot_id"]), int(x["frame_idx"])))
    
    clean_fieldnames = [c for c in fieldnames if c != "global_index"]
    with clean_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=clean_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_keyframes)

    # 6. Đồng bộ hóa với file shots.json nếu có
    shots_json_path = output_root / "shots.json"
    if shots_json_path.exists():
        try:
            with shots_json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            clean_map = {(row["video_id"], int(row["shot_id"]), int(row["frame_idx"])): row for row in final_keyframes}
            new_keyframes_json = []
            
            for kf in data.get("keyframes", []):
                key = (kf["video_id"], int(kf["shot_id"]), int(kf["frame_idx"]))
                if key in clean_map:
                    row = clean_map[key]
                    kf["ordinal"] = int(row["ordinal"])
                    kf["keyframe_id"] = row["keyframe_id"]
                    kf["image_path"] = row["image_path"]
                    new_keyframes_json.append(kf)
            
            data["keyframes"] = new_keyframes_json
            with shots_json_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("[Cosine Similarity] Đã đồng bộ thành công metadata keyframes trong shots.json.")
        except Exception as e:
            print(f"[Cosine Similarity] Cảnh báo: Lỗi đồng bộ hóa với shots.json: {e}")

    print(f"[Cosine Similarity] Đã hoàn thành lọc tinh, xóa {deleted_count} ảnh trùng lặp ngữ nghĩa.")