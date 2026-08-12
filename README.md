# Student starter — Shot Boundary Detection & Keyframe Extraction

## Mục tiêu

Hoàn thiện các TODO để xử lý 10–20 video và tạo:

```text
output/
├── keyframes/<video_id>/*.jpg
├── shots.json
├── keyframe_mapping.csv
├── benchmark.json
├── benchmark.csv
└── benchmark_summary.csv
```

Mọi frame dùng zero-based indexing; `end_frame` là inclusive. `timestamp = frame_idx / fps`.

## Setup (Conda không cần có trong PATH)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\make_sample_videos.ps1 -Count 10
```

Script tự dò đường dẫn mặc định của Miniconda/Anaconda. Không cần activate env.

## TODO

Tìm `TODO` trong `src/shotlab`:

1. `detectors/pyscenedetect.py`: ContentDetector, xử lý đúng exclusive/inclusive end.
2. `detectors/transnetv2_pytorch.py`: load model một lần và chuẩn hóa output.
3. `keyframes.py`: frame giữa và sampling shot dài.
4. `keyframes.py`: keyframe ID, timestamp, relative image path.
5. `extract.py`: FFmpeg exact-frame extraction, một lần scan cho mỗi video.

Phần probe, CLI, metadata writer và benchmark orchestration đã có sẵn.

## Chạy/test

```powershell
& 'C:\Users\trung\anaconda3\Scripts\conda.exe' run --no-capture-output -n shotlab-student pytest -q
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1
```

Để làm phần TransNetV2 PyTorch:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_transnetv2_pytorch.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 -Methods pyscenedetect,transnetv2 -Device cpu
```

Checkpoint `.pth` 30.5 MB nằm trong package bên thứ ba `transnetv2-pytorch==1.0.5`; không cài TensorFlow. Chỉ load checkpoint từ nguồn tin cậy vì `.pth` dùng Python serialization.

## Rubric gợi ý (10 điểm)

| Hạng mục | Điểm |
|---|---:|
| PySceneDetect đúng ranh giới/indexing | 2.0 |
| TransNetV2 PyTorch chạy và chuẩn hóa output | 2.0 |
| Keyframe giữa + long-shot sampling | 1.5 |
| FFmpeg trích đúng frame, một scan/video | 1.5 |
| JSON/CSV mapping hợp lệ, đường dẫn tồn tại | 1.0 |
| Benchmark công bằng trên cùng 10–20 video | 1.0 |
| Test, README và phân tích sai khác | 1.0 |

## Nội dung báo cáo

- Mô tả tập 10–20 video: duration, resolution, FPS, codec.
- Threshold của mỗi method; không dùng chung thang threshold.
- Bảng `shot_count`, `detection_seconds`, realtime factor theo video/method.
- Ít nhất ba ví dụ hai method bất đồng và nhận xét hard cut/fade/motion.
- Ghi rõ phần benchmark có loại trừ model load hay không.

Synthetic videos chỉ dùng smoke test. Kết luận nên dựa trên video thật.

---

## Vector Search & Temporal Navigation Portal (HCMC-AIC)

Chúng tôi đã bổ sung hệ thống API Backend và Dashboard Frontend phục vụ tìm kiếm Vector (Text-to-Image Search) và Điều hướng thời gian (Temporal Navigation) hỗ trợ nộp bài thi.

### 1. API Endpoints (Backend)

* **`POST /api/search`**:
  * Nhận câu mô tả văn bản từ người dùng.
  * Vector hóa câu truy vấn bằng mô hình **SigLIP** (`google/siglip-base-patch16-224`).
  * Thực hiện tìm kiếm Vector tương đồng trên cơ sở dữ liệu NumPy Vector DB (sắp xếp và tìm top 100 keyframe tương đồng nhất).
  * Kết hợp với file metadata CSV để trả về các thông tin: `keyframe_id`, `video_id`, `timestamp`, `frame_idx`, `similarity_percentage` và đường dẫn ảnh vật lý.
* **`GET /api/keyframes/neighbors`**:
  * Nhận `keyframe_id`.
  * Trả về danh sách keyframe lân cận của khung hình đó trong khoảng thời gian $\pm 10$ giây thuộc cùng video.
  * Phục vụ điều hướng tiến/lùi thời gian nhanh chóng cho người thi.
* **`POST /api/submit_proxy`**:
  * Hỗ trợ proxy gửi kết quả trực tiếp tới Server BTC nhằm tránh lỗi CORS trên trình duyệt.

### 2. Giao diện người dùng (Search Dashboard)

Giao diện được xây dựng bằng **HTML, JavaScript Vanilla và Custom CSS** với thiết kế hiện đại:
* **Cột lọc bên trái (Left Sidebar)**: Lọc kết quả theo danh sách Video, điều chỉnh số lượng kết quả hiển thị (Limit), ngưỡng tương đồng tối thiểu (Threshold), và cấu hình địa chỉ Server BTC / API Token cùng cơ chế tự động nộp bài (Auto-Submit).
* **Lưới kết quả ở giữa (Result Cards Grid)**: Hiển thị các keyframe dưới dạng card ảnh kèm Video Name, Timestamp, Score (%) và hiệu ứng phóng to ảnh mượt mà khi di chuột qua (`hover-zoom`).
* **Thanh dòng thời gian điều hướng (Temporal Navigation Timeline)**: Hiển thị danh sách các frame lân cận trong khoảng $\pm 10$ giây khi bấm vào một keyframe bất kỳ. Cho phép di chuyển nhanh và thay đổi frame hiện tại trực quan.
* **Nộp bài 1-click (One-click Submit)**: Copy chuỗi kết quả định dạng BTC (ví dụ: `L01_V001, 35.5`) hoặc tự động gửi kết quả trực tiếp sang máy chủ BTC.

### 3. Cách khởi chạy hệ thống

1. **Trích xuất Keyframe và Embedding** (nếu chưa chạy):
   ```powershell
   # Trích xuất keyframes
   python -m shotlab.cli --input data/real_videos --output output --methods pyscenedetect --primary pyscenedetect --device cpu
   
   # Vector hóa (SigLIP)
   python scripts/run_embed.py
   ```
2. **Khởi động Server Backend & Frontend**:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\start_server.ps1
   ```
3. Truy cập Dashboard tại địa chỉ: [http://localhost:8000](http://localhost:8000)

