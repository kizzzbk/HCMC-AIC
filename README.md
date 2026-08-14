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

## Hướng dẫn chạy YOLOv11 Object Detection

Module `src.shotlab.extractors.yolo_detect` hỗ trợ nhận diện các vật thể từ một thư mục chứa ảnh (ví dụ: thư mục keyframes đã trích xuất) bằng mô hình **YOLOv11** và lưu thông tin bounding box, tên lớp (class name), ID lớp và độ tin cậy ra file JSON.

### 1. Chạy qua Command Line (CLI)

#### Xử lý một thư mục ảnh duy nhất:
```bash
python -m src.shotlab.extractors.yolo_detect \
    --input-dir Keyframe/L21/output/keyframes/L21_V001 \
    --output Keyframe/L21/output/keyframes_output/L21_V001.json \
    --model yolo11n.pt \
    --conf 0.25 \
    --batch-size 16
```

#### Xử lý tất cả các thư mục con (chế độ `--batch` cho tập keyframes):
```bash
python -m src.shotlab.extractors.yolo_detect \
    --input-dir Keyframe/L21/output/keyframes \
    --output Keyframe/L21/output/keyframes_output \
    --batch \
    --model yolo11n.pt \
    --conf 0.25 \
    --batch-size 16
```
*(Nếu muốn lưu cả ảnh đã vẽ bounding box, thêm cờ `--save-images`)*

#### Các tham số:
- `-i`, `--input-dir` *(Bắt buộc)*: Đường dẫn đến thư mục chứa ảnh (hoặc thư mục chứa các thư mục con nếu dùng `--batch`).
- `-o`, `--output` *(Bắt buộc)*: Đường dẫn file JSON đầu ra (hoặc thư mục đầu ra nếu dùng `--batch`).
- `--batch`: Cờ xử lý duyệt qua tất cả các thư mục con (ví dụ `L21_V001`, `L21_V002`,...) và lưu kết quả theo tên từng thư mục vào `--output`.
- `-m`, `--model`: Tên hoặc đường dẫn mô hình YOLOv11 (`yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt` - mặc định: `yolo11n.pt`).
- `-c`, `--conf`: Ngưỡng độ tin cậy confidence threshold (mặc định: `0.25`).
- `--iou`: Ngưỡng NMS IoU threshold (mặc định: `0.45`).
- `-b`, `--batch-size`: Kích thước batch khi suy luận ảnh (mặc định: `16`).
- `-d`, `--device`: Thiết bị chạy (`cpu`, `cuda`, `0`, v.v.). Mặc định tự động chọn.
- `--save-images`: Cờ lưu ảnh đã được vẽ bounding box.

### 2. Sử dụng trong mã nguồn Python (Python API)

```python
from src.shotlab.extractors.yolo_detect import detect_batch_directories

# Xử lý tất cả các thư mục con trong keyframes và lưu vào keyframes_output
summary = detect_batch_directories(
    keyframes_root="Keyframe/L21/output/keyframes",
    output_root="Keyframe/L21/output/keyframes_output",
    model_path="yolo11n.pt",
    conf_threshold=0.25,
    batch_size=16,
    save_annotated=False  # Đặt True nếu muốn lưu cả ảnh có vẽ bounding box
)
```

### 3. Cấu trúc file JSON kết quả đầu ra

```json
{
  "model_path": "yolo11n.pt",
  "conf_threshold": 0.25,
  "iou_threshold": 0.45,
  "total_images": 1,
  "results": [
    {
      "image_name": "example.jpg",
      "relative_path": "example.jpg",
      "absolute_path": "/path/to/example.jpg",
      "detections_count": 2,
      "detections": [
        {
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.9125,
          "bbox": [150.25, 80.1, 320.0, 450.5]
        },
        {
          "class_id": 2,
          "class_name": "car",
          "confidence": 0.854,
          "bbox": [400.0, 200.0, 650.5, 380.2]
        }
      ]
    }
  ]
}
```

