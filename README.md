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
