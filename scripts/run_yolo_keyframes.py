import sys
from pathlib import Path

# Thêm đường dẫn gốc của dự án vào sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.shotlab.extractors.yolo_detect import detect_batch_directories


def main():
    keyframes_dir = project_root / "Keyframe" / "L21" / "output" / "keyframes"
    output_dir = project_root / "Keyframe" / "L21" / "output" / "keyframes_output"

    print(f"Đang chạy YOLOv11 cho các thư mục keyframes trong: {keyframes_dir}")
    print(f"Kết quả sẽ được lưu vào: {output_dir}")

    summary = detect_batch_directories(
        keyframes_root=keyframes_dir,
        output_root=output_dir,
        model_path="yolo11n.pt",
        conf_threshold=0.6,
        batch_size=16,
        save_annotated=False,  # Đổi thành True nếu muốn lưu cả các file ảnh có vẽ bounding box
    )

    print(f"\n[Thành công] Đã xử lý {len(summary)} thư mục keyframe thành công!")
    for folder_name, info in summary.items():
        print(f"  - {folder_name}: {info['total_images']} ảnh -> {info['output_json']}")


if __name__ == "__main__":
    main()
