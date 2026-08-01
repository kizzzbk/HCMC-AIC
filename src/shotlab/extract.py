import shutil
import subprocess
from pathlib import Path

from .models import Keyframe


def extract_keyframes(video_path: Path, frames: list[Keyframe], output_root: Path) -> None:
    '''
    0. Tạo output root
    1. Sắp xếp list[Keyframe] theo thứ tự tăng dần từ của list[keyframe_idx]
    2. Dùng ffmpeg trích xuất các frame sử dụng list[Keyframe]:
    - Output ra list các ảnh .jpg: List[.jpg]
    3. Đổi tên tương ứng cho List[.jpg] ánh xạ sang list[frame_idx]
    4. Ghi list[Keyframe] ra output root
    '''
    # Nếu danh sách frame cần trích xuất rỗng, thoát sớm
    if not frames:
        return

    # 1. Sắp xếp danh sách frames theo frame_idx tăng dần
    sorted_frames = sorted(frames, key=lambda x: x.frame_idx)

    # 2. Tạo một thư mục tạm thời để chứa ảnh trích xuất của FFmpeg
    temp_dir = output_root / "temp_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 3. Tạo bộ lọc select cho FFmpeg (ví dụ: select=eq(n\,10)+eq(n\,25))
    select_expr = "+".join(f"eq(n\\,{f.frame_idx})" for f in sorted_frames)
    vf_filter = f"select={select_expr}"

    # 4. Tạo lệnh FFmpeg chạy tiến trình
    # -vsync vfr để đảm bảo số lượng ảnh trùng với số lượng frame được chọn
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
        "-vf", vf_filter,
        "-vsync", "vfr",
        str(temp_dir / "%d.jpg")
    ]

    # 5. Gọi tiến trình con FFmpeg
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Lỗi FFmpeg: {result.stderr}")

    # 6. Đổi tên và di chuyển các file ảnh tạm sang thư mục đích thực tế
    for i, frame in enumerate(sorted_frames):
        temp_file = temp_dir / f"{i + 1}.jpg"
        dest_file = output_root / frame.image_path

        # Tạo thư mục cha của file đích nếu chưa tồn tại
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        if not temp_file.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise FileNotFoundError(f"Không tìm thấy file ảnh tạm do FFmpeg trích xuất: {temp_file}")

        # Di chuyển file từ thư mục tạm sang thư mục output chuẩn
        shutil.move(str(temp_file), str(dest_file))

    # 7. Xóa thư mục tạm sau khi hoàn thành công việc
    shutil.rmtree(temp_dir, ignore_errors=True)

    # 8. Kiểm tra phòng vệ (Defensive check): Đảm bảo tất cả các file đích đều đã được tạo thành công
    for frame in sorted_frames:
        dest_file = output_root / frame.image_path
        if not dest_file.exists():
            raise FileNotFoundError(f"Lỗi nghiêm trọng: File keyframe đích không tồn tại: {dest_file}")

