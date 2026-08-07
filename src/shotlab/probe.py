import json
import subprocess
from fractions import Fraction
from pathlib import Path

from .models import VideoInfo


def probe_video(path: Path) -> VideoInfo:
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",  # Đếm thực tế
        "-show_entries", 
        # BẮT BUỘC phải có nb_read_frames ở đây
        "stream=avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration:format=duration",
        "-of", "json",
        str(path),
    ]
    
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"Không tìm thấy video stream trong file: {path}")
        
    stream = streams[0]
    
    # 1. Tinh FPS (Ưu tiên avg_frame_rate cho VFR, fallback r_frame_rate)
    ratio = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not ratio or ratio == "0/0":
        fps = 25.0  # Default fallback nếu video bị hỏng hoàn toàn fps
    else:
        fps = float(Fraction(ratio))
        
    # 2. Lấy số Frame chuẩn xác: Ưu tiên nb_read_frames (kết quả từ -count_frames)
    raw_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    
    if raw_count and raw_count != "N/A":
        frame_count = int(raw_count)
    else:
        # Nếu ffprobe không đếm được, dùng decord/cv2 đếm trực tiếp thay vì nhân mò
        frame_count = 0  

    # 3. Tính Duration
    raw_duration = stream.get("duration") or payload.get("format", {}).get("duration")
    if raw_duration and raw_duration != "N/A":
        duration = float(raw_duration)
    else:
        duration = frame_count / fps if fps > 0 else 0.0

    # Nếu vẫn không lấy được duration chuẩn, tính lại từ frame_count và fps
    if duration == 0.0 and frame_count > 0 and fps > 0:
        duration = frame_count / fps

    return VideoInfo(
        stem=path.stem,
        path=path.resolve(),
        fps=fps,
        frame_count=frame_count,
        duration=duration
    )