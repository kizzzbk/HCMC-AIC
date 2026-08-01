import json
import subprocess
from fractions import Fraction
from pathlib import Path

from .models import VideoInfo


def probe_video(path: Path) -> VideoInfo:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration:format=duration",
        "-of", "json", str(path),
    ]
    payload = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    stream = payload["streams"][0]
    ratio = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    fps = float(Fraction(ratio))
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0)
    raw_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    frame_count = int(raw_count) if raw_count and raw_count != "N/A" else round(duration * fps)
    return VideoInfo(path.stem, path.resolve(), fps, frame_count, duration or frame_count / fps)
