import time
from collections import defaultdict
from pathlib import Path

from .detectors import create_detector
from .extract import extract_keyframes
from .keyframes import make_keyframes
from .metadata import write_benchmark, write_metadata
from .probe import probe_video
# from .deduplicate import deduplicate_by_histogram


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def discover_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    videos = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"Không có video trong {input_path}")
    return videos


def run_pipeline(
    input_path: Path,
    output_root: Path,
    methods: list[str],
    primary: str,
    pyscene_threshold: float,
    transnet_threshold: float,
    device: str,
    long_shot_seconds: float,
    sample_every_seconds: float,
    # hist_threshold: float = 0.95,
) -> None:
    if primary not in methods:
        raise ValueError("--primary phải nằm trong --methods")
    videos = [probe_video(path) for path in discover_videos(input_path)]
    num_videos = len(videos)
    cache = {}
    benchmark = []
    for method in methods:
        threshold = pyscene_threshold if method == "pyscenedetect" else transnet_threshold
        load_start = time.perf_counter()
        detector = create_detector(method, threshold, device)
        load_seconds = time.perf_counter() - load_start
        print('Đã load xong detector!')

        for i, video in enumerate(videos):
            print(f'Đang xử lí video {i+1}/{num_videos}')
            start = time.perf_counter()
            shots = detector.detect(video)
            elapsed = time.perf_counter() - start
            print(f'Đã xử lí xong video {i+1}/{num_videos}')
            
            cache[(method, video.video_id)] = shots
            benchmark.append({
                "method": method, "video_id": video.video_id, "fps": round(video.fps, 6),
                "frame_count": video.frame_count, "duration_seconds": round(video.duration, 6),
                "shot_count": len(shots), "model_load_seconds": round(load_seconds, 6),
                "detection_seconds": round(elapsed, 6),
                "realtime_factor": round(elapsed / video.duration, 6),
            })
        print(f'Đã xử lí xong {num_videos} videos')
        
    primary_shots = [shot for video in videos for shot in cache[(primary, video.video_id)]]
    keyframes = make_keyframes(primary_shots, long_shot_seconds, sample_every_seconds)
    by_video = defaultdict(list)

    print('Đang trích xuất Keyframe theo từng video_id ...')
    for frame in keyframes:
        by_video[frame.video_id].append(frame)

    print('Đang trích xuất từng keyframe và lưu vào folder output/ ...')
    for video in videos:
        extract_keyframes(video.path, by_video[video.video_id], output_root)
    write_metadata(output_root, primary_shots, keyframes)
    
    # print('Đang thực hiện lọc trùng lặp thô bằng Color Histogram (CPU)...')
    # deduplicate_by_histogram(output_root, threshold=hist_threshold)
    
    write_benchmark(output_root, benchmark)