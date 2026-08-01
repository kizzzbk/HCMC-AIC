from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    path: Path
    fps: float
    frame_count: int
    duration: float


@dataclass(frozen=True)
class Shot:
    video_id: str
    method: str
    shot_id: int
    start_frame: int
    end_frame: int  # inclusive
    fps: float

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps

    def to_dict(self) -> dict:
        value = asdict(self)
        value.update(
            start_time=round(self.start_frame / self.fps, 6),
            end_time=round((self.end_frame + 1) / self.fps, 6),
            duration=round(self.duration, 6),
            frame_count=self.frame_count,
        )
        return value


@dataclass(frozen=True)
class Keyframe:
    keyframe_id: str
    video_id: str
    shot_id: int
    ordinal: int
    frame_idx: int
    timestamp: float
    image_path: str

    def to_dict(self) -> dict:
        value = asdict(self)
        value["timestamp"] = round(self.timestamp, 6)
        return value
