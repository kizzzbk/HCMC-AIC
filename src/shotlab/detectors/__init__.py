from typing import Protocol

from ..models import Shot, VideoInfo


class Detector(Protocol):
    name: str
    def detect(self, video: VideoInfo) -> list[Shot]: ...


def create_detector(name: str, threshold: float, device: str = "auto") -> Detector:
    if name == "pyscenedetect":
        from .pyscenedetect import PySceneDetector
        return PySceneDetector(threshold)
    if name == "transnetv2":
        from .transnetv2_pytorch import TransNetV2Detector
        return TransNetV2Detector(threshold, device)
    raise ValueError(f"Unknown detector: {name}")
