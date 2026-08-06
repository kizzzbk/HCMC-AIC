from ..models import Shot, VideoInfo


class TransNetV2Detector:
    name = "transnetv2"

    def __init__(self, threshold: float = 0.5, device: str = "auto") -> None:
        """TODO 2a: import TransNetV2 từ transnetv2_pytorch và load model một lần."""
        self.threshold = threshold
        self.device = device
        # Lazy import để tránh nạp các thư viện nặng của PyTorch nếu không sử dụng thuật toán này
        # pyrefly: ignore [missing-import]
        from transnetv2_pytorch import TransNetV2
        self.model = TransNetV2(device=self.device)

    def detect(self, video: VideoInfo) -> list[Shot]:
        """TODO 2b: gọi model.detect_scenes và chuyển dict sang list[Shot]."""
        # 1. Thực hiện chạy mô hình phát hiện chuyển cảnh trên video
        # TransNetV2 yêu cầu đường dẫn dạng str, nên ta sử dụng str(video.path)
        scene_list = self.model.detect_scenes(str(video.path), threshold=self.threshold)

        shots = []
        # 2. Duyệt qua các cảnh phát hiện được để chuyển đổi sang cấu trúc Shot
        for i, scene in enumerate(scene_list):
            # TransNetV2 trả về start/end là các chỉ số khung hình inclusive
            shot = Shot(
                video_id=video.video_id,
                method=self.name,
                shot_id=i + 1,  # Hệ thống yêu cầu shot_id bắt đầu từ 1
                start_frame=scene["start_frame"],
                end_frame=scene["end_frame"],
                fps=video.fps
            )
            shots.append(shot)

        # 3. Fallback nếu không có chuyển cảnh nào được phát hiện
        if not shots:
            shot = Shot(
                video_id=video.video_id,
                method=self.name,
                shot_id=1,
                start_frame=0,
                end_frame=video.frame_count - 1,
                fps=video.fps
            )
            shots.append(shot)

        return shots
