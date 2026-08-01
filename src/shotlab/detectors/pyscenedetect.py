from ..models import Shot, VideoInfo
from scenedetect import detect, ContentDetector


class PySceneDetector:
    name = "pyscenedetect"

    def __init__(self, threshold: float = 27.0) -> None:
        self.threshold = threshold


    def detect(self, video: VideoInfo) -> list[Shot]:
        """TODO 1: dùng scenedetect.detect + ContentDetector.

        Quy ước output: zero-based, end_frame inclusive, shot_id bắt đầu từ 1.
        Video không có cut vẫn phải trả về đúng một shot phủ toàn video.

        Input: 
        video.path, video.video_id, video.fps, video.frame_count

        Output: 
        list[Shot] - Mỗi phần tử Shot trong danh sách phải có:
        - video_id: Mã video đang xử lý.
        - method: 'pyscenedetect'
        - shot_id: bắt đầu từ 1
        - start_frame: Chỉ số khung hình bắt đầu (0-based).
        - end_frame: Chỉ số khung hình kết thúc (0-based, inclusive).
        - fps: FPS của video.
        
        1. # Lấy thông tin của Video từ VideoInfo 
        2. Khởi tạo PyScenceDetector
        3. Detect Scene với cả video bằng cách truyền vào video.path và detector
        4. Chạy vòng lặp qua các Shot phát hiện được và chuyển đổi sang Shot Model
        5. Return list[Shot]
        """
        # 1. Gọi scenedetect.detect với ContentDetector
        detector = ContentDetector(threshold=self.threshold)
        scene_list = detect(str(video.path), detector)

        shots = []
        # 2. Lặp qua các cảnh phát hiện được và chuyển đổi sang Shot model
        for i, (start_timecode, end_timecode) in enumerate(scene_list):
            start_frame = start_timecode.get_frames()
            # Vì end_timecode của scenedetect là exclusive, ta trừ đi 1 để có end_frame inclusive
            end_frame = end_timecode.get_frames() - 1
            
            shot = Shot(
                video_id=video.video_id,
                method=self.name,
                shot_id=i + 1,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=video.fps
            )
            shots.append(shot)

        # 3. Trường hợp fallback nếu thư viện không phát hiện được bất kỳ cảnh nào
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
