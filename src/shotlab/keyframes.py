from .models import Keyframe, Shot


def select_frame_indices(
    shot: Shot, long_shot_seconds: float = 3.0, sample_every_seconds: float = 1.0
) -> list[int]:
    """
    TODO 3: frame giữa + sampling bổ sung cho shot dài.

    Kết quả phải sorted, unique và luôn nằm trong [start_frame, end_frame].

    Input:
    - Shot 
    - Threshold phân biệt shot dài
    - sample_every_seconds: Khoảng cách trung bình để lấy các keyframe trong 1 shot dài

    Output:
    - list[int]: list thứ tự ordinal của các frame trong Shot 

    1. Lấy Frame ở giữa: ordinal = mid = start_frame + (end_frame - start_frame) // 2
    2. If Shot dài:
    Lấy theo tần suất 8.0s/frame: step = int(sample_every_seconds * shot.fps).
    for frame_idx in range (start_frame, end_frame, step) ---> start_frame + step // 2
    3. Lọc trùng ---> Set(List[idx])
    4. Trả về List[idx]: list thứ tự của các frame được chọn theo tần suất 8s
    """
    
    # 1. Luôn lấy frame ở chính giữa của shot
    mid = shot.start_frame + (shot.end_frame - shot.start_frame) // 2
    indices = {}

    # 2. Nếu shot dài hơn ngưỡng long_shot_seconds, thực hiện lấy mẫu bổ sung định kỳ
    if shot.duration >= long_shot_seconds:
        step = max(1, int(sample_every_seconds * shot.fps))
        # Lấy mẫu bắt đầu từ nửa bước nhảy đầu tiên (start_frame + step // 2)
        start_point = shot.start_frame + step // 2
        for f in range(start_point, shot.end_frame + 1, step):
            indices.add(f)
    
    else:
        indices = {shot.start_frame, mid, shot.end_frame} 

    # 3. Đảm bảo tất cả các chỉ số khung hình đều nằm trong khoảng hợp lệ [start_frame, end_frame]
    valid_indices = {f for f in indices if shot.start_frame <= f <= shot.end_frame}

    # 4. Trả về danh sách đã được lọc trùng và sắp xếp tăng dần
    return sorted(list(valid_indices))


def make_keyframes(shots: list[Shot], long_shot_seconds: float, sample_every_seconds: float) -> list[Keyframe]:
    """TODO 4: tạo ID/path ổn định và timestamp = frame_idx / fps.
    
    Input:
    - List tất cả các shot của tất cả các videos
    - Threshold cho shot dài
    - sample_every_seconds: Khoảng cách trung bình để lấy các keyframe trong 1 shot dài

    Output:
    - List[Keyframe]: Trích các Keyframe được chọn từ các Shot được tổng hợp ở trên
        - keyframe_id: Mã định danh duy nhất cho keyframe (video_id_method_shot_id_ordinal)
        - video_id: Mã video
        - shot_id: ID của shot
        - ordinal: Thứ tự của keyframe trong shot
        - frame_idx: Chỉ số frame
        - timestamp: Thời gian (tính bằng giây)
        - image_path: Đường dẫn tới file ảnh keyframe

    0. keyframes = List[Keyframe]
    1. for Shot in List[Shot]:
        keyframe_ordinal = select_frame_indices(Shot)

        for ordinal in keyframe_ordinal:
            new_keyframe = Keyframe()

            new_keyframe.keyframe_id = f"{shot.video_id}_{shot.method}_{shot.shot_id:04d}_{ordinal:02d}"
            new_keyframe.video_id = Shot.video_id
            new_keyframe.shot_id = Shot.shot_id
            new_keyframe.ordinal = ordinal
            new_keyframe.frame_idx = Shot.start_frame + ordinal
        
        keyframe.append(new_keyframe)
    """
    keyframes = []

    for shot in shots:
        # Lấy danh sách chỉ số khung hình được chọn cho shot hiện tại
        frame_indices = select_frame_indices(shot, long_shot_seconds, sample_every_seconds)

        for ordinal, frame_idx in enumerate(frame_indices):
            # 1. Tạo ID ổn định, chuẩn hóa độ dài để thuận tiện cho việc sắp xếp
            keyframe_id = f"{shot.video_id}_{shot.method}_{shot.shot_id:04d}_{ordinal:02d}"

            # 2. Tính toán timestamp (thời gian tính bằng giây)
            timestamp = frame_idx / shot.fps

            # 3. Tạo đường dẫn ảnh keyframe tương đối
            image_path = f"keyframes/{shot.video_id}/{keyframe_id}.jpg"

            # 4. Tạo đối tượng Keyframe hoàn chỉnh
            new_keyframe = Keyframe(
                keyframe_id=keyframe_id,
                video_id=shot.video_id,
                shot_id=shot.shot_id,
                ordinal=ordinal,
                frame_idx=frame_idx,
                timestamp=timestamp,
                image_path=image_path
            )
            keyframes.append(new_keyframe)

    return keyframes
