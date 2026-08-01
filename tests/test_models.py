from shotlab.models import Shot


def test_shot_uses_inclusive_end_frame() -> None:
    shot = Shot("V", "method", 1, 100, 124, 25.0)
    assert shot.frame_count == 25
    assert shot.duration == 1.0
    assert shot.to_dict()["end_time"] == 5.0
