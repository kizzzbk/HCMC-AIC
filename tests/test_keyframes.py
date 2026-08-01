from shotlab.keyframes import select_frame_indices
from shotlab.models import Shot


def test_short_shot_uses_midpoint() -> None:
    shot = Shot("V001", "pyscenedetect", 1, 100, 199, 25.0)
    assert select_frame_indices(shot, 12, 8) == [149]


def test_long_shot_sampling_is_valid() -> None:
    shot = Shot("V001", "pyscenedetect", 1, 0, 999, 25.0)
    indices = select_frame_indices(shot, 12, 8)
    assert 499 in indices and len(indices) > 1
    assert indices == sorted(set(indices))
    assert all(0 <= index <= 999 for index in indices)


def test_single_frame_shot() -> None:
    shot = Shot("V001", "pyscenedetect", 1, 42, 42, 25.0)
    assert select_frame_indices(shot, 12, 8) == [42]
