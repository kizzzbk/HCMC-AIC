"""
Keyframe Image Resolver
Source of truth: Download_Keyframe.ipynb (Cell 4 & Cell 5)
"""
import os
from pathlib import Path
from typing import Optional, List
from config import BASE_DIR, KEYFRAME_ROOTS

# Đọc động từ env, không dùng giá trị cứng lúc import
def _get_keyframes_dir() -> str:
    return os.getenv("KEYFRAMES_DIR", str(BASE_DIR / "keyframes"))


def get_folder_name(image_id: str) -> str:
    """
    Derives the subfolder name from the image_id based on the pattern in Download_Keyframe.ipynb (Cell 4):
    'LXX_vYYY_...' -> 'LXX_VYYY'
    """
    if not image_id:
        return ""
    # Strip extension if present
    clean_id = image_id[:-4] if image_id.lower().endswith(".jpg") else image_id
    parts = clean_id.split('_')
    if len(parts) >= 2:
        prefix = parts[0]
        version = parts[1]
        # Capitalize 'v' if it's the first character of the version part
        if version.startswith('v') and len(version) > 1:
            version = 'V' + version[1:]
        elif version.startswith('V') and len(version) > 1:
            version = 'V' + version[1:]
        return f"{prefix}_{version}"
    return ""


from typing import Optional, List, Union

# Cache chỉ lưu kết quả TÌM THẤY (không cache None)
# → nếu file chưa có lúc đầu, lần sau vẫn tìm lại được
_resolve_cache: dict = {}


def _cached_resolve_keyframe_path(clean_id_or_path: str) -> Optional[Path]:
    """Resolver có cache thủ công — chỉ cache kết quả hợp lệ, bỏ qua None."""
    if clean_id_or_path in _resolve_cache:
        return _resolve_cache[clean_id_or_path]
    result = _do_resolve(clean_id_or_path)
    if result is not None:
        _resolve_cache[clean_id_or_path] = result  # chỉ lưu khi tìm thấy
    return result


def _do_resolve(clean_id_or_path: str) -> Optional[Path]:
    """Internal resolver implementation."""
    # Sanitize path to prevent path traversal
    clean = str(clean_id_or_path).replace("\\", "/").strip().lstrip("/")
    if ".." in clean:
        return None

    # Extract base filename and image_id
    filename = clean.split("/")[-1]
    image_id = filename[:-4] if filename.lower().endswith(".jpg") else filename

    folder_name = get_folder_name(image_id)
    if not folder_name:
        return None

    prefix = folder_name.split("_")[0]  # 'L21', 'L22', 'L23', 'L24', 'L25', 'L26', 'L27', etc.

    candidate_roots: List[Path] = []
    
    # 1. Configured roots from KEYFRAME_ROOTS
    if prefix in KEYFRAME_ROOTS:
        roots: Union[Path, List[Path]] = KEYFRAME_ROOTS[prefix]
        if isinstance(roots, list):
            candidate_roots.extend(roots)
        else:
            candidate_roots.append(roots)

    # 2. Dynamic subfolder discovery for multipart datasets (e.g. L25.1, L25.2, L25.3, L26_a.1, etc.)
    try:
        for child in BASE_DIR.iterdir():
            if child.is_dir() and (child.name.startswith(f"{prefix}.") or child.name.startswith(f"{prefix}_")):
                candidate_roots.append(child / "keyframes")
                candidate_roots.append(child / "output" / "keyframes")
                candidate_roots.append(child)
    except (OSError, PermissionError):
        pass

    # 3. Standard fallback roots
    candidate_roots.append(BASE_DIR / prefix / "output" / "keyframes")
    candidate_roots.append(BASE_DIR / prefix / "keyframes")
    candidate_roots.append(BASE_DIR / prefix)
    candidate_roots.append(Path(_get_keyframes_dir()))  # đọc động từ env
    candidate_roots.append(BASE_DIR / "keyframes")
    candidate_roots.append(BASE_DIR)

    # Possible filename variants (original, lowercase v, uppercase V)
    id_variants = [image_id]
    if "_v" in image_id:
        id_variants.append(image_id.replace("_v", "_V"))
    elif "_V" in image_id:
        id_variants.append(image_id.replace("_V", "_v"))

    for root in candidate_roots:
        try:
            if not root.exists():
                continue
        except (OSError, RuntimeError):
            continue

        # Pattern 1: <root>/<folder_name>/<var>.jpg
        for var in id_variants:
            p = (root / folder_name / f"{var}.jpg").resolve()
            if p.is_file():
                return p

        # Pattern 2: <root>/keyframes/<folder_name>/<var>.jpg
        for var in id_variants:
            p = (root / "keyframes" / folder_name / f"{var}.jpg").resolve()
            if p.is_file():
                return p

        # Pattern 3: <root>/output/keyframes/<folder_name>/<var>.jpg
        for var in id_variants:
            p = (root / "output" / "keyframes" / folder_name / f"{var}.jpg").resolve()
            if p.is_file():
                return p

        # Pattern 4: <root>/<clean>
        p_clean = (root / clean).resolve()
        if p_clean.is_file():
            return p_clean

        # Pattern 5: <root>/<var>.jpg
        for var in id_variants:
            p = (root / f"{var}.jpg").resolve()
            if p.is_file():
                return p

    return None


def clear_resolve_cache():
    """Xóa toàn bộ cache resolver — gọi sau khi giải nén keyframes."""
    _resolve_cache.clear()


def resolve_keyframe_image_path(image_id_or_path: str) -> Optional[Path]:
    """
    Resolves the exact filesystem Path for a keyframe image using the exact notebook logic.
    
    Supports:
    - Raw image_id: e.g. 'L21_v030_transnetv2_0093_06' or 'L25_V001_transnetv2_0001_00'
    - Relative path: e.g. 'keyframes/L27_V001/L27_V001_transnetv2_0001_00.jpg'
    """
    if not image_id_or_path:
        return None
    return _cached_resolve_keyframe_path(str(image_id_or_path).strip())
