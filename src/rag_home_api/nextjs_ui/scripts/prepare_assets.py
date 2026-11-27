"""asset/ 폴더의 흰색 배경 3D 캐릭터 이미지를 투명 PNG(public/characters/)로 일괄 변환한다.

실행:
    .venv/Scripts/python.exe src/rag_home_api/nextjs_ui/scripts/prepare_assets.py
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

BASE_DIR = Path(__file__).resolve().parent.parent
ASSET_DIR = BASE_DIR / "asset"
OUT_CHAR_DIR = BASE_DIR / "public" / "characters"

CHAR_MAP: dict[str, str] = {
    "코난.jpg": "conan_front.png",
    "생각코난.jpg": "conan_thinking.png",
    "3D_konan.jpg": "conan_happy.png",
    "conan_back.png": "conan_back.png",
    "장미.jpg": "haibara.png",
    "미란이.jpg": "ran.png",
    "박사님.jpg": "agasa.png",
    "아름이.jpg": "ayumi.png",
}


def remove_white_bg(src_path: Path, dst_path: Path, threshold: int = 238, soft_band: int = 22) -> None:
    """외곽 연결 flood-fill 기반 흰색 배경 제거 + 경계 안티앨리어싱 + 여백 크롭."""
    img = Image.open(src_path).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    h, w, _ = arr.shape

    rgb = arr[:, :, :3].astype(np.int16)
    # 밝고 채도가 낮은 흰색/연회색 배경 후보 마스크
    min_c = rgb.min(axis=2)
    max_c = rgb.max(axis=2)
    sat = max_c - min_c
    is_bg_candidate = (min_c >= threshold) & (sat <= 24)

    # 이미지 외곽 테두리에서 시작하는 flood-fill만 투명화 (캐릭터 내부 흰 셔츠/눈동자 보호!)
    visited = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(w):
        if is_bg_candidate[0, x] and not visited[0, x]:
            visited[0, x] = True
            queue.append((0, x))
        if is_bg_candidate[h - 1, x] and not visited[h - 1, x]:
            visited[h - 1, x] = True
            queue.append((h - 1, x))
    for y in range(h):
        if is_bg_candidate[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            queue.append((y, 0))
        if is_bg_candidate[y, w - 1] and not visited[y, w - 1]:
            visited[y, w - 1] = True
            queue.append((y, w - 1))

    while queue:
        cy, cx = queue.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and is_bg_candidate[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    alpha = np.where(visited, 0, 255).astype(np.uint8)

    # 외곽 경계 1~2px 부드러운 알파 페더링
    bg_img = Image.fromarray(alpha, mode="L")
    blurred = bg_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    blur_arr = np.array(blurred, dtype=np.float32)

    # 배경과 인접한 경계 픽셀의 흰색 halo 완화
    edge_zone = (blur_arr > 0) & (blur_arr < 250) & (~visited)
    luminance = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
    soft_alpha = np.clip((threshold + soft_band - luminance) / max(soft_band, 1) * 255.0, 0, 255)
    final_alpha = alpha.astype(np.float32)
    final_alpha[edge_zone] = np.minimum(blur_arr[edge_zone], soft_alpha[edge_zone])
    arr[:, :, 3] = np.clip(final_alpha, 0, 255).astype(np.uint8)

    out = Image.fromarray(arr, mode="RGBA")
    bbox = out.getbbox()
    if bbox:
        pad = 12
        x0 = max(0, bbox[0] - pad)
        y0 = max(0, bbox[1] - pad)
        x1 = min(w, bbox[2] + pad)
        y1 = min(h, bbox[3] + pad)
        out = out.crop((x0, y0, x1, y1))

    # 웹 최적화 크기 (최대 640px)
    out.thumbnail((640, 640), Image.Resampling.LANCZOS)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst_path, format="PNG", optimize=True)
    print(f"OK: {src_path.name} -> {dst_path.name} ({out.size[0]}x{out.size[1]})")


def main() -> None:
    OUT_CHAR_DIR.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in CHAR_MAP.items():
        src = ASSET_DIR / src_name
        if not src.exists():
            print(f"SKIP (missing): {src}")
            continue
        remove_white_bg(src, OUT_CHAR_DIR / dst_name)


if __name__ == "__main__":
    main()
