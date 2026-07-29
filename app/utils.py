import json
import struct

import cv2
import numpy as np


def _jpeg_exif_orientation(data: bytes) -> int:
    """Return EXIF orientation (1-8) for JPEG bytes, else 1."""
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return 1
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker == 0xD9 or marker == 0xDA:
            break
        if i + 4 > len(data):
            break
        seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        if seg_len < 2 or i + 2 + seg_len > len(data):
            break
        if marker in (0xE1,) and seg_len >= 8:
            payload = data[i + 4 : i + 2 + seg_len]
            if payload.startswith(b"Exif\x00\x00"):
                orient = _parse_exif_orientation(payload[6:])
                if orient:
                    return orient
        i += 2 + seg_len
    return 1


def _parse_exif_orientation(tiff: bytes) -> int | None:
    if len(tiff) < 8:
        return None
    endian = "<" if tiff[0:2] == b"II" else ">" if tiff[0:2] == b"MM" else None
    if endian is None:
        return None
    try:
        (offset,) = struct.unpack(endian + "I", tiff[4:8])
        if offset + 2 > len(tiff):
            return None
        (count,) = struct.unpack(endian + "H", tiff[offset : offset + 2])
        for n in range(count):
            entry = offset + 2 + n * 12
            if entry + 12 > len(tiff):
                break
            tag, typ, _cnt, val = struct.unpack(endian + "HHII", tiff[entry : entry + 12])
            if tag == 0x0112:  # Orientation
                if typ == 3:  # SHORT
                    return int(val & 0xFFFF) if endian == "<" else int(val >> 16)
                return int(val)
    except struct.error:
        return None
    return None


def _apply_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 2:
        return cv2.flip(image, 1)
    if orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(image, 0)
    if orientation == 5:
        return cv2.rotate(cv2.flip(image, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
    if orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.rotate(cv2.flip(image, 1), cv2.ROTATE_90_CLOCKWISE)
    if orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def read_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data")
    return _apply_orientation(image, _jpeg_exif_orientation(data))


def crop_around_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    *,
    pad_ratio: float = 0.65,
    min_side: int = 320,
) -> np.ndarray:
    """Crop a padded square around a face bbox so off-center faces are recentered."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(bw, bh) * (1.0 + pad_ratio)
    side = max(side, float(min_side))

    nx1 = int(round(cx - side / 2.0))
    ny1 = int(round(cy - side / 2.0))
    nx2 = int(round(cx + side / 2.0))
    ny2 = int(round(cy + side / 2.0))

    # Shift crop into frame instead of shrinking when face is near an edge.
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > w:
        shift = nx2 - w
        nx1 = max(0, nx1 - shift)
        nx2 = w
    if ny2 > h:
        shift = ny2 - h
        ny1 = max(0, ny1 - shift)
        ny2 = h

    crop = image[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return image

    ch, cw = crop.shape[:2]
    shortest = min(ch, cw)
    if 0 < shortest < min_side:
        scale = min_side / float(shortest)
        crop = cv2.resize(
            crop,
            (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
    return crop


def enhance_for_insightface(image: np.ndarray) -> np.ndarray:
    """
    OpenCV preprocessing for higher-quality InsightFace embeddings:
    - bilateral denoise (keeps edges)
    - CLAHE contrast on luminance
    - mild unsharp mask
    """
    if image is None or image.size == 0:
        return image

    frame = image
    h, w = frame.shape[:2]
    # Upscale small mobile crops so the detector sees more detail.
    shortest = min(h, w)
    if 0 < shortest < 480:
        scale = 480.0 / float(shortest)
        frame = cv2.resize(
            frame,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )

    denoised = cv2.bilateralFilter(frame, d=5, sigmaColor=45, sigmaSpace=45)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    merged = cv2.merge([l_ch, a_ch, b_ch])
    contrasted = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(contrasted, (0, 0), 1.0)
    sharp = cv2.addWeighted(contrasted, 1.25, blur, -0.25, 0)
    return sharp


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = normalize_embedding(np.asarray(a, dtype=np.float32))
    b_norm = normalize_embedding(np.asarray(b, dtype=np.float32))
    return float(np.dot(a_norm, b_norm))


def parse_embedding_json(raw: str) -> list[float]:
    data = json.loads(raw)
    if isinstance(data, dict) and "embedding" in data:
        data = data["embedding"]
    if not isinstance(data, list):
        raise ValueError("stored_embedding must be a JSON array of floats")
    return [float(v) for v in data]
