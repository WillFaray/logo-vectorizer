from __future__ import annotations

import numpy as np
import cv2
from PIL import Image, ImageOps

_MAX_SAMPLE = 1_000_000  # subsample do K-Means em imagens gigantes


def prepare(
    path: str,
    max_size: int = 0,
    scale: float = 1.0,
    denoise: int = 1,
) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGBA")
        alpha = im.getchannel("A")

    if alpha.getextrema()[0] < 255:
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")

    if scale != 1.0 or (max_size and max(im.size) > max_size):
        w, h = im.size
        if max_size and max(w, h) > max_size:
            k = max_size / max(w, h)
            w, h = int(round(w * k)), int(round(h * k))
        w, h = int(round(w * scale)), int(round(h * scale))
        im = im.resize((w, h), Image.LANCZOS)

    img = np.asarray(im).astype(np.uint8)
    if denoise >= 1:
        img = cv2.medianBlur(img, 3)
    if denoise >= 2:
        img = cv2.bilateralFilter(img, 5, 40, 40)
    return img


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Cor malformada: {value!r} (use #RRGGBB)")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def parse_palette(text: str | None) -> list[tuple[int, int, int]] | None:
    if not text:
        return None
    return [_hex_to_rgb(p) for p in text.split(",") if p.strip()]


def _kmeans_lloyd(data: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = data.shape[0]
    centers = np.empty((k, 3), dtype=np.float64)
    first = int(rng.integers(0, n))
    centers[0] = data[first]
    d2 = np.sum((data - centers[0].astype(np.float64)) ** 2, axis=1)
    for c in range(1, k):
        total = float(d2.sum())
        if total > 0 and np.all(np.isfinite(d2)):
            p = d2 / total
            pick = int(rng.choice(n, p=p))
        else:
            pick = int(rng.integers(0, n))
        centers[c] = data[pick]
        nd2 = np.sum((data - centers[c].astype(np.float64)) ** 2, axis=1)
        d2 = np.minimum(d2, nd2)

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(60):
        labels = _assign_to_centers(data, centers)
        new_centers = centers.copy()
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centers[c] = data[mask].mean(axis=0)
        if float(np.abs(new_centers - centers).max()) < 0.5:
            centers = new_centers
            break
        centers = new_centers
    labels = _assign_to_centers(data, centers)
    return centers, labels


def _kmeans_labels(data: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if data.shape[0] > _MAX_SAMPLE:
        idx = rng.choice(data.shape[0], _MAX_SAMPLE, replace=False)
        sample = data[idx]
    else:
        sample = data

    centers, _labels_s = _kmeans_lloyd(sample, k, rng)
    if data.shape[0] != sample.shape[0]:
        return _assign_to_centers(data, centers)
    return _labels_s


def _assign_to_centers(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
    return dist.argmin(axis=1)


def trim_borders(
    img: np.ndarray,
    bg_tol: int = 16,
    max_trim: float = 0.35,
) -> np.ndarray:
    h, w = img.shape[:2]
    corner = np.median(
        [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]], axis=0
    ).astype(np.int16)
    diff = np.abs(img.astype(np.int16) - corner).max(axis=2)
    is_bg = diff <= bg_tol

    col_bg = is_bg.mean(axis=0)
    row_bg = is_bg.mean(axis=1)
    any_col = bool((col_bg < 1.0).any())
    any_row = bool((row_bg < 1.0).any())

    lo_w = int(np.argmax(col_bg < 1.0)) if any_col else 0
    hi_w = int(w - 1 - np.argmax(col_bg[::-1] < 1.0)) if any_col else w
    lo_h = int(np.argmax(row_bg < 1.0)) if any_row else 0
    hi_h = int(h - 1 - np.argmax(row_bg[::-1] < 1.0)) if any_row else h

    cut_w = int(w * max_trim)
    cut_h = int(h * max_trim)
    lo_w = min(lo_w, cut_w)
    lo_h = min(lo_h, cut_h)
    hi_w = max(hi_w, w - cut_w)
    hi_h = max(hi_h, h - cut_h)
    return img[lo_h:hi_h, lo_w:hi_w]


def merge_centers(
    labels: np.ndarray, centers: np.ndarray, merge_dist: float = 28.0
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(centers, dtype=np.float64)
    k = centers.shape[0]
    members: list[list[int]] = [[i] for i in range(k)]

    changed = True
    while changed and len(members) > 1:
        changed = False
        m = len(members)
        best_pair = (-1, -1)
        best_d = merge_dist
        for i in range(m):
            ci = centers[members[i][0]]
            for j in range(i + 1, m):
                d = float(np.linalg.norm(ci - centers[members[j][0]]))
                if d < best_d:
                    best_d, best_pair = d, (i, j)
        if best_pair != (-1, -1):
            a, b = best_pair
            members[a] += members[b]
            members.pop(b)
            changed = True

    remap: dict[int, int] = {}
    final_centers: list[np.ndarray] = []
    for new_id, group in enumerate(members):
        final_centers.append(np.mean(centers[group], axis=0))
        for old in group:
            remap[old] = new_id

    new_labels = np.empty(labels.shape, dtype=np.int32)
    for old, new in remap.items():
        new_labels[labels == old] = new
    return new_labels, np.asarray(final_centers)


def detect_background(
    labels: np.ndarray,
    centers: np.ndarray,
    bg: str = "auto",
) -> int | None:
    if bg == "none":
        return None
    if labels.size == 0:
        return None

    border_labels = _border_labels(labels)
    if len(border_labels) == 0:
        return None

    if bg == "auto":
        values, counts = np.unique(border_labels, return_counts=True)
        top = values[np.argmax(counts)]
        if counts.max() / counts.sum() >= 0.5:
            return int(top)
        return None

    try:
        target = _hex_to_rgb(bg)
    except ValueError:
        from PIL import ImageColor

        target = ImageColor.getrgb(bg)
    dist = np.linalg.norm(centers - np.asarray(target, dtype=np.float64), axis=1)
    if dist.min() > 120.0:
        return None
    return int(dist.argmin())


def _border_labels(labels: np.ndarray, frac: float = 0.03) -> np.ndarray:
    h, w = labels.shape
    t = max(2, int(min(h, w) * frac))
    ring = np.zeros((h, w), dtype=bool)
    ring[:t, :] = True
    ring[-t:, :] = True
    ring[:, :t] = True
    ring[:, -t:] = True
    return labels[ring]


def quantize(
    img: np.ndarray,
    colors: int = 6,
    palette: list[tuple[int, int, int]] | None = None,
    bg: str = "auto",
    merge_dist: float = 28.0,
    seed: int = 12345,
) -> tuple[np.ndarray, list[tuple[int, int, int]], int | None]:
    h, w = img.shape[:2]
    data = img.reshape(-1, 3).astype(np.float32)

    if palette:
        centers = np.asarray(palette, dtype=np.float32)
        labels = _assign_to_centers(data, centers)
    else:
        k = max(2, min(colors, 16))
        labels = _kmeans_labels(data, k, seed)
        centroids = np.zeros((k, 3), dtype=np.float64)
        counts = np.zeros(k)
        np.add.at(centroids, labels, data.astype(np.float64))
        np.add.at(counts, labels, 1)
        counts[counts == 0] = 1.0
        centers = centroids / counts[:, None]

    labels, centers = merge_centers(labels, centers, merge_dist)

    labels = labels.reshape(h, w)
    bg_label = detect_background(labels, centers, bg)

    flat = labels.ravel()
    counts = np.bincount(flat, minlength=len(centers)).astype(np.float64)
    order = np.argsort(-counts)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[int(l)] for l in flat], dtype=np.int32)
    centers = centers[order]
    if bg_label is not None:
        bg_label = int(remap[bg_label])

    color_list = [tuple(int(round(c)) for c in centers[i]) for i in range(len(centers))]
    return labels.reshape(h, w), color_list, bg_label