from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class BezSeg:
    c1: np.ndarray | None
    c2: np.ndarray | None
    end: np.ndarray


@dataclass
class Loop:

    start: np.ndarray
    segments: list


def contour_to_chain(contour: np.ndarray) -> np.ndarray:
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) == 0:
        return pts
    keep = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - keep[-1]) > 0.49:
            keep.append(p)
    if len(keep) > 1 and np.linalg.norm(keep[0] - keep[-1]) <= 0.49:
        keep.pop()
    return np.asarray(keep)


def circular_slice(chain: np.ndarray, a: int, b: int) -> np.ndarray:
    n = len(chain)
    return chain[np.arange(a, b + 1) % n]


def _local_angle_at(pts: np.ndarray, i: int, r: int) -> float:
    n = len(pts)
    a, b, c = pts[(i - r) % n], pts[i], pts[(i + r) % n]
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cos = float(np.clip(float(v1 @ v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def detect_corners(
    pts: np.ndarray,
    angle_deg: float = 110.0,
    min_gap: int = 4,
) -> list[int]:
    n = len(pts)
    if n < 9:
        return []
    r = int(np.clip(round(n * 0.006), 3, 12))

    ang = np.empty(n, dtype=np.float64)
    for i in range(n):
        ang[i] = _local_angle_at(pts, i, r)

    is_min = np.zeros(n, dtype=bool)
    for i in range(n):
        vals = [ang[(i + d) % n] for d in range(-r, r + 1)]
        is_min[i] = bool(ang[i] <= min(vals) + 1e-9)

    candidates = [i for i in range(n) if is_min[i] and ang[i] <= angle_deg]
    candidates.sort(key=lambda i: ang[i])

    picked: list[int] = []
    for i in candidates:
        if all(min((i - j) % n, (j - i) % n) >= min_gap for j in picked):
            picked.append(i)
    picked.sort()
    return picked


def _antipodal_pair(n: int) -> list[int]:
    i = 0
    best, bj = -1, 0
    for j in range(n):
        d = min((j - i) % n, (i - j) % n)
        if d > best:
            best, bj = d, j
    return [i, bj]
def _robust_tangent(pts: np.ndarray, at_start: bool = True) -> np.ndarray | None:
    ds: list[np.ndarray] = []
    lim = min(4, len(pts))
    if at_start:
        for k in range(1, lim):
            ds.append(pts[k].astype(np.float64) - pts[0])
    else:
        for k in range(1, lim):
            ds.append(pts[-1].astype(np.float64) - pts[-1 - k])
    if not ds:
        return None
    v = np.sum(ds, axis=0)
    nm = float(np.linalg.norm(v))
    return v / nm if nm > 1e-9 else None


def _straightness(pts: np.ndarray) -> float:
    p0, p3 = pts[0], pts[-1]
    chord = float(np.linalg.norm(p3 - p0))
    if chord < 1e-6:
        return 0.0
    d = pts[1:-1] - p0
    if d.size == 0:
        return 0.0
    cross = d[:, 0] * (p3[1] - p0[1]) - d[:, 1] * (p3[0] - p0[0])
    return float(np.abs(cross).max()) / chord


def fit_bezier(
    pts: np.ndarray,
    tol: float = 0.6,
    depth: int = 0,
    max_depth: int = 22,
) -> list[BezSeg]:
    n = len(pts)
    p0, p3 = pts[0], pts[-1]
    chord = float(np.linalg.norm(p3 - p0))
    if n < 4 or chord < 1e-6:
        return [BezSeg(None, None, p3)]

    if _straightness(pts) <= tol:
        return [BezSeg(None, None, p3)]

    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    t = s / s[-1]

    t0 = _robust_tangent(pts, True)
    t1 = _robust_tangent(pts, False)

    chord_dir = p3 - p0
    chord_len = float(np.linalg.norm(chord_dir))
    fallback = chord_dir / chord_len if chord_len > 1e-9 else np.array([1.0, 0.0])
    if t0 is None:
        t0 = fallback
    if t1 is None:
        t1 = fallback

    ti = t[1:-1]
    qi = pts[1:-1]
    if ti.size == 0:
        alpha = chord * 0.5
        beta = chord * 0.5
    else:
        a0 = 3.0 * (1.0 - ti) ** 2 * ti
        a1 = 3.0 * (1.0 - ti) * ti**2
        coef0 = (1.0 - ti) ** 2 * (1.0 + 2.0 * ti)
        coef3 = ti**2 * (3.0 - 2.0 * ti)
        base = coef0[:, None] * p0 + coef3[:, None] * p3
        r = qi - base

        b00 = float((a0 * a0).sum())
        b11 = float((a1 * a1).sum())
        dot = float(np.dot(t0, t1)) if (t0 is not None and t1 is not None) else 0.0
        b01 = float((a0 * a1).sum()) * dot
        v0 = float(np.sum(a0 * (r @ t0), axis=0)) if t0 is not None else 0.0
        v1 = float(np.sum(a1 * (r @ t1), axis=0)) if t1 is not None else 0.0

        det = b00 * b11 - b01 * b01
        if abs(det) < 1e-12 or t0 is None or t1 is None:
            alpha = chord * 0.35
            beta = chord * 0.35
        else:
            alpha = (v0 * b11 - b01 * v1) / det
            beta = (b00 * v1 - b01 * v0) / det
            alpha = float(np.clip(alpha, -3.0 * chord, 3.0 * chord))
            beta = float(np.clip(beta, -3.0 * chord, 3.0 * chord))

    c1 = p0 + alpha * t0
    c2 = p3 + beta * t1

    if ti.size:
        b = (
            ((1.0 - ti) ** 3)[:, None] * p0
            + (3.0 * (1.0 - ti) ** 2 * ti)[:, None] * c1
            + (3.0 * (1.0 - ti) * ti**2)[:, None] * c2
            + (ti**3)[:, None] * p3
        )
        dev = np.linalg.norm(b - qi, axis=1)
        imax = int(dev.argmax())
        max_dev = float(dev[imax])
    else:
        max_dev, imax = 0.0, -1

    if max_dev > tol and depth < max_depth and len(pts) >= 6:
        k = imax + 1
        left = fit_bezier(pts[: k + 1], tol, depth + 1, max_depth)
        right = fit_bezier(pts[k:], tol, depth + 1, max_depth)
        return left + right

    return [BezSeg(c1, c2, p3)]


def chain_to_loop(
    chain: np.ndarray,
    tol: float = 0.6,
    corner_angle: float = 110.0,
    min_gap: int = 4,
) -> Loop | None:
    n = len(chain)
    if n < 4:
        return None
    corners = detect_corners(chain, corner_angle, min_gap)
    if len(corners) < 2:
        corners = _antipodal_pair(n)

    starts = sorted(corners)
    segments: list[BezSeg] = []
    for i, a in enumerate(starts):
        b = starts[i + 1] if i + 1 < len(starts) else starts[0] + n
        seg_pts = circular_slice(chain, a, b)
        segments.extend(fit_bezier(seg_pts, tol))

    if not segments:
        return None
    return Loop(start=chain[starts[0]].copy(), segments=segments)