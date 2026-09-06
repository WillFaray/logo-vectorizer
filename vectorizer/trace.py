from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import Loop, chain_to_loop, contour_to_chain


@dataclass
class Region:

    color: tuple[int, int, int]
    paths: list[list[Loop]] = field(default_factory=list)


def trace_mask(
    mask: np.ndarray,
    tol: float = 0.6,
    corner_angle: float = 110.0,
    min_area_px: float = 6.0,
    smooth_sigma: float = 0.6,
    min_gap: int = 4,
) -> list[list[Loop]]:
    work = mask
    if smooth_sigma and smooth_sigma > 0.0:
        blurred = cv2.GaussianBlur(mask, (0, 0), smooth_sigma)
        work = np.where(blurred > 127, np.uint8(255), np.uint8(0)).astype(np.uint8)

    contours, hierarchy = cv2.findContours(work, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0:
        return []

    hierarchy = hierarchy[0]
    n = len(contours)

    top: list[int] = []
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        parent = int(hierarchy[i][3])
        if parent < 0:
            top.append(i)
        else:
            children[parent].append(i)

    paths: list[list[Loop]] = []
    for i in top:
        loops: list[Loop] = []
        for cid in [i] + children[i]:
            area = cv2.contourArea(contours[cid])
            if area < max(1.0, min_area_px):
                continue
            chain = contour_to_chain(contours[cid])
            loop = chain_to_loop(chain, tol=tol, corner_angle=corner_angle, min_gap=min_gap)
            if loop is not None:
                loops.append(loop)
        if loops:
            paths.append(loops)
    return paths