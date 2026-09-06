"""Valida a fidelidade: re-renderiza o SVG em bitmap e compara com a imagem quantizada.

Uso: python tests/validate.py <imagem> [--out-dir out]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorizer.pipeline import VectorizeOptions, vectorize
from vectorizer.geometry import Loop, BezSeg
from vectorizer.trace import Region

SAMPLE_PER_CUBIC = 24


def _sample_loop(loop: Loop, n: int) -> np.ndarray:
    """Amostra um Loop fechado em pontos densos para fillPoly."""
    pts: list[tuple[float, float]] = [(float(loop.start[0]), float(loop.start[1]))]
    cur = loop.start.copy()
    for seg in loop.segments:
        if seg.c1 is None or seg.c2 is None:
            pts.append((float(seg.end[0]), float(seg.end[1])))
            cur = seg.end
        else:
            p0, p1, p2, p3 = cur, seg.c1, seg.c2, seg.end
            for i in range(1, n):
                t = i / n
                mt = 1.0 - t
                x = (mt**3) * p0[0] + 3 * (mt**2) * t * p1[0] + 3 * mt * (t**2) * p2[0] + (t**3) * p3[0]
                y = (mt**3) * p0[1] + 3 * (mt**2) * t * p1[1] + 3 * mt * (t**2) * p2[1] + (t**3) * p3[1]
                pts.append((x, y))
            cur = seg.end
            pts.append((float(seg.end[0]), float(seg.end[1])))
    return np.array(pts, dtype=np.int32)


def render_regions(regions, w: int, h: int) -> np.ndarray:
    """Re-renderiza as regiões vetoriais em um bitmap RGB."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for region in regions:
        polys = []
        for path in region.paths:
            for loop in path:
                pts = _sample_loop(loop, SAMPLE_PER_CUBIC)
                if len(pts) >= 3:
                    polys.append(pts)
        if polys:
            cv2.fillPoly(canvas, polys, region.color)
    return canvas


def _parse_path_bezier(d: str, color):
    """Parse completo do 'd' (M/L/C/Z) gerando Region com loops de Bézier reais."""
    tokens = re.findall(r"[MLCZmlcz]|-?\d+(?:\.\d+)?", d)
    i = 0
    commands: list[tuple[str, list[float]]] = []
    while i < len(tokens):
        cmd = tokens[i].upper()
        if cmd in "MLCZ":
            nums: list[float] = []
            i += 1
            while i < len(tokens) and re.match(r"^-?\d", tokens[i]):
                nums.append(float(tokens[i]))
                i += 1
            commands.append((cmd, nums))
        else:
            i += 1

    loops: list[Loop] = []
    start: np.ndarray | None = None
    cur: np.ndarray | None = None
    segs: list[BezSeg] = []

    def flush() -> None:
        nonlocal segs, start
        if start is not None and segs:
            loops.append(Loop(start=start.copy(), segments=segs))
        segs, start = [], None

    for cmd, nums in commands:
        if cmd == "M" and len(nums) >= 2:
            flush()
            start = np.array([nums[0], nums[1]])
            cur = start
        elif cmd in ("L", "Z") and len(nums) >= 2 and cur is not None:
            if cmd == "L":
                end = np.array([nums[0], nums[1]])
            else:
                end = start
            segs.append(BezSeg(None, None, end))
            cur = end
        elif cmd == "C" and len(nums) >= 6 and cur is not None:
            c1 = np.array([nums[0], nums[1]])
            c2 = np.array([nums[2], nums[3]])
            end = np.array([nums[4], nums[5]])
            segs.append(BezSeg(c1, c2, end))
            cur = end
    flush()
    return Region(color=color, paths=[loops])


def regions_from_svg(path: str) -> list[Region]:
    """Converte o SVG gerado de volta em lista de Region (parse lendo o 'd')."""
    regions: list[Region] = []
    body = open(path, encoding="utf-8").read()
    for m in re.finditer(r'<path fill="#([0-9A-Fa-f]{6})"[^>]*d="([^"]+)"', body):
        hexc = m.group(1)
        color = (int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16))
        regions.append(_parse_path_bezier(m.group(2), color))
    return regions


def quantized_reference(img: np.ndarray, colors: int = 6) -> np.ndarray:
    """Replica a quantização (mesma seed); fundo vira (0,0,0) como no SVG transparente."""
    from vectorizer.quantize import quantize

    labels, colors_list, bg = quantize(img, colors=colors, palette=None, bg="auto")
    ref = np.zeros_like(img)
    for k, color in enumerate(colors_list):
        if k == bg:
            continue
        ref[labels == k] = color
    return ref


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--out-dir", default="out")
    p.add_argument("--tolerance", type=float, default=0.6)
    p.add_argument("--corners", type=float, default=110.0)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--colors", type=int, default=6)
    p.add_argument("--denoise", type=int, default=1)
    p.add_argument("--min-area", type=float, default=0.0003)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.image))[0]
    svg_out = os.path.join(args.out_dir, f"{base}_vector.svg")

    opts = VectorizeOptions(
        tolerance=args.tolerance,
        corner_angle=args.corners,
        colors=args.colors,
        bg="auto",
        transparent=True,
        scale=args.scale,
        denoise=args.denoise,
        min_area=args.min_area,
    )
    _svg_text, stats = vectorize(args.image, svg_out, opts)

    from vectorizer.quantize import prepare

    img = prepare(args.image, max_size=stats.width, scale=args.scale, denoise=args.denoise)
    h, w = img.shape[:2]
    reference = quantized_reference(img, colors=args.colors)
    rendered = render_regions(regions_from_svg(svg_out), w, h)

    exact = float(np.mean(np.all(rendered == reference, axis=2)))
    close = float(np.mean(np.max(np.abs(rendered.astype(int) - reference.astype(int)), axis=2) <= 24))
    mean_err = float(np.mean(np.abs(rendered.astype(int) - reference.astype(int))))

    print(f"arquivo   : {args.image}")
    print(f"dimensoes : {stats.width}x{stats.height}")
    print(f"cores     : {stats.colors}")
    print(f"paths     : {stats.paths}  segmentos: {stats.segments}")
    print(f"exact_match : {exact*100:.2f}%")
    print(f"close_match : {close*100:.2f}%  (<=24/255)")
    print(f"mean_abs_err: {mean_err:.2f} / 255")
    print(f"tempo     : {stats.elapsed_s:.2f}s")
    print(f"svg       : {svg_out}")

    combo = np.hstack([img, reference, rendered])
    combo = cv2.resize(combo, (combo.shape[1] // 2, combo.shape[0] // 2), interpolation=cv2.INTER_AREA)
    preview = os.path.join(args.out_dir, f"{base}_preview.png")
    cv2.imwrite(preview, combo)
    print(f"preview   : {preview}")

    ok = exact > 0.985
    print("RESULTADO:", "[APROVADO]" if ok else "[REPROVADO] (esperado >98.5%)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()