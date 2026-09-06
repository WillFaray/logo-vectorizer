"""Pipeline completo: imagem -> SVG vetorizado de alta qualidade."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from .quantize import parse_palette, prepare, quantize, trim_borders
from .svg import build_svg, count_path_data
from .trace import Region, trace_mask


@dataclass
class VectorizeOptions:
    colors: int = 6                       # nº de cores alvo (auto-quantização K-Means)
    palette: str | None = None            # paleta forçada "#RRGGBB,#RRGGBB,..."
    bg: str = "auto"                      # auto | none | #RRGGBB | nome CSS
    transparent: bool = True              # remove o fundo do SVG (alpha real)
    max_size: int = 0                     # limita o maior lado em px (0 = original)
    scale: float = 1.0                    # upscale Lanczos antes do traço
    denoise: int = 1                      # 0 | 1 | 2
    tolerance: float = 0.6                # erro máx. (px) do Bézier vs imagem
    corner_angle: float = 110.0           # graus <= este => canto preservado
    min_area: float = 0.0003              # fração mínima da área p/ manter um blob
    merge: float = 28.0                   # distância RGB p/ fundir cores quase iguais
    smooth: float = 0.6                   # sigma da suavização da máscara
    seed: int = 12345                     # semente para determinismo/consistência
    min_gap: int = 4                      # separação mínima entre cantos (px)
    trim: bool = False                    # recorta margens uniformes de fundo
    decimals: int = 3                     # casas decimais no SVG


@dataclass
class VectorizeStats:
    width: int = 0
    height: int = 0
    colors: int = 0
    regions: int = 0
    paths: int = 0
    segments: int = 0
    background_removed: bool = False
    bg_color: tuple[int, int, int] | None = None
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)


def vectorize(
    input_path: str,
    output_svg: str | None = None,
    opts: VectorizeOptions | None = None,
) -> tuple[str, VectorizeStats]:
    t0 = time.perf_counter()
    o = opts or VectorizeOptions()

    img = prepare(input_path, max_size=o.max_size, scale=o.scale, denoise=o.denoise)
    if o.trim:
        img = trim_borders(img)
    h, w = img.shape[:2]

    palette = parse_palette(o.palette)
    labels, colors, bg_label = quantize(
        img,
        colors=o.colors,
        palette=palette,
        bg=o.bg,
        merge_dist=o.merge,
        seed=o.seed,
    )

    regions: list[Region] = []
    background: tuple[int, int, int] | None = None
    if bg_label is not None and not o.transparent:
        background = colors[bg_label]

    total_px = h * w
    for k in range(len(colors)):
        if bg_label is not None and o.transparent and k == bg_label:
            continue
        mask = np.where(labels == k, np.uint8(255), np.uint8(0)).astype(np.uint8)
        min_area_px = max(8.0, total_px * o.min_area)
        paths = trace_mask(
            mask,
            tol=o.tolerance,
            corner_angle=o.corner_angle,
            min_area_px=min_area_px,
            smooth_sigma=o.smooth,
            min_gap=o.min_gap,
        )
        if paths:
            regions.append(Region(color=colors[k], paths=paths))

    svg = build_svg(regions, w, h, decimals=o.decimals, background=background)

    stats = VectorizeStats(
        width=w,
        height=h,
        colors=len(regions),
        regions=len(regions),
        paths=sum(len(r.paths) for r in regions),
        segments=count_path_data(regions),
        background_removed=bool(o.transparent and bg_label is not None),
        bg_color=colors[bg_label] if bg_label is not None else None,
        elapsed_s=time.perf_counter() - t0,
    )
    if o.transparent and bg_label is None:
        stats.notes.append(
            "sem cor de fundo detectada na borda — use --bg <cor|hex> para fundo transparente"
        )

    if output_svg:
        parent = os.path.dirname(os.path.abspath(output_svg))
        os.makedirs(parent, exist_ok=True)
        with open(output_svg, "w", encoding="utf-8") as f:
            f.write(svg)

    return svg, stats