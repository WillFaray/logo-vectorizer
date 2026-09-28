"""Pipeline completo: imagem -> SVG vetorizado de alta qualidade."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

from .quantize import (
    DEFAULT_MAX_PIXELS,
    parse_color,
    parse_palette,
    prepare,
    quantize,
    trim_borders,
)
from .svg import build_svg, count_path_data
from .trace import Region, trace_mask


@dataclass
class VectorizeOptions:
    colors: int = 6                       # nº de cores alvo (auto-quantização K-Means)
    palette: str | None = None            # paleta forçada "#RRGGBB,#RRGGBB,..."
    bg: str = "auto"                      # auto | none | #RRGGBB | nome CSS
    alpha_bg: str = "#FFFFFF"             # fundo para compor alpha parcial antes da quantizacao
    transparent: bool = True              # remove o fundo do SVG (alpha real)
    max_size: int = 0                     # limita o maior lado em px (0 = original)
    max_pixels: int = DEFAULT_MAX_PIXELS  # limite de pixels da imagem de origem
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

    def __post_init__(self) -> None:
        if not 1 <= self.colors <= 16:
            raise ValueError("colors deve estar entre 1 e 16")
        if self.max_size < 0:
            raise ValueError("max_size deve ser >= 0")
        if self.max_pixels <= 0:
            raise ValueError("max_pixels deve ser > 0")
        float_options = (
            ("scale", self.scale),
            ("tolerance", self.tolerance),
            ("corner_angle", self.corner_angle),
            ("min_area", self.min_area),
            ("merge", self.merge),
            ("smooth", self.smooth),
        )
        for name, value in float_options:
            if not np.isfinite(value):
                raise ValueError(f"{name} deve ser um número finito")
        if self.scale <= 0:
            raise ValueError("scale deve ser > 0")
        if self.denoise not in (0, 1, 2):
            raise ValueError("denoise deve ser 0, 1 ou 2")
        if self.tolerance < 0:
            raise ValueError("tolerance deve ser >= 0")
        if not 0 <= self.corner_angle <= 180:
            raise ValueError("corner_angle deve estar entre 0 e 180")
        if not 0 <= self.min_area <= 1:
            raise ValueError("min_area deve estar entre 0 e 1")
        if self.merge < 0:
            raise ValueError("merge deve ser >= 0")
        if self.smooth < 0:
            raise ValueError("smooth deve ser >= 0")
        if self.min_gap < 0:
            raise ValueError("min_gap deve ser >= 0")
        if self.decimals < 0:
            raise ValueError("decimals deve ser >= 0")
        parse_color(self.alpha_bg)
        if self.bg not in ("auto", "none"):
            parse_color(self.bg)
        if self.palette is not None:
            parse_palette(self.palette)


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

    alpha_bg = parse_color(o.alpha_bg)
    img = prepare(
        input_path,
        max_size=o.max_size,
        max_pixels=o.max_pixels,
        scale=o.scale,
        denoise=o.denoise,
        alpha_bg=alpha_bg,
    )
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
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(output_svg)}.",
            suffix=".tmp",
            dir=parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(svg)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, output_svg)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

    return svg, stats