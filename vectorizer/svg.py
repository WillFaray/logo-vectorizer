from __future__ import annotations

from .geometry import BezSeg, Loop
from .trace import Region


def _fmt(x: float, decimals: int = 3) -> str:
    return f"{x:.{decimals}f}"


def _loop_to_d(loop: Loop, decimals: int = 3) -> str:
    parts: list[str] = [f"M {_fmt(loop.start[0], decimals)} {_fmt(loop.start[1], decimals)}"]
    for seg in loop.segments:
        if seg.c1 is None or seg.c2 is None:
            parts.append(f"L {_fmt(seg.end[0], decimals)} {_fmt(seg.end[1], decimals)}")
        else:
            parts.append(
                f"C {_fmt(seg.c1[0], decimals)} {_fmt(seg.c1[1], decimals)} "
                f"{_fmt(seg.c2[0], decimals)} {_fmt(seg.c2[1], decimals)} "
                f"{_fmt(seg.end[0], decimals)} {_fmt(seg.end[1], decimals)}"
            )
    parts.append("Z")
    return " ".join(parts)


def build_svg(
    regions: list[Region],
    width: int,
    height: int,
    decimals: int = 3,
    background: tuple[int, int, int] | None = None,
) -> str:
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'version="1.1" shape-rendering="geometricPrecision">'
    ]
    if background is not None:
        fill = "#{:02X}{:02X}{:02X}".format(*background)
        lines.append(f'  <rect width="{width}" height="{height}" fill="{fill}"/>')

    for region in regions:
        d_parts: list[str] = []
        for path in region.paths:
            for loop in path:
                d_parts.append(_loop_to_d(loop, decimals))
        d = " ".join(d_parts)
        if d:
            fill = "#{:02X}{:02X}{:02X}".format(*region.color)
            lines.append(f'  <path fill="{fill}" fill-rule="evenodd" d="{d}"/>')

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def count_path_data(regions: list[Region]) -> int:
    total = 0
    for region in regions:
        for path in region.paths:
            for loop in path:
                total += len(loop.segments)
    return total