"""Gera logos sintéticos (com texto) para validar o vetorizador.

Uso: python tests/make_sample.py [--outdir out_dir]
"""

from __future__ import annotations

import argparse
import math
import os

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = r"C:\Windows\Fonts"


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = os.path.join(FONT_DIR, name)
    return ImageFont.truetype(path, size)


def logo_roundel_text(path: str, size: int = 1200) -> None:
    """Roundel com círculo, estrela e texto (curvas, cantos e furos)."""
    w = h = size
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    d.ellipse([w * 0.10, h * 0.10, w * 0.90, h * 0.90], fill="#E9EEF5")
    d.ellipse([w * 0.16, h * 0.16, w * 0.84, h * 0.84], fill="#1F4E9C")

    cx, cy, R = w * 0.5, h * 0.5, w * 0.24
    pts = []
    for k in range(10):
        ang = -math.pi / 2 + k * math.pi / 5
        r = R if k % 2 == 0 else R * 0.42
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    d.polygon(pts, fill="#F5B301")

    f_main = _font("arialbd.ttf", int(size * 0.11))
    d.text((w * 0.5, h * 0.5 + size * 0.21), "VECTOR LAB", font=f_main,
           fill="#FFFFFF", anchor="mm")

    f_sub = _font("ariali.ttf", int(size * 0.045))
    d.text((w * 0.5, h * 0.5 + size * 0.28), "desde 2026", font=f_sub,
           fill="#CFE0F5", anchor="mm")

    img.save(path)
    print(f"criado {path}")


def logo_serif_text(path: str, size: tuple[int, int] = (1400, 700)) -> None:
    """Letreiro serifado + faixas diagonais (texto fino e cantos)."""
    w, h = size
    img = Image.new("RGB", (w, h), "#FFFFFF")
    d = ImageDraw.Draw(img)

    d.polygon([(0, 0), (w * 0.34, 0), (0, h * 0.4)], fill="#0B3D2E")
    d.polygon([(w, h), (w * 0.66, h), (w, h * 0.6)], fill="#B23A48")

    f_main = _font("georgia.ttf", int(h * 0.20)) if os.path.exists(os.path.join(FONT_DIR, "georgia.ttf")) \
        else _font("times.ttf", int(h * 0.16))
    d.text((w * 0.5, h * 0.42), "Athena & Co.", font=f_main, fill="#101820", anchor="mm")

    d.line([(w * 0.30, h * 0.62), (w * 0.70, h * 0.62)], fill="#B23A48", width=int(h * 0.012))

    f_sub = _font("arial.ttf", int(h * 0.055))
    d.text((w * 0.5, h * 0.78), "EST 1998 • LUXURY HOUSE", font=f_sub, fill="#4A5560", anchor="mm")

    img.save(path)
    print(f"criado {path}")


def logo_lowres_jpeg(path: str, source: str, factor: float = 0.4, quality: int = 55) -> None:
    """Versão reduzida + JPEG com artefatos (testa denoise/upscale)."""
    with Image.open(source) as im:
        w, h = int(im.width * factor), int(im.height * factor)
        im = im.resize((w, h), Image.LANCZOS)
        im.save(path, "JPEG", quality=quality)
    print(f"criado {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "samples"))
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    outdir = args.outdir

    a = os.path.join(outdir, "logo_roundel_text.png")
    b = os.path.join(outdir, "logo_serif_text.png")
    logo_roundel_text(a)
    logo_serif_text(b)
    logo_lowres_jpeg(os.path.join(outdir, "logo_lowres.jpg"), a)


if __name__ == "__main__":
    main()