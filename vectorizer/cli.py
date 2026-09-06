"""CLI do vetorizador de logos."""

from __future__ import annotations

import argparse
import glob
import os
import sys

from .pipeline import VectorizeOptions, vectorize


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="logo-vectorizer",
        description=(
            "Vetoriza logos (inclusive com texto) em SVG: quantizacao de cores, "
            "remocao de fundo e curvas Bezier otimizadas."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Imagem de entrada (PNG/JPG/WebP) ou glob/diretorio com --batch")
    p.add_argument("-o", "--output", help="Arquivo SVG de saida (default: <input>_vector.svg)")
    p.add_argument(
        "--batch",
        action="store_true",
        help="Processa todas as imagens do caminho de entrada (diretorio ou glob)",
    )
    p.add_argument("--out-dir", help="Diretorio de saida no modo --batch", default="out")
    p.add_argument("--colors", type=int, default=6, help="Numero alvo de cores (K-Means)")
    p.add_argument(
        "--palette",
        help="Paleta forcada: '#FF0000,#00FF00,#0000FF' (usa as cores nessa ordem)",
    )
    p.add_argument(
        "--bg",
        default="auto",
        help="Fundo: auto | none | '#FFFFFF' | nome CSS (ex.: white, black)",
    )
    p.add_argument(
        "--no-transparent",
        action="store_true",
        help="Mantem a cor de fundo como <rect> de base (em vez de transparente)",
    )
    p.add_argument("--max-size", type=int, default=0, help="Limita o maior lado em px (0=original)")
    p.add_argument("--scale", type=float, default=1.0, help="Upscale Lanczos antes do traco (ex.: 2)")
    p.add_argument("--denoise", type=int, choices=[0, 1, 2], default=1, help="Nivel de denoise")
    p.add_argument(
        "-t", "--tolerance", type=float, default=0.6,
        help="Tolerancia do Bezier em px (menor = mais fiel, mais curvas)",
    )
    p.add_argument(
        "--corners", type=float, default=110.0,
        help="Angulo (graus) para preservar canto; menor = curvas mais suaves",
    )
    p.add_argument(
        "--min-area", type=float, default=0.0003,
        help="Fracao minima da area para manter um blob (remove sujeira)",
    )
    p.add_argument("--merge", type=float, default=28.0, help="Distancia RGB para fundir cores quase iguais")
    p.add_argument("--smooth", type=float, default=0.6, help="Sigma de suavizacao da mascara (0=off)")
    p.add_argument("--trim", action="store_true", help="Recorta margens uniformes de fundo (SVG no tamanho justo)")
    p.add_argument("--seed", type=int, default=12345, help="Semente p/ consistencia entre execucoes")
    p.add_argument("--quiet", action="store_true", help="So imprime o resultado final")
    return p


def _collect_inputs(arg_value: str, batch: bool) -> list[str]:
    if not batch:
        return [arg_value]
    if os.path.isdir(arg_value):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff")
        files: list[str] = []
        for e in exts:
            files.extend(glob.glob(os.path.join(arg_value, e)))
        return sorted(files)
    return sorted(glob.glob(arg_value))


def _out_name(in_path: str, out_dir: str | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    base = os.path.splitext(os.path.basename(in_path))[0]
    folder = out_dir or os.path.dirname(os.path.abspath(in_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return os.path.join(folder, f"{base}_vector.svg")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = _collect_inputs(args.input, args.batch)

    if not inputs:
        print(f"[erro] nenhuma imagem encontrada em: {args.input}", file=sys.stderr)
        return 2

    total_ok = 0
    for i, in_path in enumerate(inputs, 1):
        opts = VectorizeOptions(
            colors=args.colors,
            palette=args.palette,
            bg=args.bg,
            transparent=not args.no_transparent,
            max_size=args.max_size,
            scale=args.scale,
            denoise=args.denoise,
            tolerance=args.tolerance,
            corner_angle=args.corners,
            min_area=args.min_area,
            merge=args.merge,
            smooth=args.smooth,
            seed=args.seed,
            trim=args.trim,
        )
        out = _out_name(in_path, args.out_dir if args.batch else None, None if args.batch else args.output)
        try:
            _, stats = vectorize(in_path, out, opts)
            total_ok += 1
            if not args.quiet:
                notes = f"  ({'; '.join(stats.notes)})" if stats.notes else ""
                print(
                    f"[{i}/{len(inputs)}] {os.path.basename(in_path)} -> {out}\n"
                    f"  {stats.width}x{stats.height}px | {stats.colors} cores | "
                    f"{stats.paths} paths | {stats.segments} segs | "
                    f"{stats.elapsed_s:.2f}s{' | fundo removido' if stats.background_removed else ''}"
                    f"{notes}"
                )
        except Exception as exc:
            print(f"[{i}/{len(inputs)}] {os.path.basename(in_path)} -> ERRO: {exc}", file=sys.stderr)

    if not args.quiet:
        print(f"\nConcluido: {total_ok}/{len(inputs)} imagens vetorizadas.")
    return 0 if total_ok == len(inputs) else 1


if __name__ == "__main__":
    raise SystemExit(main())