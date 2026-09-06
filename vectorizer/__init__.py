"""logo-vectorizer: vetorização de logos (imagens planas, com texto) em SVG."""

from .pipeline import VectorizeOptions, VectorizeStats, vectorize
from .svg import build_svg
from .trace import Region
from .geometry import Loop, BezSeg

__version__ = "1.0.0"
__all__ = ["vectorize", "VectorizeOptions", "VectorizeStats", "build_svg", "Region", "Loop", "BezSeg", "__version__"]