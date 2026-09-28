from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vectorizer.cli import main
from vectorizer.pipeline import VectorizeOptions, vectorize
from vectorizer.quantize import prepare, trim_borders


def test_trim_preserves_content_at_right_and_bottom_edges() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[-1, -1] = 255

    trimmed = trim_borders(image)

    assert trimmed.max() == 255


def test_invalid_batch_output_is_rejected() -> None:
    with pytest.raises(SystemExit) as error:
        main(["*.png", "--batch", "--output", "result.svg"])

    assert error.value.code == 2


def test_vectorize_writes_svg_to_requested_path_atomically(tmp_path) -> None:
    image_path = tmp_path / "logo.png"
    output_path = tmp_path / "nested" / "logo.svg"
    image = Image.new("RGB", (32, 32), "white")
    ImageDraw.Draw(image).rectangle((8, 8, 24, 24), fill="black")
    image.save(image_path)

    vectorize(str(image_path), str(output_path), VectorizeOptions(denoise=0))

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").startswith("<svg ")
    assert not list(output_path.parent.glob("*.tmp"))


def test_prepare_rejects_source_over_pixel_limit(tmp_path) -> None:
    image_path = tmp_path / "logo.png"
    Image.new("RGB", (4, 4), "white").save(image_path)

    with pytest.raises(ValueError, match="limite"):
        prepare(str(image_path), max_pixels=15, denoise=0)


@pytest.mark.parametrize("field", ["scale", "tolerance", "corner_angle", "min_area", "merge", "smooth"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_options_reject_non_finite_floats(field, value) -> None:
    with pytest.raises(ValueError, match="finito"):
        VectorizeOptions(**{field: value})
