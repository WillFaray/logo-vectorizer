from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vectorizer.cli import _collect_inputs, main
from vectorizer.pipeline import VectorizeOptions, vectorize
from vectorizer.quantize import prepare, quantize, trim_borders


def test_trim_preserves_content_at_right_and_bottom_edges() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[-1, -1] = 255

    trimmed = trim_borders(image)

    assert trimmed.max() == 255


def test_invalid_batch_output_is_rejected() -> None:
    with pytest.raises(SystemExit) as error:
        main(["*.png", "--batch", "--output", "result.svg"])

    assert error.value.code == 2


def test_quantize_supports_one_color() -> None:
    image = np.array(
        [
            [[20, 30, 40], [200, 210, 220]],
            [[20, 30, 40], [200, 210, 220]],
        ],
        dtype=np.uint8,
    )

    labels, colors, background = quantize(image, colors=1, bg="none", merge_dist=0)

    assert labels.shape == image.shape[:2]
    assert len(colors) == 1
    assert background is None


def test_batch_collects_only_supported_files_case_insensitively(tmp_path) -> None:
    Image.new("RGB", (4, 4), "white").save(tmp_path / "LOGO.PNG")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    (tmp_path / "nested.png").mkdir()

    assert _collect_inputs(str(tmp_path), batch=True) == [str(tmp_path / "LOGO.PNG")]


def test_batch_out_dir_failure_returns_controlled_error(tmp_path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    Image.new("RGB", (4, 4), "white").save(input_dir / "logo.png")
    output_blocker = tmp_path / "not-a-directory"
    output_blocker.write_text("blocked", encoding="utf-8")

    assert main([str(input_dir), "--batch", "--out-dir", str(output_blocker), "--quiet"]) == 2


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
