"""Grad-CAM overlay PNG: original image + colour-mapped CAM + "not seen by the model" marking.

The CAM is drawn on the ORIGINAL image, mapped through the exact inverse of the val transform
(:class:`~visionbench.explain.geometry.ValGeometry`). Small originals are enlarged by an integer
factor with nearest-neighbour (pixels stay pixels); large ones are reduced to fit. Pixels the
model never saw (e.g. the border removed by a centre crop) are greyed out and hatched, and the
legend says so, with the unseen fraction.
"""

from __future__ import annotations

import io
import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from visionbench.explain.geometry import ValGeometry

# Approximation of matplotlib's "inferno" (perceptually ordered, dark = 0, light = 1).
_ANCHORS = np.array(
    [
        [0.0, 0, 0, 4],
        [0.25, 87, 16, 110],
        [0.5, 188, 55, 84],
        [0.75, 249, 142, 9],
        [1.0, 252, 255, 164],
    ]
)
ALPHA = 0.5  # weight of the heatmap over the image in the seen region
MIN_DISPLAY = 256
MAX_DISPLAY = 512
UNSEEN_RGB = (128, 128, 128)
HATCH_RGB = (230, 230, 230)
HATCH_STEP = 8


def colormap(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to uint8 RGB with the inferno approximation."""
    v = np.clip(values, 0.0, 1.0)
    rgb = [np.interp(v, _ANCHORS[:, 0], _ANCHORS[:, i]) for i in (1, 2, 3)]
    return np.stack(rgb, axis=-1).round().astype(np.uint8)


def display_size(width: int, height: int) -> tuple[int, int]:
    long = max(width, height)
    if long < MIN_DISPLAY:
        f: float = math.ceil(MIN_DISPLAY / long)
    elif long > MAX_DISPLAY:
        f = MAX_DISPLAY / long
    else:
        f = 1
    return max(1, round(width * f)), max(1, round(height * f))


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow built without FreeType
        return ImageFont.load_default()


def render_overlay(
    image: Image.Image,
    cam: torch.Tensor,
    geometry: ValGeometry,
    *,
    all_zero: bool,
    caption: list[str],
) -> tuple[bytes, dict[str, int]]:
    """PNG bytes of the overlay, and the display size used."""
    w, h = image.size
    dw, dh = display_size(w, h)
    resample = Image.Resampling.NEAREST if dw >= w else Image.Resampling.LANCZOS
    base = np.asarray(image.convert("RGB").resize((dw, dh), resample), dtype=np.float64)
    values = geometry.map_to_original(cam, dw, dh)  # NaN where unseen
    seen = ~np.isnan(values)
    heat = colormap(np.nan_to_num(values, nan=0.0)).astype(np.float64)
    # An all-zero CAM highlights nothing: show the image itself, not a uniformly dark tint.
    blended = base if all_zero else (1 - ALPHA) * base + ALPHA * heat
    out = np.where(seen[..., None], blended, 0.0)
    gray = base.mean(axis=-1, keepdims=True)
    unseen_px = 0.35 * gray + 0.65 * np.array(UNSEEN_RGB, dtype=np.float64)
    out = np.where(seen[..., None], out, unseen_px)
    yy, xx = np.mgrid[0:dh, 0:dw]
    hatch = (~seen) & (((xx + yy) % HATCH_STEP) == 0)
    out[hatch] = HATCH_RGB
    canvas_img = Image.fromarray(out.round().clip(0, 255).astype(np.uint8))

    font = _font(12)
    line_h = 16
    legend_h = 8 + 14 + 6 + line_h * (2 + len(caption)) + 6
    unseen = geometry.unseen_fraction()
    unseen_text = f"not seen by the model ({unseen:.1%} of the image)"
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_w = max(measure.textlength(s, font=font) for s in [unseen_text, *caption])
    width = max(dw, 320, int(math.ceil(text_w)) + 34)  # the legend text is never cut off
    canvas = Image.new("RGB", (width, dh + legend_h), (255, 255, 255))
    canvas.paste(canvas_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    if seen.any() and not seen.all():  # outline of the region the model saw
        ys, xs = np.nonzero(seen)
        draw.rectangle(
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], outline=(255, 255, 255)
        )
    y = dh + 8
    bar_w = min(200, width - 60)
    bar = colormap(np.linspace(0.0, 1.0, bar_w))[None, :, :].repeat(14, axis=0)
    canvas.paste(Image.fromarray(bar), (30, y))
    draw.text((8, y), "0", fill=(0, 0, 0), font=font)
    draw.text((34 + bar_w, y), "1", fill=(0, 0, 0), font=font)
    y += 14 + 6
    draw.rectangle([8, y + 2, 20, y + 12], fill=UNSEEN_RGB, outline=HATCH_RGB)
    draw.text((26, y), unseen_text, fill=(0, 0, 0), font=font)
    y += line_h
    status = (
        "CAM is all zeros: no positive evidence"
        if all_zero
        else "Grad-CAM, relative (0 = none, 1 = max)"
    )
    draw.text((8, y), status, fill=(150, 0, 0) if all_zero else (0, 0, 0), font=font)
    y += line_h
    for line in caption:
        draw.text((8, y), line, fill=(0, 0, 0), font=font)
        y += line_h
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue(), {"width": dw, "height": dh}
