"""Make the Aghosh memento's bright-on-dark panels suitable for extrusion.

The rest of the artwork uses dark ink on white.  The binary-state and BPF
panels use bright coloured features on dark backgrounds, so a global grayscale
threshold turns each panel into a solid block.  This script locally extracts
those bright features and writes them as dark linework on white.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("color_png", type=Path, help="Colour raster of the SVG")
    parser.add_argument("base_grayscale_png", type=Path, help="Existing grayscale master")
    parser.add_argument("output_png", type=Path, help="Corrected grayscale master")
    parser.add_argument("--size-mm", type=float, default=70.0)
    parser.add_argument(
        "--thin-px",
        type=int,
        default=0,
        help="Thin heavy linework by this many pixels per edge (default: 0)",
    )
    parser.add_argument(
        "--thin-full-image",
        action="store_true",
        help="Apply label-preserving heavy-stroke thinning to the complete image",
    )
    return parser.parse_args()


def bright_features_on_white(
    source: Image.Image, *, include_broad_bright_areas: bool
) -> Image.Image:
    rgb = np.asarray(source.convert("RGB"), dtype=np.float32)
    # Max(R,G,B) treats saturated blue, green, and red equally. Standard
    # luminance would make the blue rings much weaker than the red rings.
    value = rgb.max(axis=2)
    value_image = Image.fromarray(np.uint8(np.clip(value, 0, 255)), mode="L")
    blur_radius = max(2.0, min(source.size) / 28.0)
    blurred = np.asarray(
        value_image.filter(ImageFilter.GaussianBlur(radius=blur_radius)),
        dtype=np.float32,
    )
    local_signal = np.maximum(value - blurred, 0.0) * 4.0
    baseline = float(np.percentile(value, 35))
    broad_signal = np.maximum(value - (baseline + 24.0), 0.0) * 1.7
    signal = (
        np.maximum(local_signal, broad_signal)
        if include_broad_bright_areas
        else local_signal
    )
    signal[signal < 22.0] = 0.0
    return Image.fromarray(np.uint8(255.0 - np.clip(signal, 0.0, 255.0)), mode="L")


def thin_heavy_linework(source: Image.Image, pixels: int) -> Image.Image:
    """Thin heavy strokes while preserving fine labels and annotations."""
    if pixels <= 0:
        return source

    thinned = source.filter(ImageFilter.MaxFilter(2 * pixels + 1))
    # A 7 x 7 erosion leaves a core only in genuinely heavy local strokes.
    # Expanding that core covers corners and endpoints without propagating the
    # selection along fine graph lines connected to a thick shaded feature.
    source_array = np.asarray(source, dtype=np.uint8)
    core = source.filter(ImageFilter.MaxFilter(7))
    core_array = np.asarray(core, dtype=np.uint8) < 160
    core_mask = Image.fromarray(
        np.where(core_array, 0, 255).astype(np.uint8), mode="L"
    )
    heavy_mask = np.asarray(core_mask.filter(ImageFilter.MinFilter(9))) < 128
    thinned_array = np.asarray(thinned, dtype=np.uint8)
    return Image.fromarray(
        np.where(heavy_mask, thinned_array, source_array).astype(np.uint8),
        mode="L",
    )


def main() -> None:
    args = parse_args()
    if args.thin_px < 0:
        raise SystemExit("--thin-px must be zero or greater")
    source_image = Image.open(args.color_png)
    if "A" in source_image.getbands():
        rgba = source_image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white.alpha_composite(rgba)
        color = white.convert("RGB")
    else:
        color = source_image.convert("RGB")
    base = Image.open(args.base_grayscale_png).convert("L")
    if color.size != base.size:
        if color.width >= base.width and color.height >= base.height and max(
            color.width - base.width, color.height - base.height
        ) <= 4:
            color = color.crop((0, 0, base.width, base.height))
        else:
            raise SystemExit(f"Input sizes differ: colour={color.size}, grayscale={base.size}")
    px_per_mm = color.width / args.size_mm

    def box_mm(x: float, y: float, width: float, height: float) -> tuple[int, int, int, int]:
        return tuple(round(v * px_per_mm) for v in (x, y, x + width, y + height))

    def replace_dark_panel(
        search_box: tuple[int, int, int, int], *, include_broad: bool
    ) -> None:
        source = color.crop(search_box)
        value = np.asarray(source, dtype=np.uint8).max(axis=2)
        dark = value < 200
        rows = np.flatnonzero(dark.mean(axis=1) > 0.50)
        if not len(rows):
            raise RuntimeError(f"No dark panel found in {search_box}")
        cols = np.flatnonzero(dark[rows, :].mean(axis=0) > 0.25)
        if not len(cols):
            raise RuntimeError(f"No dark panel found in {search_box}")
        panel = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
        # Clear the complete old panel, then add only the extracted bright detail.
        base.paste(
            Image.new("L", (source.width, panel[3] - panel[1]), 255),
            (search_box[0], search_box[1] + panel[1]),
        )
        processed = bright_features_on_white(
            source.crop(panel), include_broad_bright_areas=include_broad
        )
        if args.thin_px and not args.thin_full_image:
            processed = thin_heavy_linework(processed, args.thin_px)
        base.paste(processed, (search_box[0] + panel[0], search_box[1] + panel[1]))

    # Search rectangles are in the 70 x 70 mm SVG coordinate system.
    replace_dark_panel(box_mm(54.48819, 13.736439, 7.1567187, 9.3907213), include_broad=True)
    replace_dark_panel(box_mm(54.5, 23.5, 8.0, 12.5), include_broad=True)
    replace_dark_panel(box_mm(7.8, 43.8, 10.3, 18.8), include_broad=False)

    if args.thin_px and args.thin_full_image:
        original = base.copy()
        base = thin_heavy_linework(base, args.thin_px)
        # Never thin the main identity/header text.
        header_box = box_mm(0.0, 0.0, 70.0, 10.5)
        base.paste(original.crop(header_box), (header_box[0], header_box[1]))

        # Lightly narrow each paper title. Blending the original with a
        # one-pixel erosion gives about half-pixel narrowing after the artwork
        # is sampled for the STL, without breaking small letters.
        for title_box_mm in (
            (8.0, 11.7, 25.0, 4.0),
            (37.0, 11.7, 26.0, 4.0),
            (8.0, 40.7, 25.0, 4.0),
            (37.0, 40.7, 26.0, 4.5),
        ):
            title_box = box_mm(*title_box_mm)
            original_title = original.crop(title_box)
            eroded_title = original_title.filter(ImageFilter.MaxFilter(3))
            title_crop = Image.blend(original_title, eroded_title, 0.45)
            base.paste(title_crop, (title_box[0], title_box[1]))

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    base.save(args.output_png, optimize=True, dpi=(600, 600))
    print(f"Wrote: {args.output_png}")


if __name__ == "__main__":
    main()
