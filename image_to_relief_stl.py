"""Convert a raster image into a binary STL for relief or laser features.

Dark pixels become raised geometry. The default mode creates a watertight
bas-relief slab. ``--features-only`` instead creates separate constant-height
closed solids with no background plate, like an extruded vector design. STL
does not store units; this script uses millimetres by convention.

Requirements:
    python -m pip install pillow numpy

Example:
    python image_to_relief_stl.py input.png output.stl --features-only \
        --size 70 --relief 0.5 --threshold 0.12 --resolution 512
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def load_ink_map(
    path: Path,
    resolution: int,
    mirror: bool,
) -> np.ndarray:
    image = Image.open(path).convert("L")
    image = ImageOps.fit(
        image,
        (resolution, resolution),
        method=Image.Resampling.LANCZOS,
    )
    image = image.filter(ImageFilter.GaussianBlur(radius=0.45))
    if mirror:
        image = ImageOps.mirror(image)

    gray = np.asarray(image, dtype=np.float32) / 255.0
    ink = 1.0 - gray
    ink = np.power(ink, 0.90)

    # Image row zero is visually at the top. Flip it so the design appears
    # upright when viewed from +Z in Cartesian coordinates.
    return np.flipud(ink)


def load_height_map(
    path: Path,
    resolution: int,
    base_mm: float,
    relief_mm: float,
    mirror: bool,
    threshold: float,
) -> np.ndarray:
    ink = load_ink_map(path, resolution, mirror)
    ink = np.where(
        ink >= threshold,
        (ink - threshold) / (1.0 - threshold),
        0.0,
    )
    return base_mm + relief_mm * ink


def make_mesh(height: np.ndarray, size_mm: float) -> tuple[np.ndarray, np.ndarray]:
    nrows, ncols = height.shape
    if nrows != ncols:
        raise ValueError("The fitted height map must be square")
    n = nrows

    x = np.linspace(0.0, size_mm, n, dtype=np.float32)
    y = np.linspace(0.0, size_mm, n, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    top = np.column_stack((xx.ravel(), yy.ravel(), height.ravel())).astype(
        np.float32
    )

    # Counter-clockwise perimeter when viewed from +Z.
    boundary = np.concatenate(
        (
            np.arange(0, n, dtype=np.int64),
            np.arange(2 * n - 1, n * n, n, dtype=np.int64),
            np.arange(n * n - 2, n * (n - 1) - 1, -1, dtype=np.int64),
            np.arange(n * (n - 2), 0, -n, dtype=np.int64),
        )
    )

    bottom_boundary_start = len(top)
    bottom_boundary = top[boundary].copy()
    bottom_boundary[:, 2] = 0.0
    bottom_center_index = bottom_boundary_start + len(boundary)
    vertices = np.vstack(
        (
            top,
            bottom_boundary,
            np.array([[size_mm / 2, size_mm / 2, 0.0]], dtype=np.float32),
        )
    )

    # Two upward-facing triangles for every height-map cell.
    row = np.arange(n - 1, dtype=np.int64)[:, None]
    col = np.arange(n - 1, dtype=np.int64)[None, :]
    v00 = row * n + col
    v10 = v00 + 1
    v01 = v00 + n
    v11 = v01 + 1
    top_faces = np.stack(
        (
            np.stack((v00, v10, v11), axis=-1),
            np.stack((v00, v11, v01), axis=-1),
        ),
        axis=-2,
    ).reshape(-1, 3)

    m = len(boundary)
    next_index = np.roll(np.arange(m, dtype=np.int64), -1)
    bottom_indices = bottom_boundary_start + np.arange(m, dtype=np.int64)
    next_bottom = bottom_boundary_start + next_index
    next_top = boundary[next_index]

    # Side faces point outwards. Bottom fan faces point towards -Z.
    side_faces = np.vstack(
        (
            np.column_stack((boundary, next_bottom, next_top)),
            np.column_stack((boundary, bottom_indices, next_bottom)),
        )
    )
    bottom_faces = np.column_stack(
        (
            np.full(m, bottom_center_index, dtype=np.int64),
            next_bottom,
            bottom_indices,
        )
    )

    faces = np.vstack((top_faces, side_faces, bottom_faces)).astype(np.int64)
    return vertices, faces


def make_features_only_mesh(
    ink: np.ndarray,
    size_mm: float,
    feature_height_mm: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrude thresholded image cells as separate watertight solids.

    Adjacent dark cells are joined. Boundary walls are emitted only where a
    dark cell touches a white cell or the image edge, so there is no plate and
    no internal wall geometry.
    """
    nrows, ncols = ink.shape
    if nrows != ncols:
        raise ValueError("The fitted ink map must be square")
    mask = ink >= threshold
    # Two pixels touching only at a corner create a non-manifold vertical edge
    # after extrusion. Join each such pair through the stronger of the two
    # intervening pixels. This keeps fine strokes connected and every STL edge
    # incident to exactly two triangles.
    for _ in range(8):
        a = mask[:-1, :-1]
        b = mask[:-1, 1:]
        c = mask[1:, :-1]
        d = mask[1:, 1:]
        diagonal_ad = a & d & ~b & ~c
        diagonal_bc = b & c & ~a & ~d
        if not np.any(diagonal_ad) and not np.any(diagonal_bc):
            break
        additions = np.zeros_like(mask)

        row, col = np.nonzero(diagonal_ad)
        choose_b = ink[row, col + 1] >= ink[row + 1, col]
        additions[row[choose_b], col[choose_b] + 1] = True
        additions[row[~choose_b] + 1, col[~choose_b]] = True

        row, col = np.nonzero(diagonal_bc)
        choose_a = ink[row, col] >= ink[row + 1, col + 1]
        additions[row[choose_a], col[choose_a]] = True
        additions[row[~choose_a] + 1, col[~choose_a] + 1] = True
        mask |= additions
    if not np.any(mask):
        raise ValueError(
            "The threshold removed every feature; lower --threshold"
        )

    n = nrows
    grid_n = n + 1
    x = np.linspace(0.0, size_mm, grid_n, dtype=np.float32)
    y = np.linspace(0.0, size_mm, grid_n, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    bottom = np.column_stack(
        (xx.ravel(), yy.ravel(), np.zeros(grid_n * grid_n, dtype=np.float32))
    )
    top = bottom.copy()
    top[:, 2] = feature_height_mm
    vertices = np.vstack((bottom, top)).astype(np.float32)
    top_offset = len(bottom)

    row, col = np.nonzero(mask)
    b00 = row * grid_n + col
    b10 = b00 + 1
    b01 = b00 + grid_n
    b11 = b01 + 1
    t00 = b00 + top_offset
    t10 = b10 + top_offset
    t01 = b01 + top_offset
    t11 = b11 + top_offset

    # Flat top (+Z) and bottom (-Z) for every retained image cell.
    top_faces = np.vstack(
        (
            np.column_stack((t00, t10, t11)),
            np.column_stack((t00, t11, t01)),
        )
    )
    bottom_faces = np.vstack(
        (
            np.column_stack((b00, b11, b10)),
            np.column_stack((b00, b01, b11)),
        )
    )

    lower_edge = np.zeros_like(mask)
    lower_edge[0, :] = mask[0, :]
    lower_edge[1:, :] = mask[1:, :] & ~mask[:-1, :]
    upper_edge = np.zeros_like(mask)
    upper_edge[-1, :] = mask[-1, :]
    upper_edge[:-1, :] = mask[:-1, :] & ~mask[1:, :]
    left_edge = np.zeros_like(mask)
    left_edge[:, 0] = mask[:, 0]
    left_edge[:, 1:] = mask[:, 1:] & ~mask[:, :-1]
    right_edge = np.zeros_like(mask)
    right_edge[:, -1] = mask[:, -1]
    right_edge[:, :-1] = mask[:, :-1] & ~mask[:, 1:]

    side_groups: list[np.ndarray] = []
    for edge, orientation in (
        (lower_edge, "lower"),
        (upper_edge, "upper"),
        (left_edge, "left"),
        (right_edge, "right"),
    ):
        erow, ecol = np.nonzero(edge)
        e00 = erow * grid_n + ecol
        e10 = e00 + 1
        e01 = e00 + grid_n
        e11 = e01 + 1
        et00 = e00 + top_offset
        et10 = e10 + top_offset
        et01 = e01 + top_offset
        et11 = e11 + top_offset
        if orientation == "lower":
            group = np.vstack(
                (
                    np.column_stack((e00, et10, et00)),
                    np.column_stack((e00, e10, et10)),
                )
            )
        elif orientation == "upper":
            group = np.vstack(
                (
                    np.column_stack((e01, et01, et11)),
                    np.column_stack((e01, et11, e11)),
                )
            )
        elif orientation == "left":
            group = np.vstack(
                (
                    np.column_stack((e00, et00, et01)),
                    np.column_stack((e00, et01, e01)),
                )
            )
        else:
            group = np.vstack(
                (
                    np.column_stack((e10, et11, et10)),
                    np.column_stack((e10, e11, et11)),
                )
            )
        side_groups.append(group)

    faces = np.vstack((top_faces, bottom_faces, *side_groups)).astype(np.int64)
    return vertices, faces


def make_smooth_features_mesh(
    ink: np.ndarray,
    size_mm: float,
    feature_height_mm: float,
    threshold: float,
    simplify_mm: float,
    smoothing_mm: float,
    min_feature_area_mm2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace, smooth, triangulate, and extrude dark image contours."""
    try:
        from shapely import constrained_delaunay_triangles
        from shapely.geometry import LineString, Polygon
        from shapely.geometry.polygon import orient
        from shapely.ops import polygonize, polylabel, unary_union
        from skimage.measure import find_contours
    except ImportError as error:
        raise RuntimeError(
            "Smooth features require Shapely and scikit-image. Install them "
            "with: python -m pip install shapely scikit-image"
        ) from error

    nrows, ncols = ink.shape
    if nrows != ncols:
        raise ValueError("The fitted ink map must be square")
    n = nrows
    padded = np.pad(ink, 1, mode="constant", constant_values=0)
    scale = size_mm / n
    lines = []
    for contour in find_contours(
        padded,
        threshold,
        fully_connected="high",
    ):
        xy = np.column_stack(
            (
                (contour[:, 1] - 1) * scale,
                (contour[:, 0] - 1) * scale,
            )
        )
        xy = np.clip(xy, 0.0, size_mm)
        if len(xy) >= 4:
            xy[-1] = xy[0]
            lines.append(LineString(xy))
    if not lines:
        raise ValueError("The threshold removed every feature; lower --threshold")

    # Polygonization partitions nested outlines into foreground and holes.
    # Sample the original ink at the deepest interior point of each partition.
    # A generic representative point can land on an antialiased edge of a
    # short word and misclassify the whole word as background.
    cells = list(polygonize(lines))
    foreground = []
    for polygon in cells:
        point = polylabel(polygon, tolerance=scale / 2.0)
        col = min(n - 1, max(0, int(point.x / size_mm * n)))
        row = min(n - 1, max(0, int(point.y / size_mm * n)))
        if ink[row, col] >= threshold:
            foreground.append(polygon)
    if not foreground:
        raise ValueError("No closed foreground contours were found")

    geometry = unary_union(foreground)
    if simplify_mm > 0:
        geometry = geometry.simplify(simplify_mm, preserve_topology=True)
    if smoothing_mm > 0:
        geometry = geometry.buffer(
            smoothing_mm,
            quad_segs=3,
            join_style="round",
        ).buffer(
            -smoothing_mm,
            quad_segs=3,
            join_style="round",
        )

    if isinstance(geometry, Polygon):
        polygons = [geometry]
    else:
        polygons = [part for part in geometry.geoms if isinstance(part, Polygon)]
    polygons = [
        polygon
        for polygon in polygons
        if polygon.area >= min_feature_area_mm2
    ]
    if not polygons:
        raise ValueError(
            "All features were removed; lower --threshold or --min-feature-area"
        )

    triangle_vertices: list[np.ndarray] = []
    for polygon in polygons:
        # Exterior rings become counter-clockwise and holes clockwise, leaving
        # the solid interior on the left side of every directed boundary.
        polygon = orient(polygon, sign=1.0)
        for triangle in constrained_delaunay_triangles(polygon).geoms:
            xyz = np.zeros((3, 3), dtype=np.float32)
            xyz[:, :2] = np.asarray(
                triangle.exterior.coords,
                dtype=np.float32,
            )[:3]
            if np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])[2] < 0:
                xyz[[1, 2]] = xyz[[2, 1]]
            top = xyz.copy()
            top[:, 2] = feature_height_mm
            triangle_vertices.append(top)
            triangle_vertices.append(xyz[[0, 2, 1]])

        for ring in (polygon.exterior, *polygon.interiors):
            coordinates = np.asarray(ring.coords, dtype=np.float32)
            for point_a, point_b in zip(coordinates[:-1], coordinates[1:]):
                bottom_a = np.array(
                    [point_a[0], point_a[1], 0.0], dtype=np.float32
                )
                bottom_b = np.array(
                    [point_b[0], point_b[1], 0.0], dtype=np.float32
                )
                top_a = np.array(
                    [point_a[0], point_a[1], feature_height_mm],
                    dtype=np.float32,
                )
                top_b = np.array(
                    [point_b[0], point_b[1], feature_height_mm],
                    dtype=np.float32,
                )
                triangle_vertices.append(np.stack((bottom_a, top_b, top_a)))
                triangle_vertices.append(np.stack((bottom_a, bottom_b, top_b)))

    vertices = np.asarray(triangle_vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    triangles = vertices[faces].astype(np.float32)
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    # Contour simplification can occasionally collapse a side segment to less
    # than float32 precision. Such zero-area facets add no geometry and can
    # create formally non-manifold edge counts, so omit them from the STL.
    keep = np.isfinite(lengths) & (lengths > 1e-10)
    triangles = triangles[keep]
    normals = normals[keep]
    lengths = lengths[keep]
    normals /= lengths[:, None]

    record_type = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("v3", "<f4", (3,)),
            ("attribute", "<u2"),
        ]
    )
    records = np.zeros(len(triangles), dtype=record_type)
    records["normal"] = normals
    records["v1"] = triangles[:, 0]
    records["v2"] = triangles[:, 1]
    records["v3"] = triangles[:, 2]

    header = b"Raster-to-STL generated geometry; dimensions in mm"
    header = header[:80].ljust(80, b" ")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<I", len(triangles)))
        records.tofile(stream)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="grayscale PNG or JPEG")
    parser.add_argument("output", type=Path, help="output binary STL")
    parser.add_argument("--size", type=float, default=70.0, help="XY size in mm")
    parser.add_argument("--base", type=float, default=0.8, help="base thickness in mm")
    parser.add_argument(
        "--relief", type=float, default=0.6, help="maximum relief height in mm"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="height-map samples per axis; 512 produces about 26 MB",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.12,
        help="minimum darkness retained, from 0 to 1; higher removes more",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="trace and extrude only dark features; omit the background plate",
    )
    parser.add_argument(
        "--pixel-features",
        action="store_true",
        help="use square raster cells instead of smooth traced contours",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=0.035,
        help="smooth-contour simplification tolerance in mm; default 0.035",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.025,
        help="smooth-contour rounding radius in mm; default 0.025",
    )
    parser.add_argument(
        "--min-feature-area",
        type=float,
        default=0.01,
        help="discard smaller smooth features in square mm; default 0.01",
    )
    parser.add_argument(
        "--mirror", action="store_true", help="mirror horizontally for back-face use"
    )
    args = parser.parse_args()
    if args.size <= 0 or args.base <= 0 or args.relief < 0:
        parser.error("size and base must be positive; relief cannot be negative")
    if not 32 <= args.resolution <= 1200:
        parser.error("resolution must be between 32 and 1200")
    if not 0.0 <= args.threshold < 1.0:
        parser.error("threshold must be at least 0 and less than 1")
    if args.simplify < 0 or args.smoothing < 0 or args.min_feature_area < 0:
        parser.error("simplify, smoothing, and min-feature-area cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    if args.features_only:
        ink = load_ink_map(args.input, args.resolution, args.mirror)
        if args.pixel_features:
            vertices, faces = make_features_only_mesh(
                ink,
                args.size,
                args.relief,
                args.threshold,
            )
            mode = "pixel features only"
        else:
            vertices, faces = make_smooth_features_mesh(
                ink,
                args.size,
                args.relief,
                args.threshold,
                args.simplify,
                args.smoothing,
                args.min_feature_area,
            )
            mode = "smooth features only"
        z_max = args.relief
    else:
        height = load_height_map(
            args.input,
            args.resolution,
            args.base,
            args.relief,
            args.mirror,
            args.threshold,
        )
        vertices, faces = make_mesh(height, args.size)
        z_max = float(height.max())
        mode = "watertight bas-relief"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_binary_stl(args.output, vertices, faces)
    print(f"Wrote: {args.output}")
    print(f"Mode: {mode}")
    print(f"Size: {args.size:.3f} x {args.size:.3f} mm")
    print(f"Z range: 0.000 to {z_max:.3f} mm")
    print(f"Vertices: {len(vertices):,}")
    print(f"Triangles: {len(faces):,}")


if __name__ == "__main__":
    main()
