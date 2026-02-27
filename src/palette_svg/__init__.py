"""palette_svg

Core functions to extract dominant colors and generate an SVG of color "islands".
"""
from __future__ import annotations

from typing import List, Tuple, Optional
import math
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import svgwrite
from skimage import measure
from skimage.color import rgb2lab, lab2rgb


def load_image(path: str, max_size: int = 800) -> Image.Image:
    img = Image.open(path).convert("RGB")
    # downscale large images for performance
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        # Pillow 10+ moved resampling filters under Image.Resampling
        try:
            resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except Exception:
            # Older Pillow may expose LANCZOS at module level or ANTIALIAS
            resample = getattr(Image, 'LANCZOS', getattr(Image, 'ANTIALIAS', None))
            if resample is None:
                # Pillow fallback constant
                resample = getattr(Image, 'BICUBIC', 3)
        img = img.resize(new_size, resample)
    return img


def extract_palette(img: Image.Image, n_colors: int = 8, sample_size: int = 10000) -> np.ndarray:
    """Extract n_colors dominant RGB colors from an image using k-means.

    Returns an (n_colors, 3) array of uint8 colors.
    """
    arr = np.array(img)
    h, w, c = arr.shape
    # Use LAB color space for k-means for more perceptually uniform clustering
    pixels_rgb = arr.reshape((-1, 3)).astype(float) / 255.0
    pixels_lab = rgb2lab(pixels_rgb)

    if len(pixels_lab) > sample_size:
        idx = np.random.choice(len(pixels_lab), sample_size, replace=False)
        sample = pixels_lab[idx]
    else:
        sample = pixels_lab
    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init=10)
    kmeans.fit(sample)
    
    # Convert cluster centers from LAB back to RGB
    centers_lab = kmeans.cluster_centers_
    centers_rgb = lab2rgb(centers_lab)
    centers = np.clip(centers_rgb * 255, 0, 255).astype(np.uint8)
    return centers


def _parse_hex_color(s: str) -> Optional[Tuple[int, int, int]]:
    s = s.strip()
    if not s:
        return None
    if s.startswith('#'):
        s = s[1:]
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)
    except ValueError:
        return None


def load_palette_from_file(path: str, max_colors: Optional[int] = None) -> np.ndarray:
    """Load colors from a text file (one hex per line). Returns (n,3) uint8 array.

    - Accepts lines like '#RRGGBB' or 'RRGGBB'.
    - Ignores blank lines; stops at max_colors if provided.
    - Keeps file order.
    """
    colors: List[Tuple[int, int, int]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            c = _parse_hex_color(line)
            if c is not None:
                colors.append(c)
                if max_colors is not None and len(colors) >= max_colors:
                    break
    if not colors:
        raise ValueError(f"No valid colors found in palette file: {path}")
    arr = np.array(colors, dtype=np.uint8)
    return arr


def select_palette_subset_for_image(img: Image.Image, candidates: np.ndarray, k: int, sample_size: int = 10000) -> np.ndarray:
    """Pick k colors from candidate palette that best cover the image colors.

    Greedy facility-location: iteratively add the candidate that most reduces the
    sum of distances from sampled image pixels to the nearest selected color.
    - img: PIL Image in RGB
    - candidates: (m,3) uint8
    - k: number of colors to select (k <= m)
    - sample_size: number of pixels to sample for scoring
    Returns (k,3) uint8 array in the order selected.
    """
    m = candidates.shape[0]
    if k >= m:
        return candidates.copy()

    arr = np.array(img)
    pixels_rgb = arr.reshape((-1, 3)).astype(float) / 255.0
    if len(pixels_rgb) > sample_size:
        idx = np.random.choice(len(pixels_rgb), sample_size, replace=False)
        pixels_rgb = pixels_rgb[idx]
    
    pixels_lab = rgb2lab(pixels_rgb)
    candidates_lab = rgb2lab(candidates[np.newaxis, :, :]/255.0)[0]

    # Precompute distances from pixels to all candidates (N x m) in LAB space
    dists = np.sqrt(((pixels_lab[:, None, :] - candidates_lab[None, :, :]) ** 2).sum(axis=2))
    # Greedy selection
    selected: List[int] = []
    remaining = list(range(m))
    # Start with candidate that minimizes total distance alone
    totals = dists.sum(axis=0)  # (m,)
    first = int(np.argmin(totals))
    selected.append(first)
    remaining.remove(first)
    current_min = dists[:, first].copy()

    for _ in range(1, k):
        # Initialize with the first remaining candidate to satisfy type checkers
        init_j = remaining[0]
        best_c = init_j
        best_score = float(np.minimum(current_min, dists[:, init_j]).sum())
        for j in remaining[1:]:
            new_min = np.minimum(current_min, dists[:, j])
            score = float(new_min.sum())
            if score < best_score:
                best_score = score
                best_c = j
        selected.append(int(best_c))
        remaining.remove(int(best_c))
        current_min = np.minimum(current_min, dists[:, int(best_c)])

    return candidates[np.array(selected, dtype=int)]


def map_pixels_to_palette(img: Image.Image, palette: np.ndarray) -> np.ndarray:
    """Map each pixel to nearest palette index. Returns 2D int array of shape (h,w)."""
    arr = np.array(img)
    h, w, _ = arr.shape
    
    pixels_rgb = arr.reshape((-1, 3)).astype(float) / 255.0
    pixels_lab = rgb2lab(pixels_rgb)
    
    palette_rgb = palette.astype(float) / 255.0
    palette_lab = rgb2lab(palette_rgb[np.newaxis, :, :])[0]

    # compute distances in LAB space
    dists = np.sqrt(((pixels_lab[:, None, :] - palette_lab[None, :, :]) ** 2).sum(axis=2))
    idxs = np.argmin(dists, axis=1)
    return idxs.reshape((h, w))


def label_components(label_map: np.ndarray) -> Tuple[np.ndarray, int]:
    """Connected-component labeling for 2D arrays using skimage for performance.

    Returns (labels, n_components) where labels are 1..n for components (0 for background).
    """
    # Treat any nonzero entry as foreground; use 4-connectivity to match previous behavior
    labels = measure.label(label_map != 0, connectivity=1)
    n_components = int(labels.max())
    return labels.astype(np.int32, copy=False), n_components


def _convex_hull(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Andrew's monotone chain convex hull for 2D integer points.
    Returns points in CCW order without repeating the first point.
    """
    # sort by x then y
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return hull


def generate_svg_from_image(img: Image.Image, palette: np.ndarray, labelled: np.ndarray, out_path: str, pixel_size: int = 1, min_region_size: int = 10, background_color_hex: Optional[str] = None, background_color_idx: Optional[int] = None):
    """Generate an SVG file showing color islands.

    - img: PIL.Image (used for size)
    - palette: (n,3) uint8 array
    - labelled: 2D array of palette indices per pixel
    - pixel_size: scale factor for each image pixel in SVG
    - min_region_size: ignore components smaller than this (in pixels)
    - background_color_hex: if provided, sets the SVG background.
    - background_color_idx: index in palette corresponding to the color to be skipped (stenciled).
    """
    h, w = labelled.shape
    dwg = svgwrite.Drawing(out_path, size=(w*pixel_size, h*pixel_size))

    # Determine the SVG's base background fill.
    if background_color_hex:
        # A stencil color is provided, use it for the SVG background.
        dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill=f'#{background_color_hex.lstrip("#")}'))
    else:
        # No stencil. Use the image's most dominant color as the background.
        counts = np.bincount(labelled.flatten())
        image_bg_idx = np.argmax(counts)
        bg_color_rgb = tuple(int(v) for v in palette[image_bg_idx])
        bg_color_hex = '#%02x%02x%02x' % bg_color_rgb
        dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill=bg_color_hex))
        # In this mode, the color to skip is the image's own background.
        background_color_idx = int(image_bg_idx)

    n_colors = palette.shape[0]
    # Draw islands for all colors, skipping only the designated stencil/background color.
    for c in range(n_colors):
        if c == background_color_idx:
            continue

        mask = (labelled == c).astype(np.uint8)
        if mask.sum() == 0:
            continue
        comp_labels, ncomp = label_components(mask)
        color_rgb = tuple(int(v) for v in palette[c])
        color_hex = '#%02x%02x%02x' % color_rgb
        for comp_id in range(1, ncomp+1):
            ys, xs = np.where(comp_labels == comp_id)
            if len(xs) < min_region_size:
                continue
            # Try contour tracing to get a better (possibly concave) outline for the component
            comp_mask = (comp_labels == comp_id).astype(np.uint8)
            # Pad mask by 1px on all sides to ensure contours close at image edges
            comp_mask_padded = np.pad(comp_mask, pad_width=1, mode='constant', constant_values=0)
            # find_contours expects 2D array where rows are y, cols are x; returns many (row, col) contours
            contours = measure.find_contours(comp_mask_padded, level=0.5)
            if not contours:
                # fallback to bounding rect
                minx = int(min(xs)); maxx = int(max(xs))
                miny = int(min(ys)); maxy = int(max(ys))
                insert_x = int(minx * pixel_size)
                insert_y = int(miny * pixel_size)
                width = int((maxx - minx + 1) * pixel_size)
                height = int((maxy - miny + 1) * pixel_size)
                dwg.add(dwg.rect(insert=(insert_x, insert_y), size=(width, height), fill=color_hex, stroke='none'))
                continue

            # Build a compound path from all contours (outer and inner) and use even-odd fill
            subpaths: List[List[Tuple[float, float]]] = []
            for contour in contours:
                # Filter out very small contours before processing
                if len(contour) < 4:  # Need at least 4 points for a meaningful shape
                    continue
                try:
                    contour = measure.approximate_polygon(contour, tolerance=1.5)
                except Exception:
                    pass
                poly: List[Tuple[float, float]] = []
                for (ry, rx) in contour:
                    # subtract 1px padding to return to original coordinate system
                    ry -= 1.0
                    rx -= 1.0
                    # convert to pixel centers and scale
                    x = float(rx * pixel_size + pixel_size / 2.0)
                    y = float(ry * pixel_size + pixel_size / 2.0)
                    poly.append((x, y))
                if len(poly) >= 3:
                    # Calculate the area of the polygon to filter out tiny artifacts
                    area = 0.0
                    n = len(poly)
                    for i in range(n):
                        j = (i + 1) % n
                        area += poly[i][0] * poly[j][1]
                        area -= poly[j][0] * poly[i][1]
                    area = abs(area) / 2.0
                    # Only include contours with reasonable area (scaled by pixel_size)
                    min_area = min_region_size * pixel_size * pixel_size
                    if area >= min_area:
                        subpaths.append(poly)

            if not subpaths:
                # fallback bounding box
                minx = int(min(xs)); maxx = int(max(xs))
                miny = int(min(ys)); maxy = int(max(ys))
                insert_x = int(minx * pixel_size)
                insert_y = int(miny * pixel_size)
                width = int((maxx - minx + 1) * pixel_size)
                height = int((maxy - miny + 1) * pixel_size)
                dwg.add(dwg.rect(insert=(insert_x, insert_y), size=(width, height), fill=color_hex, stroke='none'))
                continue

            # Construct SVG path 'd' with multiple subpaths
            def _path_from_points(points: List[Tuple[float, float]]) -> str:
                d = []
                # Move to first point
                x0, y0 = points[0]
                d.append(f"M {x0:.2f},{y0:.2f}")
                # Line to subsequent points
                for (x, y) in points[1:]:
                    d.append(f"L {x:.2f},{y:.2f}")
                # Close path
                d.append("Z")
                return " ".join(d)

            d_parts: List[str] = []
            for pts in subpaths:
                d_parts.append(_path_from_points(pts))
            d_attr = " ".join(d_parts)

            path = dwg.path(d=d_attr, fill=color_hex, stroke='none')
            # Ensure holes render correctly regardless of orientation
            try:
                # svgwrite supports 'fill_rule' attribute
                path['fill-rule'] = 'evenodd'
            except Exception:
                # Fallback: assign via kwargs if supported
                try:
                    path.update({'fill-rule': 'evenodd'})
                except Exception:
                    pass
            dwg.add(path)

    dwg.save()


def image_to_svg_file(input_path: str, output_path: str, n_colors: int = 8, max_size: int = 800, pixel_size: int = 2, min_region_size: int = 8, palette_file: Optional[str] = None, background_color_hex: Optional[str] = None):
    img = load_image(input_path, max_size=max_size)
    
    bg_color_rgb = None
    if background_color_hex:
        bg_color_rgb = _parse_hex_color(background_color_hex)
        if bg_color_rgb is None:
            raise ValueError(f"Invalid background color hex: {background_color_hex}")

    if palette_file:
        # Load all candidates from file
        candidates = load_palette_from_file(palette_file, max_colors=None)
        # Build selection pool excluding background (we'll append background after selection)
        if bg_color_rgb is not None:
            bg_arr = np.array(bg_color_rgb, dtype=np.uint8)
            mask = ~(candidates == bg_arr).all(axis=1)
            pool = candidates[mask]
        else:
            pool = candidates

        if n_colors is None or n_colors >= len(pool):
            selected = pool
        else:
            selected = select_palette_subset_for_image(img, pool, n_colors)

        # Final palette: selected colors plus background (if provided)
        if bg_color_rgb is not None:
            palette = np.vstack([selected, np.array(bg_color_rgb, dtype=np.uint8)])
        else:
            palette = selected
    else:
        # K-means extraction: always keep n_colors image colors; append background as extra entry
        if n_colors > 0:
            base_palette = extract_palette(img, n_colors=n_colors)
        else:
            base_palette = np.empty((0,3), dtype=np.uint8)
        if bg_color_rgb is not None:
            palette = np.vstack([base_palette, np.array(bg_color_rgb, dtype=np.uint8)])
        else:
            palette = base_palette

    if palette.shape[0] == 0:
        import warnings
        warnings.warn("Could not determine a palette. Output will be empty.")
        dwg = svgwrite.Drawing(output_path, size=(img.width*pixel_size, img.height*pixel_size))
        if background_color_hex:
            dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill=f'#{background_color_hex}'))
        dwg.save()
        return

    labelled = map_pixels_to_palette(img, palette)
    
    bg_color_idx = None
    if bg_color_rgb is not None:
        # Prefer exact match if we appended it; fallback to nearest
        bg_arr = np.array(bg_color_rgb, dtype=np.uint8)
        eq = (palette == bg_arr).all(axis=1)
        if eq.any():
            bg_color_idx = int(np.argmax(eq))
        else:
            dists = np.linalg.norm(palette.astype(float) - np.array(bg_color_rgb, dtype=float), axis=1)
            bg_color_idx = int(np.argmin(dists))
    else:
        bg_color_idx = None

    generate_svg_from_image(img, palette, labelled, output_path, pixel_size=pixel_size, min_region_size=min_region_size, background_color_hex=background_color_hex, background_color_idx=bg_color_idx)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert image to limited-palette SVG with color islands')
    parser.add_argument('input', help='input image path')
    parser.add_argument('output', help='output svg path')
    parser.add_argument('--colors', type=int, default=8, help='number of palette colors')
    parser.add_argument('--max-size', type=int, default=800, help='max dimension to downscale for performance')
    parser.add_argument('--pixel-size', type=int, default=2, help='pixel size in SVG units')
    parser.add_argument('--min-region', type=int, default=8, help='minimum region size to render')
    args = parser.parse_args()
    image_to_svg_file(args.input, args.output, n_colors=args.colors, max_size=args.max_size, pixel_size=args.pixel_size, min_region_size=args.min_region)
