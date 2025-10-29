# spot_svg — image → limited-palette SVG

This small tool extracts a limited palette (default 8 colors) from an input image and produces an SVG where areas dominated by each palette color are rendered as simple polygon "islands".

Quick start

1. Create a virtual environment and install requirements:

```powershell
# from the project root
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the converter (k-means palette):

```powershell
python src\image_to_svg.py input.jpg output.svg --colors 8 --max-size 800 --pixel-size 2
```

Palette from file (nearest-color mapping)

- Provide a text file containing one hex color per line (with or without leading `#`). Example: `tests/fixtures/colors.txt`.
- If `--palette-file` is provided, the program maps each pixel to the nearest color from the file instead of running k-means.
- If `--colors N` is provided and the file contains more than N colors, the program automatically selects N colors from the file that best cover the image (greedy facility-location on sampled pixels, using LAB color space for perceptual accuracy), rather than just taking the first N.
- If `--colors` is omitted, all colors from the file are used.

Example:

```powershell
python src\image_to_svg.py input.jpg output.svg --palette-file tests\fixtures\colors.txt --colors 8
```

Notes
- The implementation downsamples large images for speed (default max dimension 800px).
- The SVG islands are computed by finding connected regions of pixels assigned to the same palette color and approximating each region with its convex hull polygon for compactness.
- This is a starting point; you can tune `--pixel-size` and `--min-region` to influence appearance.

## Command Line Options

The following options are available for the `image_to_svg.py` script:

- `input` (required): Path to the input image file.
- `output` (required): Path to the output SVG file.
- `--colors N` (optional): Number of palette colors to use (default: 8).
- `--max-size N` (optional): Maximum dimension (width or height) to downscale the input image for performance (default: 800).
- `--pixel-size N` (optional): Size of each image pixel in SVG units (default: 2).
- `--min-region N` (optional): Minimum region size (in pixels) to render (default: 8).
- `--palette-file PATH` (optional): Path to a text file containing hex colors (one per line). If provided, the program maps each pixel to the nearest color from the file instead of running k-means.
- `--background HEX` (optional): Hex color to use for the SVG background (e.g., `FFFFFF`). Shapes of this color are not drawn, creating a stencil effect.

Files
- `src/palette_svg/__init__.py` — core library
- `src/image_to_svg.py` — CLI wrapper
- `requirements.txt` — dependencies
- `tests/` — minimal tests

License
AGPL
