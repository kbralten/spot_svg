"""Small CLI wrapper to call the library from the command line.
"""
from palette_svg import image_to_svg_file


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Image to palette SVG')
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--colors', type=int, default=8)
    parser.add_argument('--max-size', type=int, default=800)
    parser.add_argument('--pixel-size', type=int, default=2)
    parser.add_argument('--min-region', type=int, default=8)
    parser.add_argument('--palette-file', type=str, default=None, help='Path to text file with hex colors, one per line. If provided, nearest-color mapping is used instead of k-means.')
    parser.add_argument('--background', type=str, default=None, help='Hex color to use for the SVG background (e.g., FFFFFF). When provided, shapes of this color are not drawn.')
    args = parser.parse_args()
    image_to_svg_file(args.input, args.output, n_colors=args.colors, max_size=args.max_size, pixel_size=args.pixel_size, min_region_size=args.min_region, palette_file=args.palette_file, background_color_hex=args.background)


if __name__ == '__main__':
    main()
