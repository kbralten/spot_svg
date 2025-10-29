"""Unit tests for core palette_svg functions using unittest framework."""
import unittest
import os
import tempfile
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Ensure tests can import the local package by adding the project's src/ to sys.path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

try:
    from palette_svg import (
        load_image,
        extract_palette,
        map_pixels_to_palette,
        load_palette_from_file,
        select_palette_subset_for_image,
        image_to_svg_file
    )
except ModuleNotFoundError as e:
    # If core dependencies aren't installed in the environment (sklearn, skimage, etc.),
    # skip the entire test module with a clear message.
    raise unittest.SkipTest(f"Missing dependency while importing tests: {e}")


class TestLoadImage(unittest.TestCase):
    """Test image loading and downscaling."""

    def test_load_small_image(self):
        """Test loading an image that doesn't need downscaling."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (100, 100), (255, 0, 0))
            img.save(f.name)
            temp_path = f.name

        try:
            loaded = load_image(temp_path, max_size=800)
            self.assertEqual(loaded.size, (100, 100))
        finally:
            os.unlink(temp_path)

    def test_load_large_image_downscale(self):
        """Test that large images are downscaled."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (1600, 1200), (0, 255, 0))
            img.save(f.name)
            temp_path = f.name

        try:
            loaded = load_image(temp_path, max_size=800)
            self.assertLessEqual(max(loaded.size), 800)
            self.assertEqual(loaded.mode, 'RGB')
        finally:
            os.unlink(temp_path)


class TestExtractPalette(unittest.TestCase):
    """Test k-means color palette extraction."""

    def test_extract_palette_shape(self):
        """Test that extracted palette has correct shape."""
        img = Image.new('RGB', (100, 100))
        for y in range(100):
            for x in range(100):
                img.putpixel((x, y), (x * 2, y * 2, 128))
        
        palette = extract_palette(img, n_colors=5)
        self.assertEqual(palette.shape, (5, 3))
        self.assertEqual(palette.dtype, np.uint8)

    def test_extract_palette_uniform_image(self):
        """Test palette extraction on uniform color image."""
        img = Image.new('RGB', (50, 50), (100, 150, 200))
        palette = extract_palette(img, n_colors=3)
        self.assertEqual(palette.shape, (3, 3))
        # All colors should be close to the uniform color
        for color in palette:
            dist = np.linalg.norm(color.astype(float) - np.array([100, 150, 200]))
            self.assertLess(dist, 50)


class TestMapPixelsToPalette(unittest.TestCase):
    """Test pixel-to-palette mapping."""

    def test_map_simple_image(self):
        """Test mapping with a simple two-color image."""
        img = Image.new('RGB', (20, 10))
        for y in range(10):
            for x in range(20):
                img.putpixel((x, y), (255, 0, 0) if x < 10 else (0, 0, 255))
        
        palette = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
        labelled = map_pixels_to_palette(img, palette)
        
        self.assertEqual(labelled.shape, (10, 20))
        # Left half should be one color, right half another
        self.assertTrue(np.all(labelled[:, :10] == labelled[0, 0]))
        self.assertTrue(np.all(labelled[:, 10:] == labelled[0, 10]))
        self.assertNotEqual(labelled[0, 0], labelled[0, 10])


class TestLoadPaletteFromFile(unittest.TestCase):
    """Test loading palette from hex color file."""

    def test_load_valid_palette_file(self):
        """Test loading a valid palette file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('FF0000\n')
            f.write('#00FF00\n')
            f.write('0000FF\n')
            temp_path = f.name

        try:
            palette = load_palette_from_file(temp_path)
            self.assertEqual(palette.shape, (3, 3))
            np.testing.assert_array_equal(palette[0], [255, 0, 0])
            np.testing.assert_array_equal(palette[1], [0, 255, 0])
            np.testing.assert_array_equal(palette[2], [0, 0, 255])
        finally:
            os.unlink(temp_path)

    def test_load_palette_with_max_colors(self):
        """Test loading palette with max_colors limit."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for i in range(10):
                f.write(f'{i:02x}{i:02x}{i:02x}\n')
            temp_path = f.name

        try:
            palette = load_palette_from_file(temp_path, max_colors=5)
            self.assertEqual(palette.shape[0], 5)
        finally:
            os.unlink(temp_path)

    def test_load_palette_ignores_blank_lines(self):
        """Test that blank lines are ignored."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('FF0000\n')
            f.write('\n')
            f.write('00FF00\n')
            f.write('  \n')
            temp_path = f.name

        try:
            palette = load_palette_from_file(temp_path)
            self.assertEqual(palette.shape[0], 2)
        finally:
            os.unlink(temp_path)


class TestSelectPaletteSubset(unittest.TestCase):
    """Test greedy palette subset selection."""

    def test_select_subset_smaller_than_candidates(self):
        """Test selecting k colors from a larger set."""
        img = Image.new('RGB', (50, 50))
        for y in range(50):
            for x in range(50):
                img.putpixel((x, y), (255, 0, 0) if x < 25 else (0, 0, 255))
        
        candidates = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 0, 0]
        ], dtype=np.uint8)
        
        selected = select_palette_subset_for_image(img, candidates, k=2)
        self.assertEqual(selected.shape, (2, 3))

    def test_select_subset_k_equals_candidates(self):
        """Test that k >= candidates returns all candidates."""
        img = Image.new('RGB', (20, 20), (128, 128, 128))
        candidates = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        
        selected = select_palette_subset_for_image(img, candidates, k=3)
        self.assertEqual(selected.shape, (2, 3))


class TestEndToEnd(unittest.TestCase):
    """Test end-to-end SVG generation."""

    def test_image_to_svg_basic(self):
        """Test basic image to SVG conversion."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as img_file:
            img = Image.new('RGB', (40, 40))
            for y in range(40):
                for x in range(40):
                    if x < 20:
                        img.putpixel((x, y), (255, 0, 0))
                    else:
                        img.putpixel((x, y), (0, 0, 255))
            img.save(img_file.name)
            img_path = img_file.name

        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as svg_file:
            svg_path = svg_file.name

        try:
            image_to_svg_file(img_path, svg_path, n_colors=2, max_size=100, pixel_size=2)
            self.assertTrue(os.path.exists(svg_path))
            self.assertGreater(os.path.getsize(svg_path), 0)
            
            # Check that SVG contains expected elements
            with open(svg_path, 'r') as f:
                content = f.read()
                self.assertIn('<svg', content)
                self.assertIn('</svg>', content)
        finally:
            os.unlink(img_path)
            if os.path.exists(svg_path):
                os.unlink(svg_path)

    def test_image_to_svg_with_palette_file(self):
        """Test SVG generation with palette file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as img_file:
            img = Image.new('RGB', (30, 30), (100, 100, 100))
            img.save(img_file.name)
            img_path = img_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as pal_file:
            pal_file.write('FF0000\n00FF00\n0000FF\n')
            pal_path = pal_file.name

        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as svg_file:
            svg_path = svg_file.name

        try:
            image_to_svg_file(img_path, svg_path, n_colors=2, palette_file=pal_path)
            self.assertTrue(os.path.exists(svg_path))
        finally:
            os.unlink(img_path)
            os.unlink(pal_path)
            if os.path.exists(svg_path):
                os.unlink(svg_path)

    def test_image_to_svg_with_background(self):
        """Test SVG generation with background color."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as img_file:
            img = Image.new('RGB', (30, 30))
            for y in range(30):
                for x in range(30):
                    img.putpixel((x, y), (255, 255, 255) if x < 15 else (0, 0, 0))
            img.save(img_file.name)
            img_path = img_file.name

        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as svg_file:
            svg_path = svg_file.name

        try:
            image_to_svg_file(img_path, svg_path, n_colors=2, background_color_hex='FFFFFF')
            self.assertTrue(os.path.exists(svg_path))
            
            with open(svg_path, 'r') as f:
                content = f.read()
                self.assertIn('#FFFFFF', content)
        finally:
            os.unlink(img_path)
            if os.path.exists(svg_path):
                os.unlink(svg_path)

    def test_holes_render_with_evenodd(self):
        """Shapes with internal holes should render using even-odd fill in a single path."""
        # Create a ring on white background: black outer rect with white inner rect (hole)
        img = Image.new('RGB', (40, 40), (255, 255, 255))
        for y in range(5, 35):
            for x in range(5, 35):
                img.putpixel((x, y), (0, 0, 0))
        for y in range(15, 25):
            for x in range(15, 25):
                img.putpixel((x, y), (255, 255, 255))

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as img_file:
            img_path = img_file.name
            img.save(img_path)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as pal_file:
            pal_file.write('000000\n')  # only black; background will be white
            pal_path = pal_file.name

        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as svg_file:
            svg_path = svg_file.name

        try:
            image_to_svg_file(
                img_path,
                svg_path,
                n_colors=1,
                max_size=100,
                pixel_size=2,
                min_region_size=1,
                palette_file=pal_path,
                background_color_hex='FFFFFF'
            )
            self.assertTrue(os.path.exists(svg_path))
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_text = f.read()
            # Ensure even-odd fill is present
            self.assertIn('fill-rule="evenodd"', svg_text)
            # Heuristic: compound path should have multiple 'M ' move commands
            self.assertGreaterEqual(svg_text.count('M '), 2)
        finally:
            os.unlink(img_path)
            os.unlink(pal_path)
            if os.path.exists(svg_path):
                os.unlink(svg_path)


if __name__ == '__main__':
    unittest.main()
