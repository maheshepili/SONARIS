import unittest

from PIL import Image

from src.data.preprocess import preprocess_sonar_image


class PreprocessTests(unittest.TestCase):
    def test_preprocessing_preserves_dimensions_and_rgb_output(self):
        source = Image.new("L", (12, 8), color=128)
        result = preprocess_sonar_image(source)
        self.assertEqual(result.size, (12, 8))
        self.assertEqual(result.mode, "RGB")

    def test_invalid_percentiles_are_rejected(self):
        with self.assertRaises(ValueError):
            preprocess_sonar_image(Image.new("L", (1, 1)), 99, 1)
