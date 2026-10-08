import unittest

import cv2
import numpy as np

from image_processing import (
    apply_clahe,
    canny_edges,
    connected_components_analysis,
    crop_image,
    denoise_image,
    morphology_open_close,
    otsu_segment,
    process_roi,
    to_grayscale,
)


class ImageProcessingTests(unittest.TestCase):
    def test_crop_clamps_bounds(self):
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        crop = crop_image(image, -3, 2, 5, 20)
        self.assertEqual(crop.shape, (8, 5))
        self.assertEqual(int(crop[0, 0]), 20)

    def test_color_to_grayscale(self):
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[..., 2] = 255
        gray = to_grayscale(bgr)
        self.assertEqual(gray.shape, (4, 4))
        self.assertGreater(int(gray[0, 0]), 70)

    def test_denoise_removes_impulse(self):
        image = np.zeros((9, 9), dtype=np.uint8)
        image[4, 4] = 255
        self.assertEqual(int(denoise_image(image, "median", 3)[4, 4]), 0)

    def test_clahe_preserves_shape_and_type(self):
        image = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
        result = apply_clahe(image)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_otsu_separates_two_levels(self):
        image = np.hstack((np.full((20, 20), 20, np.uint8), np.full((20, 20), 220, np.uint8)))
        threshold, binary = otsu_segment(image)
        self.assertGreaterEqual(threshold, 20)
        self.assertEqual(int(binary[:, :20].sum()), 0)
        self.assertTrue(np.all(binary[:, 20:] == 255))

    def test_canny_detects_boundary(self):
        image = np.zeros((50, 50), dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (39, 39), 255, -1)
        _low, _high, edges = canny_edges(image, 50, 150)
        self.assertGreater(int(np.count_nonzero(edges)), 80)

    def test_morphology_and_components(self):
        binary = np.zeros((40, 40), dtype=np.uint8)
        binary[5:15, 5:15] = 255
        binary[25:35, 25:35] = 255
        binary[1, 1] = 255
        cleaned = morphology_open_close(binary, 3, 3)
        filtered, _labels, components = connected_components_analysis(cleaned, min_area=20)
        self.assertEqual(len(components), 2)
        self.assertEqual(int(filtered[1, 1]), 0)

    def test_full_pipeline(self):
        image = np.full((80, 80), 180, dtype=np.uint8)
        cv2.circle(image, (40, 40), 14, 25, -1)
        result = process_roi(image, dark_foreground=True, min_component_area=30)
        self.assertTrue({"gray", "clahe", "otsu", "canny", "morphology", "components"} <= result.keys())
        self.assertGreater(len(result["components"]), 0)


if __name__ == "__main__":
    unittest.main()
