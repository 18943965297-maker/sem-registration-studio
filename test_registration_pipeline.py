import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, TiffImagePlugin

from registration_pipeline import (
    crop_sem_content,
    infer_magnification,
    largest_valid_rectangle,
    read_sem_metadata,
    read_image_unicode,
    register_arrays,
)


class RegistrationPipelineTests(unittest.TestCase):
    def test_infer_magnification(self):
        self.assertEqual(infer_magnification("sample_500x_before.tif"), 500.0)
        self.assertEqual(infer_magnification("SEM-2kx-after.png"), 2000.0)
        self.assertEqual(infer_magnification("倍率2000倍"), 2000.0)
        self.assertIsNone(infer_magnification("unknown.png"))

    def test_crop_portrait_sem_footer(self):
        image = np.zeros((140, 100), np.uint8)
        cropped, bounds = crop_sem_content(image)
        self.assertEqual(cropped.shape, (100, 100))
        self.assertEqual(bounds, (0, 0, 100, 100))

    def test_reads_tescan_header_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "2.tif"
            image_path.touch()
            (Path(folder) / "2-tif.hdr").write_text(
                "[MAIN]\nMagnification=2000.011\nPixelSizeX=0.000000136327369\nPixelSizeY=0.000000136327369\n",
                encoding="utf-8",
            )
            metadata = read_sem_metadata(image_path)
        self.assertAlmostEqual(metadata["magnification"], 2000.011)
        self.assertAlmostEqual(metadata["pixel_size_x_um"], 0.136327369)

    def test_reads_tescan_metadata_embedded_in_tiff(self):
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "embedded.tif"
            tiff_info = TiffImagePlugin.ImageFileDirectory_v2()
            tiff_info[50431] = (
                b"\x00Magnification=500.125\n"
                b"PixelSizeX=0.000000545312824\n"
                b"PixelSizeY=0.000000545312824\n\x00"
            )
            Image.fromarray(np.zeros((32, 32), np.uint16)).save(image_path, tiffinfo=tiff_info)
            metadata = read_sem_metadata(image_path)
        self.assertEqual(metadata["metadata_source"], "TIFF内嵌")
        self.assertAlmostEqual(metadata["magnification"], 500.125)
        self.assertAlmostEqual(metadata["pixel_size_x_um"], 0.545312824)

    def test_reader_preserves_uint16_sem_tiff_precision(self):
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "sem16.tif"
            source = np.array([[0, 257], [4095, 65535]], dtype=np.uint16)
            Image.fromarray(source).save(image_path)
            loaded = read_image_unicode(image_path)
        self.assertEqual(loaded.dtype, np.uint16)
        np.testing.assert_array_equal(loaded, source)

    def test_largest_valid_rectangle(self):
        mask = np.zeros((8, 10), bool)
        mask[2:7, 3:9] = True
        self.assertEqual(largest_valid_rectangle(mask), (3, 2, 9, 7))

    def test_translation_registration_improves_alignment(self):
        rng = np.random.default_rng(7)
        before = rng.normal(110, 24, (384, 384)).clip(0, 255).astype(np.uint8)
        for _ in range(80):
            x, y = rng.integers(15, 369, size=2)
            radius = int(rng.integers(2, 10))
            color = int(rng.integers(20, 235))
            cv2.circle(before, (int(x), int(y)), radius, color, -1)
        after = cv2.warpAffine(
            before, np.float32([[1, 0, 9], [0, 1, -6]]), (384, 384),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
        result = register_arrays(before, after, 500, 500, use_ecc=True)
        self.assertGreater(result.metrics.inliers, 20)
        self.assertLess(result.metrics.reprojection_median_px, 1.0)
        self.assertGreater(result.metrics.ncc_after, 0.90)
        self.assertGreater(result.before_core.size, 100_000)

    def test_physical_pixel_size_has_priority_for_scaling(self):
        rng = np.random.default_rng(17)
        coarse = rng.integers(0, 256, (192, 192), dtype=np.uint8)
        fine = cv2.resize(coarse, (768, 768), interpolation=cv2.INTER_CUBIC)
        result = register_arrays(
            coarse, fine, 500, 2000,
            before_pixel_size_um=0.545312824,
            after_pixel_size_um=0.136327369,
            use_ecc=False,
        )
        self.assertEqual(result.metrics.scale_source, "PixelSizeX")
        self.assertAlmostEqual(result.metrics.after_scale_factor, 0.25, places=3)
        self.assertAlmostEqual(result.metrics.target_pixel_size_um, 0.545312824)

    def test_cross_magnification_uses_coarser_grid(self):
        rng = np.random.default_rng(11)
        coarse = rng.normal(120, 28, (256, 256)).clip(0, 255).astype(np.uint8)
        for _ in range(50):
            x, y = rng.integers(10, 246, size=2)
            cv2.circle(coarse, (int(x), int(y)), int(rng.integers(2, 7)), int(rng.integers(10, 245)), -1)
        high_mag_before = cv2.resize(coarse, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        result = register_arrays(high_mag_before, coarse, 2000, 500, use_ecc=False)
        self.assertEqual(result.before_crop.shape, (256, 256))
        self.assertEqual(result.metrics.target_magnification, 500.0)
        self.assertEqual(result.metrics.before_scale_factor, 0.25)
        self.assertEqual(result.metrics.after_scale_factor, 1.0)
        self.assertGreater(result.metrics.ncc_after, 0.90)


if __name__ == "__main__":
    unittest.main()
