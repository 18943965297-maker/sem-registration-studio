import unittest

import numpy as np

from mask_editor_app import geojson_to_mask, rasterize_sam_candidates, sam_roi_bounds, transform_sam_feature, ImagePane


class GeoJsonMaskTests(unittest.TestCase):
    def test_api_marching_squares_roundtrip(self):
        from skimage.measure import find_contours
        for start in (0, 9, 10, 11, 27):
            a = np.zeros((32,32), bool)
            a[start:start+5, start:start+5] = True
            ring = (find_contours(np.pad(a,1), .5)[0]-1)[:, ::-1].tolist()
            feature = {'geometry': {'type':'Polygon','coordinates':[ring]}}
            np.testing.assert_array_equal(geojson_to_mask([feature], a.shape), a)

    def test_hole_does_not_erase_other_feature(self):
        def square(low, high):
            return [[low,low],[high,low],[high,high],[low,high],[low,low]]
        solid = {'geometry':{'type':'Polygon','coordinates':[square(3.5,5.5)]}}
        donut = {'geometry':{'type':'Polygon','coordinates':[square(.5,8.5),square(2.5,6.5)]}}
        a = geojson_to_mask([solid,donut], (10,10))
        b = geojson_to_mask([donut,solid], (10,10))
        np.testing.assert_array_equal(a,b)
        self.assertTrue(a[4,4])
        self.assertFalse(a[3,3])

    def test_float_geometry_rendered_without_contour_roundtrip(self):
        from unittest.mock import MagicMock, patch
        from PIL import Image
        f = {'geometry':{'type':'Polygon','coordinates':[[[.5,.5],[2.5,.5],[2.5,2.5],[.5,.5]]]}}
        mapped = transform_sam_feature(f, .5, .5, (10,20))
        self.assertEqual(mapped['geometry']['coordinates'][0][0], [11.,21.])
        self.assertEqual(f['geometry']['coordinates'][0][0], [.5,.5])
        pane = ImagePane.__new__(ImagePane)
        pane.base_pil = Image.new('RGB',(32,32))
        pane.canvas = MagicMock()
        pane.offset_x = pane.offset_y = 0
        pane.scale = 10
        with patch('mask_editor_app.cv2.findContours', side_effect=AssertionError('must use vector geometry')):
            pane.set_sam_overlay(np.ones((32,32),bool), feature=f)
        self.assertEqual(pane.canvas.create_line.call_args.args[:4], (5.,5.,25.,5.))

    def test_polygon_and_hole(self):
        feature = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[1, 1], [8, 1], [8, 8], [1, 8], [1, 1]],
                    [[3, 3], [6, 3], [6, 6], [3, 6], [3, 3]],
                ],
            }
        }
        mask = geojson_to_mask([feature], (10, 10))
        self.assertTrue(mask[2, 2])
        self.assertFalse(mask[4, 4])

    def test_multipolygon(self):
        feature = {
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [[[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]],
                    [[[6, 6], [8, 6], [8, 8], [6, 8], [6, 6]]],
                ],
            }
        }
        mask = geojson_to_mask([feature], (10, 10))
        self.assertTrue(mask[2, 2])
        self.assertTrue(mask[7, 7])
        self.assertFalse(mask[5, 5])

    def test_local_roi_is_clipped_at_image_edge(self):
        self.assertEqual(sam_roi_bounds((5, 7), (300, 400), 128), (0, 0, 128, 128))
        self.assertEqual(sam_roi_bounds((399, 299), (300, 400), 128), (272, 172, 400, 300))

    def test_point_focused_candidate_avoids_near_global_mask(self):
        global_feature = {
            "properties": {"quality": 0.99},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [99, 0], [99, 99], [0, 99], [0, 0]]]},
        }
        local_feature = {
            "properties": {"quality": 0.91},
            "geometry": {"type": "Polygon", "coordinates": [[[35, 35], [65, 35], [65, 65], [35, 65], [35, 35]]]},
        }
        options = rasterize_sam_candidates(
            [global_feature, local_feature], (100, 100), [((50, 50), 1)], "点选小目标"
        )
        self.assertLess(options[0]["area"], 2000)


if __name__ == "__main__":
    unittest.main()
