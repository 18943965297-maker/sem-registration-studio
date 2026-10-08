import unittest
import cv2
import numpy as np
from mask_editor_app import sam_context_bounds, display_contour_points


class ContextTests(unittest.TestCase):
    def test_fine_context_and_border(self):
        self.assertEqual(sam_context_bounds((200,200), (761,761), (185,185,215,215), minimum=64),
                         (168,168,232,232))
        self.assertEqual(sam_context_bounds((2,2), (761,761), (0,0,20,20), minimum=64),
                         (0,0,64,64))
        # Fine mode still keeps 1.4x context when the visible field is larger.
        self.assertEqual(sam_context_bounds((200,200), (761,761), (150,150,250,250), minimum=64),
                         (130,130,270,270))

    def test_tiny_view_keeps_context(self):
        self.assertEqual(sam_context_bounds((200, 200), (761, 761), (185, 185, 215, 215)),
                         (72, 72, 328, 328))

    def test_context_clips_at_edge(self):
        self.assertEqual(sam_context_bounds((2, 2), (100, 100), (0, 0, 20, 20)),
                         (0, 0, 100, 100))

    def test_high_zoom_preserves_vertices(self):
        mask = np.zeros((30, 30), np.uint8)
        cv2.circle(mask, (15, 15), 7, 1, -1)
        contour = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        np.testing.assert_array_equal(display_contour_points(contour, 12), contour.reshape(-1, 2))
