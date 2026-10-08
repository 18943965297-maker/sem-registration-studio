"""Regression checks for sparse mouse events and viewport-sized rendering."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from mask_editor_app import MaskEditorApp, ImagePane


class Value:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value


class InteractionTests(unittest.TestCase):
    def brush(self, diameter=1):
        return SimpleNamespace(mask=np.zeros((80, 80), bool), brush_size=Value(diameter),
            brush_kind=Value('normal'), mode=Value('add'), last_brush_point=None,
            schedule_mask_refresh=lambda: None)

    def test_sparse_events_form_connected_stroke(self):
        app = self.brush()
        for x in (10, 20, 30, 40, 50):
            MaskEditorApp.paint(app, x, 30)
        self.assertEqual(int(app.mask.sum()), 41)
        self.assertTrue(app.mask[30, 10:51].all())
        app.mode.value = 'erase'
        app.last_brush_point = None
        for x in (10, 50):
            MaskEditorApp.paint(app, x, 30)
        self.assertFalse(app.mask.any())

    def test_separate_strokes_do_not_connect(self):
        app = self.brush()
        MaskEditorApp.paint(app, 10, 10)
        app.last_brush_point = None
        MaskEditorApp.paint(app, 50, 50)
        self.assertEqual(int(app.mask.sum()), 2)

    def test_even_brush_diameter_is_not_rounded_to_one(self):
        app = self.brush(2)
        MaskEditorApp.paint(app, 30, 30)
        ys, xs = np.where(app.mask)
        self.assertEqual(int(xs.max()-xs.min()+1), 2)
        self.assertEqual(int(ys.max()-ys.min()+1), 2)

    def test_viewport_allocation_is_bounded_at_high_zoom(self):
        from unittest.mock import MagicMock
        canvas = MagicMock()
        canvas.winfo_width.return_value = 480
        canvas.winfo_height.return_value = 360
        pane = ImagePane.__new__(ImagePane)
        pane.canvas = canvas
        pane.base_pil = Image.fromarray(np.zeros((1024, 1024, 3), np.uint8))
        pane.image_id = None
        pane.cross_items = []
        pane.sam_overlay = None
        for zoom in (1, 4, 12, 24, 64):
            pane.view_getter = lambda: (zoom, 512, 512)
            with patch('mask_editor_app.ImageTk.PhotoImage', side_effect=lambda im: im):
                pane.refresh()
            self.assertEqual(pane.photo.size, (480, 360))
            self.assertEqual(pane.canvas_to_image(240, 180), (512, 512))

    def test_cancel_invalidates_predictions_and_hover(self):
        app = SimpleNamespace(sam_generation=4, sam_pending_side='before',
            sam_hover_side='before', sam_hover_points={'before': (4, 5)},
            sam_hover_job=None)
        MaskEditorApp.invalidate_sam_requests(app)
        self.assertEqual(app.sam_generation, 5)
        self.assertIsNone(app.sam_pending_side)
        self.assertTrue(all(app.sam_restart_required.values()))


if __name__ == '__main__':
    unittest.main()
