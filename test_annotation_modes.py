import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
import numpy as np
from PIL import Image
from annotation_modes import AnnotationLayers, MODES, pore_colors
from mask_editor_app import MaskEditorApp


class AnnotationTests(unittest.TestCase):
    def test_lossless_labels_and_independent_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            a = AnnotationLayers((8, 8), folder)
            a.layers['before']['mask'][1, 1] = True
            a.layers['multi']['mask'][2, 2] = 255
            a.layers['multi']['mask'][3, 3] = 3
            a.used.update(('before', 'multi'))
            a.save()
            b = AnnotationLayers((8, 8), folder)
            self.assertFalse(b.layers['after']['mask'].any())
            self.assertEqual(b.layers['multi']['mask'][2, 2], 255)
            self.assertEqual(b.layers['multi']['mask'][3, 3], 3)
            self.assertEqual(set(np.unique(Image.open(Path(folder)/'pore_masks/before_pore_mask.png'))), {0, 1})
            with self.assertRaises(ValueError):
                AnnotationLayers((9, 9), folder)

    def test_overlap_is_distinct(self):
        a = np.array([[True, False, True]])
        b = np.array([[False, True, True]])
        self.assertEqual(pore_colors(a,b).tolist(), [[[40,120,255],[255,60,60],[190,60,230]]])

    def test_app_sam_routing_undo_modes_and_save_reload(self):
        with tempfile.TemporaryDirectory() as folder, patch('mask_editor_app.messagebox.showinfo'), patch.object(MaskEditorApp, 'check_sam_service'):
            root = tk.Tk()
            root.withdraw()
            try:
                app = MaskEditorApp(root)
                base = Path(folder)
                for name in ('before', 'after', 'mask'):
                    Image.fromarray(np.zeros((32,32), np.uint8)).save(base/f'{name}.png')
                row = {k: str(base/f'{k}.png') for k in ('before','after','mask')}
                row.update(pair_id='test', annotation_dir=folder)
                app.rows, app.idx = [row], 0
                app.load_current()
                candidate = np.zeros((32,32), bool)
                candidate[5:8,5:8] = True
                for side in ('before', 'after'):
                    app.sam_candidates[side] = candidate.copy()
                    with patch.object(app, '_selected_sam_side', return_value=side):
                        app.accept_sam_candidate()
                    self.assertEqual(app.active_annotation, side)
                app.undo()
                self.assertFalse(app.mask.any())
                self.assertTrue(app.annotations.layers['before']['mask'].any())
                app.redo()
                app.annotation_mode.set(MODES[2])
                app.change_annotation_mode()
                app.annotation_class.set('255 不确定/忽略')
                app.sam_candidates['after'] = candidate.copy()
                with patch.object(app, '_selected_sam_side', return_value='after'):
                    app.accept_sam_candidate()
                self.assertEqual(app.mask[5,5],255)
                app.annotation_mode.set(MODES[1])
                app.change_annotation_mode()
                self.assertFalse(app.mask.any())
                app.mask[5,5] = True
                app.refresh_images()
                for pane in (app.before_pane, app.after_pane, app.overlay_pane):
                    red, green, blue = pane.base_pil.getpixel((5,5))
                    self.assertGreater(red, green)
                    self.assertGreater(green, blue)
                    self.assertEqual(pane.base_pil.getpixel((0,0)), (0,0,0))
                self.assertEqual(app.mask_pane.base_pil.getpixel((5,5)), (255,255,255))
                app.save_mask()
                with Image.open(base/'change_binary/mask.png') as saved:
                    self.assertEqual(set(np.unique(saved)), {0,1})
                self.assertTrue((base/'annotation_project.json').exists())
                app.load_current()
                self.assertEqual(app.annotations.layers['multi']['mask'][5,5],255)
                self.assertTrue(app.annotations.layers['after']['mask'][5,5])
            finally:
                root.destroy()


if __name__ == '__main__':
    unittest.main()
