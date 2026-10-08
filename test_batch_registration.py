import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cv2
from PIL import Image
from batch_registration import scan_dataset, run_batch, audit_arrays, audit_folder, validate_scales
from registration_pipeline import register_arrays


class BatchTests(unittest.TestCase):
    def test_confirmed_pair_outputs_separate_annotation_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'input'
            a=np.random.default_rng(5).integers(0,255,(256,256),dtype=np.uint8)
            for side in ('before','after'):
                (root/side).mkdir(parents=True)
                Image.fromarray(a).save(root/side/'sample.tif')
            scan=scan_dataset(root)
            scan['pairs'][0]['confirmed']=True
            for values in scan['inventory'].values():
                values[0]['magnification']=500
            folder,records=run_batch(scan,Path(tmp)/'out',False)
            self.assertEqual(records[0]['status'],'accepted')
            self.assertTrue((Path(records[0]['folder'])/'annotation_project.json').exists())
            self.assertEqual(len(list(root.rglob('*.tif'))),2)

    def test_scan_never_confirms_by_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for side in ('before','after'):
                (root/side).mkdir()
                Image.fromarray(np.zeros((128,128),np.uint8)).save(root/side/'sample.tif')
            scan=scan_dataset(root)
            self.assertTrue(scan['pairs'][0]['before'])
            self.assertFalse(scan['pairs'][0]['confirmed'])
            with patch('batch_registration.register_arrays') as run:
                folder,records=run_batch(scan,root/'output',False)
                run.assert_not_called()
            self.assertEqual(records[0]['status'],'skipped')
            self.assertTrue((folder/'batch_summary.json').exists())

    def test_ambiguous_stems_not_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for side in ('before','after'):(root/side).mkdir()
            for name in ('before/sample.tif','before/sample.png','after/sample.tif'):
                Image.fromarray(np.zeros((128,128),np.uint8)).save(root/name)
            self.assertEqual(scan_dataset(root)['pairs'][0]['before'],'')

    def test_unknown_and_cross_scale_rejected(self):
        with self.assertRaises(ValueError):validate_scales({}, {})
        with self.assertRaises(ValueError):validate_scales({'magnification':500},{'magnification':2000})
        validate_scales({'pixel_size_um':.4},{'pixel_size_um':.4})

    def test_audit_measures_shift_without_refitting(self):
        a=np.random.default_rng(20).integers(0,255,(384,384),dtype=np.uint8)
        b=cv2.warpAffine(a,np.float32([[1,0,8],[0,1,0]]),(384,384))
        with patch('cv2.estimateAffinePartial2D',side_effect=AssertionError('no refit')):
            r=audit_arrays(a,b)
        self.assertEqual(r['status'],'需复核')
        self.assertAlmostEqual(r['direct_displacement_median_px'],8,delta=.6)
        self.assertEqual(audit_arrays(a,a)['status'],'需复核')
        with self.assertRaises(ValueError):audit_arrays(a,b[:200])

    def test_already_cropped_portrait_preserved(self):
        a=np.random.default_rng(21).integers(0,255,(300,200),dtype=np.uint8)
        r=register_arrays(a,a,500,500,crop_footer=False,use_ecc=False)
        self.assertEqual(r.before_crop.shape,(300,200))
        self.assertEqual(r.before_crop_bounds_xyxy,(0,0,200,300))

    def test_audit_folder_does_not_edit_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'input';root.mkdir()
            a=np.random.default_rng(3).integers(0,255,(128,128),dtype=np.uint8)
            for name in ('before_core.png','after_registered_core.png'):Image.fromarray(a).save(root/name)
            originals={p.name:p.read_bytes() for p in root.iterdir()}
            folder,reports=audit_folder(root,Path(tmp)/'out')
            self.assertEqual(reports[0]['status'],'需复核')
            self.assertTrue((folder/'audit_summary.json').exists())
            self.assertEqual(originals,{p.name:p.read_bytes() for p in root.iterdir()})
