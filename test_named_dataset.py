import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, TiffImagePlugin

from annotation_modes import AnnotationLayers
from build_named_sem_dataset import acquisition, content_bounds, names, specimen_key, matrices
from dataset_paths import portable_row, resolve_row
from registration_pipeline import register_arrays, save_registration_result, read_image_unicode


class NamedDatasetTests(unittest.TestCase):
    def test_footer_from_instrument_metadata_not_square_guess(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.tif'
            tags=TiffImagePlugin.ImageFileDirectory_v2()
            tags[50431]=b'\x00ImageStripSize=25\nMagnification=500\nPixelSizeX=1e-6\nPixelSizeY=1e-6\n'
            Image.fromarray(np.zeros((180,200),np.uint16)).save(p,tiffinfo=tags)
            self.assertEqual(content_bounds(acquisition(p)),[0,0,200,155])
            info=acquisition(p);info['image_strip_values']=[]
            with self.assertRaises(ValueError):content_bounds(info)

    def test_failed_features_do_not_become_phase_evidence(self):
        a=np.random.default_rng(22).integers(0,255,(128,128),dtype=np.uint8)
        kp=[cv2.KeyPoint(float(x),float(x),1) for x in range(5)]
        matches=[cv2.DMatch(i,i,0) for i in range(5)]
        with patch('registration_pipeline._detect_and_match',return_value=('SIFT',kp,kp,matches)), patch('registration_pipeline._phase_translation',return_value=np.eye(3)):
            r=register_arrays(a,a,500,500,use_ecc=False)
        self.assertEqual(r.metrics.detector,'phase-correlation')
        self.assertEqual(r.metrics.inliers,0)
        self.assertTrue(np.isnan(r.metrics.reprojection_median_px))

    def test_ecc_rejects_geometric_drift_even_with_good_ncc(self):
        a=np.random.default_rng(9).integers(0,255,(256,256),dtype=np.uint8)
        with patch('registration_pipeline.cv2.findTransformECC',return_value=(.99,np.float32([[1,0,20],[0,1,0]]))), patch('registration_pipeline._ncc',return_value=.95):
            r=register_arrays(a,a,500,500)
        self.assertFalse(r.metrics.ecc_used)
        self.assertFalse(r.ecc_validation['geometry_ok'])
        self.assertLess(r.metrics.reprojection_median_px,.01)

    def test_named_files_keep_depth_and_checkerboard_uses_display_scale(self):
        a=np.random.default_rng(31).integers(0,255,(256,256),dtype=np.uint8)
        r=register_arrays(a.astype(np.uint16)*257,a,500,500,use_ecc=False)
        with tempfile.TemporaryDirectory() as d:
            files=save_registration_result(r,Path(d)/'out',filenames={'before':'CO2-30MPa_第2组_500倍_Before.png','after':'CO2-30MPa_第2组_500倍_After.png'})
            self.assertEqual(read_image_unicode(files['before']).dtype,np.uint16)
            self.assertEqual(read_image_unicode(files['after']).dtype,np.uint8)
            board=read_image_unicode(files['checkerboard'])
            # Identical intensities in mixed depths must not make alternating black tiles.
            self.assertLess(abs(float(board[:64,:64].mean())-float(board[:64,64:128].mean())),10)
            self.assertIn('500倍',Path(files['before']).name)

    def test_complete_coordinate_transform_roundtrip(self):
        a=np.random.default_rng(31).integers(0,255,(260,256),dtype=np.uint8)
        r=register_arrays(a,a,500,500,before_content_bounds=[0,0,256,240],after_content_bounds=[0,0,256,240],use_ecc=False)
        m=matrices(r)
        for side in ('before','after'):
            np.testing.assert_allclose(np.array(m[f'label_to_{side}_raw'])@np.array(m[f'{side}_raw_to_label']),np.eye(3),atol=1e-8)

    def test_names_keep_condition_group_scale_and_both_original_sequences(self):
        b={'condition':'2-CO2-30MPa-1m','group':'2','nominal_magnification':500,'subdirectory':'2-1',
           'stem':'3_1','original_relative_path':'2/2-1/3_1.tif','source_sha256':'b'}
        a=dict(b,stem='1',original_relative_path='2-CO2-30MPa-1m/2-1/1.tif',source_sha256='a')
        pair,uid,files=names(b,a)
        for key,filename in files.items():
            self.assertIn('CO2-30MPa-1m_第2组_500倍',filename)
            self.assertIn('前3_1_后1',filename)
        self.assertEqual(specimen_key('1','E1.1-5'),'1-5')
        self.assertEqual(specimen_key('3','3号-未浸泡A'),'A')

    def test_named_labels_and_portable_row_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            row={'before':str(p/'image.png'),'annotation_dir':str(p),'label_prefix':'CO2-30MPa_第2组_500倍'}
            self.assertEqual(resolve_row(portable_row(row,p),p),row)
            x=AnnotationLayers((12,12),p,filename_prefix=row['label_prefix'])
            x.layers['before']['mask'][1,2]=True;x.layers['multi']['mask'][2,3]=255
            x.used.update(('before','multi'));x.save()
            y=AnnotationLayers((12,12),p,filename_prefix=row['label_prefix'])
            self.assertTrue(y.layers['before']['mask'][1,2])
            self.assertEqual(y.layers['multi']['mask'][2,3],255)
            self.assertFalse((p/'pore_masks/before_pore_mask.png').exists())
            self.assertTrue(all('第2组' in q.name for q in p.rglob('*.png')))


if __name__=='__main__':unittest.main()
