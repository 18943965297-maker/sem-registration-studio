import unittest
import numpy as np
import cv2
from registration_pipeline import register_arrays, RegistrationError


class RegistrationModesTests(unittest.TestCase):
    def data(self):
        a=np.random.default_rng(8).integers(0,255,(256,256),dtype=np.uint8)
        b=cv2.warpAffine(a,np.float32([[1,0,7],[0,1,4]]),(256,256))
        fit=[[30,30],[120,30],[220,30],[30,210],[120,210],[220,210]]
        check=[[70,80],[180,90],[100,170]]
        return a,b,{key:[[p,[p[0]+7,p[1]+4]] for p in pts] for key,pts in [('fit',fit),('check',check)]}

    def test_manual_transform_and_independent_checks(self):
        a,b,p=self.data()
        r=register_arrays(a,b,500,500,matcher='manual',manual_points=p,use_ecc=False)
        self.assertEqual(r.metrics.quality,'良好')
        self.assertLess(r.manual_validation['check_p95_px'],.01)
        np.testing.assert_allclose(r.transform_after_to_before[:2,2],[-7,-4],atol=.01)

    def test_wrong_checks_do_not_change_fit_and_fail_gate(self):
        a,b,p=self.data()
        for point in p['check']:
            point[1][0]+=20
        r=register_arrays(a,b,500,500,matcher='manual',manual_points=p,use_ecc=False)
        self.assertEqual(r.metrics.quality,'需复核')
        np.testing.assert_allclose(r.transform_after_to_before[:2,2],[-7,-4],atol=.01)

    def test_missing_points_never_falls_back_to_phase(self):
        a,b,p=self.data()
        p['check']=[]
        with self.assertRaises(RegistrationError):
            register_arrays(a,b,500,500,matcher='manual',manual_points=p)

    def test_manual_points_respect_physical_scale(self):
        a,b,p=self.data()
        a=cv2.resize(a,(512,512),interpolation=cv2.INTER_NEAREST)
        for group in p.values():
            for pair in group:
                pair[0]=[x*2 for x in pair[0]]
        r=register_arrays(a,b,2000,500,before_pixel_size_um=.5,after_pixel_size_um=1,
                          matcher='manual',manual_points=p,use_ecc=False)
        self.assertLess(r.manual_validation['check_p95_px'],.01)
