from .kf_v4 import f

import unittest
import numpy as np
from timeit import default_timer as timer


class TestKFV4(unittest.TestCase):

    def test_compile_correctly(self):
        self.assertEqual(f.x.size, 800)
        self.assertEqual(f.P.size, 800 * 800)
        self.assertEqual(f.x[5], 0)
        # self.assertEqual(f.P[1][1], 1)
        # self.assertEqual(f.P[1][3], 0)

    def test_able_to_predict(self):
        f.predict(u=np.array([[2], [1]]))
        # self.assertEqual(f.x[1], 2)
        # self.assertEqual(f.x[2], 1)
        # self.assertEqual(f.x[6], 2)
        # self.assertEqual(f.x[7], 1)
        # self.assertEqual(f.x[796], 2)
        # self.assertEqual(f.x[797], 1)

    def test_able_to_update(self):
        f.update(z=np.zeros(800))

    def test_x_P_same_shape_after_predict_and_update(self):
        f.predict(u=np.array([[2], [1]]))
        f.update(z=np.zeros(800))
        self.assertEqual(f.x.size, 800)
        self.assertEqual(f.P.size, 800 * 800)

    def test_predict_increase_state_uncertainty(self):
        for i in range(2):
            det_before = np.linalg.det(f.P)
            f.predict(u=np.array([[3 + i/10], [-2]]))
            det_after = np.linalg.det(f.P)
            print(det_before, det_after)
            self.assertLess(det_before, det_after)

    def test_update_lower_state_uncertainty(self):
        for i in range(3):
            det_before = np.linalg.det(f.P)
            obs = np.zeros(800)
            obs[275] = 0.88
            obs[276] = 200
            obs[277] = 300
            obs[278] = 63
            obs[279] = 27
            f.update(z=obs)
            det_after = np.linalg.det(f.P)
            self.assertGreater(det_before, det_after)
