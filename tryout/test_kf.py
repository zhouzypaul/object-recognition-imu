from .kalman_filter import KF

import unittest
import numpy as np


class TestKF(unittest.TestCase):
    def test_dimension(self):
        x = 0.2
        v = 0.3
        a = 0.1

        kf = KF(initial_x=x, initial_v=v, accel_variance=a)
        self.assertAlmostEqual(kf.mean.size, 2)
        self.assertAlmostEqual(kf.cov.size, 4)

    def test_after_calling_predict_mean_and_cov_are_of_right_shape(self):
        x = 0.2
        v = 2.3

        kf = KF(x, v, 1.2)
        kf.predict(0.1)

        self.assertEqual(kf.cov.shape, (2, 2))
        self.assertEqual(kf.mean.shape, (2, ))

    def test_calling_predict_increases_state_uncertainty(self):
        x = 0.2
        v = 2.3

        kf = KF(x, v, 1.2)

        for i in range(10):
            det_before = np.linalg.det(kf.cov)
            kf.predict(0.1)
            det_after = np.linalg.det(kf.cov)

            self.assertGreater(det_after, det_before)
            print(det_before, det_after)

    def test_calling_update_decreases_state_uncertainty(self):
        x = 0.2
        v = 2.3

        kf = KF(x, v, 1.2)

        det_before = np.linalg.det(kf.cov)
        kf.update(0.1, 0.01)
        det_after = np.linalg.det(kf.cov)

        self.assertLess(det_after, det_before)


class TestKFPythonAndCppYieldSameResults(unittest.TestCase):
    def test_yield_same_results_with_predict(self):
        x = 0.2
        v = 2.3

        kf_py = KF(x, v, 1.2)
        kf_cpp = KF(x, v, 1.2)

        for i in range(10):
            kf_py.predict(0.1)
            kf_cpp.predict(0.1)

            self.assertTrue(np.allclose(kf_py.cov, kf_cpp.cov, 1e-6))
            self.assertTrue(np.allclose(kf_py.mean, kf_cpp.mean, 1e-6))

    def test_yield_same_results_with_update(self):
        x = 0.2
        v = 2.3

        kf_py = KF(x, v, 1.2)
        kf_cpp = KF(x, v, 1.2)

        kf_cpp.update(0.1, 0.01)
        kf_py.update(0.1, 0.01)

        self.assertTrue(np.allclose(kf_py.cov, kf_cpp.cov, 1e-6))
        self.assertTrue(np.allclose(kf_py.mean, kf_cpp.mean, 1e-6))