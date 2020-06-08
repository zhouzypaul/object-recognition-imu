from .kf_v2 import KF

import unittest
import numpy as np
from numba import jit, cuda, njit


class TestKFV2(unittest.TestCase):

    @njit(target='cuda', nopython=False)
    def test_able_to_compile_correctly(self):
        kf = KF()
        kf2 = KF(init_state=np.ones((8000, 1)), state_var=10)
        self.assertEqual(kf.cov[1, 1], 1)
        self.assertEqual(kf.cov[2, 2], 1)
        self.assertEqual(kf2.cov[1, 1], 10)
        self.assertEqual(kf.mean[4, 0], 0)
        self.assertEqual(kf2.mean[39, 0], 1)

    # @njit(fastmath=True, target='cuda', nopython=True)
    # def test_able_to_predict(self):
    #     kf = KF()
    #     kf.predict(delta_x=5, delta_y=-3)
    #
    # @njit(fastmath=True, target='cuda', nopython=True)
    # def test_after_calling_predict_and_update_mean_and_cov_are_of_right_shape(self):
    #     kf = KF()
    #     kf.predict(delta_x=5, delta_y=-3)
    #     self.assertEqual(kf.cov.shape, (8000, 8000))
    #     self.assertEqual(kf.mean.shape, (8000, 1))
    #     kf.update(meas_value=np.ones((8000, 1)))
    #     self.assertEqual(kf.cov.shape, (8000, 8000))
    #     self.assertEqual(kf.mean.shape, (8000, 1))
    #
    # @njit(fastmath=True, target='cuda', nopython=True)
    # def test_predict_increase_state_uncertainty(self):
    #     kf = KF(state_var=0.01)
    #     kf.predict(delta_x=0.5, delta_y=-0.3)
    #
    #     for i in range(10):
    #         det_before = np.linalg.det(kf.cov)
    #         kf.predict(delta_x=0.1, delta_y=-0.1, process_noise_var=0.1)
    #         det_after = np.linalg.det(kf.cov)
    #
    #         self.assertGreater(det_after, det_before)
    #         # print(det_before, det_after)
    #
    # @njit(fastmath=True, target='cuda', nopython=True)
    # def test_able_to_update(self):
    #     kf = KF()
    #     kf.predict(delta_x=0.5, delta_y=-0.3)
    #     kf.update(meas_value=np.ones((8000, 1)))
    #
    # @njit(fastmath=True, target='cuda', nopython=True)
    # def test_update_decrease_state_uncertainty(self):
    #     kf = KF(state_var=1)
    #     kf.predict(delta_x=0.5, delta_y=-0.1, process_noise_var=0.1)
    #
    #     for i in range(10):
    #         det_before = np.linalg.det(kf.cov)
    #         kf.update(meas_value=np.ones((8000, 1)), meas_variance=0.1)
    #         det_after = np.linalg.det(kf.cov)
    #         kf.predict(delta_x=0.05, delta_y=-0.03, process_noise_var=0.1)
    #
    #         self.assertLess(det_after, det_before)
    #         print(det_before, det_after)
