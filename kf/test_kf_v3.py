from kf_v3 import f

import unittest
import numpy as np
from timeit import default_timer as timer

class TestKFV3(unittest.TestCase):

    def test_compile_correctly(self):
        self.assertEqual(f.x.size, 840)
        self.assertEqual(f.P.size, 840 * 840)
        # self.assertEqual(f.x[49], 0)
        # self.assertEqual(f.P[1][1], 1)
        # self.assertEqual(f.P[1][3], 0)

    def test_able_to_predict(self):
        f.predict(u=np.array([[2], [1]]))

    def test_able_to_update(self):
        f.update(z=np.zeros(840))

    def test_x_P_same_shape_after_predict_and_update(self):
        f.predict(u=np.array([[2], [1]]))
        f.update(z=np.zeros(840))
        self.assertEqual(f.x.size, 840)
        self.assertEqual(f.P.size, 840 * 840)

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
            obs = np.zeros(840)
            obs[49] = 0.88
            obs[80] = 200
            obs[81] = 300
            obs[82] = 63
            obs[83] = 27
            f.update(z=obs)
            det_after = np.linalg.det(f.P)
            self.assertGreater(det_before, det_after)


if __name__ == "__main__":
    start = timer()
    f.predict(u=np.array([[2], [3]]))
    print("predict: ", timer() - start)

    start = timer()
    obs = np.zeros(840)
    obs[49] = 0.88
    obs[80] = 200
    obs[81] = 300
    obs[82] = 63
    obs[83] = 27
    f.update(z=obs)
    print("update: ", timer() - start)
