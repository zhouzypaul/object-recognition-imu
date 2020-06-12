from kf_v2 import KF
from numba import jit
import numpy as np
# to measure exec time
from timeit import default_timer as timer


# normal function to run on cpu
# def func():
#     kf = KF()
#     kf.predict(delta_x=0.5, delta_y=-0.1)
#     kf.update(meas_value=np.ones(8000))


# function optimized to run on gpu
@jit
def func2():
    kf = KF()
    kf.predict(delta_x=0.5, delta_y=-0.1)
    kf.update(meas_value=np.ones(8000))


if __name__ == "__main__":

    # start = timer()
    # func()
    # print("without GPU:", timer() - start)

    start = timer()
    func2()
    print("with GPU:", timer() - start)
