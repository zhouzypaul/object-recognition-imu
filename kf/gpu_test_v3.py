from filterpy.kalman import KalmanFilter
import numpy as np
from numba import njit, jit, cuda
from timeit import default_timer as timer


# parameters used:
num_max = 10  # the number of maximum objects assumed to be detected in each frame
num_obj = 80  # the total number of classes of objects that can be recognized by the recognition alg
iX = 0  # relative index of x, the x-position of the center of the bounding box
iY = 1  # relative index of y, the y-position of the center of the bounding box
iW = 2  # relative index of w, the width of the bounding box
iH = 3  # relative index of h, the height of the bounding box
dim = num_max * (num_obj + (iH + 1))  # the dimension of the state vector


# run on the cpu
def cpu():
    f = KalmanFilter(dim_x=dim, dim_z=dim, dim_u=2)
    initial_state = np.zeros((dim, 1))  # TODO: change this
    f.x = initial_state
    f.F = np.eye(dim)  # state transition matrix
    f.H = np.eye(dim)  # the measurement function
    B = np.zeros((dim, 2))
    for i in range(num_max):
        start_index = i * (num_obj + (iH + 1)) + num_obj
        x_index = start_index + iX
        y_index = start_index + iY
        B[x_index][0] = 1
        B[y_index][1] = 1
    f.B = B  # control transition matrix
    f.predict(u=np.array([[2], [3]]))
    obs = np.zeros(840)
    obs[49] = 0.88
    obs[80] = 200
    obs[81] = 300
    obs[82] = 63
    obs[83] = 27
    f.update(z=obs)


def cpu_smaller():
    f = KalmanFilter(dim_x=400, dim_z=400, dim_u=2)
    initial_state = np.zeros((400, 1))  # TODO: change this
    f.x = initial_state
    f.F = np.eye(400)  # state transition matrix
    f.H = np.eye(400)  # the measurement function
    B = np.zeros((400, 2))
    for i in range(80):
        start_index = i
        x_index = start_index + iX
        y_index = start_index + iY
        B[x_index][0] = 1
        B[y_index][1] = 1
    f.B = B  # control transition matrix
    f.predict(u=np.array([[2], [3]]))
    obs = np.zeros(400)
    obs[49] = 0.88
    obs[80] = 200
    obs[81] = 300
    obs[82] = 63
    obs[83] = 27
    f.update(z=obs)


@jit(target="cuda")
def gpu():
    f = KalmanFilter(dim_x=dim, dim_z=dim, dim_u=2)
    initial_state = np.zeros((dim, 1))  # TODO: change this
    f.x = initial_state
    f.F = np.eye(dim)  # state transition matrix
    f.H = np.eye(dim)  # the measurement function
    B = np.zeros((dim, 2))
    for i in range(num_max):
        start_index = i * (num_obj + (iH + 1)) + num_obj
        x_index = start_index + iX
        y_index = start_index + iY
        B[x_index][0] = 1
        B[y_index][1] = 1
    f.B = B  # control transition matrix
    f.predict(u=np.array([[2], [3]]))
    obs = np.zeros(840)
    obs[49] = 0.88
    obs[80] = 200
    obs[81] = 300
    obs[82] = 63
    obs[83] = 27
    f.update(z=obs)


if __name__ == "__main__":
    start = timer()
    cpu()
    print("on cpu: ", timer() - start)

    start = timer()
    cpu_smaller()
    print("on cpu: smaller model", timer() - start)

    # start = timer()
    # gpu()
    # print("on gpu: ", timer() - start)
