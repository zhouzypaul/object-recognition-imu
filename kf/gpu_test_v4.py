from timeit import default_timer as timer
from filterpy.kalman import KalmanFilter
import numpy as np


def one_iter():
    num_max = 1  # the number of maximum objects assumed to be detected in each frame
    num_obj = 80  # the total number of classes of objects that can be recognized by the recognition alg
    iC = 0  # relative index of c, the confidence score of the object recognized, taking value in [0, 1]
    iX = 1  # relative index of x, the x-position of the center of the bounding box
    iY = 2  # relative index of y, the y-position of the center of the bounding box
    iW = 3  # relative index of w, the width of the bounding box
    iH = 4  # relative index of h, the height of the bounding box
    dim = num_max * num_obj * (iH + 1)  # the dimension of the state vector
    f = KalmanFilter(dim_x=dim, dim_z=dim, dim_u=2)
    initial_state = np.zeros((dim, 1))  # TODO: change this
    f.x = initial_state
    f.F = np.eye(dim)  # state transition matrix
    f.H = np.eye(dim)  # the measurement function
    B = np.zeros((dim, 2))
    for i in range(num_obj * num_max):
        start_index = i * (iH + 1)
        x_index = start_index + iX
        y_index = start_index + iY
        B[x_index][0] = 1
        B[y_index][1] = 1
    f.B = B  # control transition matrix
    f.predict(u=np.array([[2], [3]]))
    obs = np.zeros(400)
    obs[275] = 0.88
    obs[276] = 200
    obs[277] = 300
    obs[278] = 63
    obs[279] = 27
    f.update(z=obs)


def two_iter():
    num_max = 2  # the number of maximum objects assumed to be detected in each frame
    num_obj = 80  # the total number of classes of objects that can be recognized by the recognition alg
    iC = 0  # relative index of c, the confidence score of the object recognized, taking value in [0, 1]
    iX = 1  # relative index of x, the x-position of the center of the bounding box
    iY = 2  # relative index of y, the y-position of the center of the bounding box
    iW = 3  # relative index of w, the width of the bounding box
    iH = 4  # relative index of h, the height of the bounding box
    dim = num_max * num_obj * (iH + 1)  # the dimension of the state vector
    f = KalmanFilter(dim_x=dim, dim_z=dim, dim_u=2)
    initial_state = np.zeros((dim, 1))  # TODO: change this
    f.x = initial_state
    f.F = np.eye(dim)  # state transition matrix
    f.H = np.eye(dim)  # the measurement function
    B = np.zeros((dim, 2))
    for i in range(num_obj * num_max):
        start_index = i * (iH + 1)
        x_index = start_index + iX
        y_index = start_index + iY
        B[x_index][0] = 1
        B[y_index][1] = 1
    f.B = B  # control transition matrix
    f.predict(u=np.array([[2], [3]]))
    obs = np.zeros(800)
    obs[275] = 0.88
    obs[276] = 200
    obs[277] = 300
    obs[278] = 63
    obs[279] = 27
    f.update(z=obs)


def three_iter():
    num_max = 3  # the number of maximum objects assumed to be detected in each frame
    num_obj = 80  # the total number of classes of objects that can be recognized by the recognition alg
    iC = 0  # relative index of c, the confidence score of the object recognized, taking value in [0, 1]
    iX = 1  # relative index of x, the x-position of the center of the bounding box
    iY = 2  # relative index of y, the y-position of the center of the bounding box
    iW = 3  # relative index of w, the width of the bounding box
    iH = 4  # relative index of h, the height of the bounding box
    dim = num_max * num_obj * (iH + 1)  # the dimension of the state vector
    f = KalmanFilter(dim_x=dim, dim_z=dim, dim_u=2)
    initial_state = np.zeros((dim, 1))  # TODO: change this
    f.x = initial_state
    f.F = np.eye(dim)  # state transition matrix
    f.H = np.eye(dim)  # the measurement function
    B = np.zeros((dim, 2))
    for i in range(num_obj * num_max):
        start_index = i * (iH + 1)
        x_index = start_index + iX
        y_index = start_index + iY
        B[x_index][0] = 1
        B[y_index][1] = 1
    f.B = B  # control transition matrix
    f.predict(u=np.array([[2], [3]]))
    obs = np.zeros(1200)
    obs[275] = 0.88
    obs[276] = 200
    obs[277] = 300
    obs[278] = 63
    obs[279] = 27
    f.update(z=obs)


def four_iter():
    num_max = 4  # the number of maximum objects assumed to be detected in each frame
    num_obj = 80  # the total number of classes of objects that can be recognized by the recognition alg
    iC = 0  # relative index of c, the confidence score of the object recognized, taking value in [0, 1]
    iX = 1  # relative index of x, the x-position of the center of the bounding box
    iY = 2  # relative index of y, the y-position of the center of the bounding box
    iW = 3  # relative index of w, the width of the bounding box
    iH = 4  # relative index of h, the height of the bounding box
    dim = num_max * num_obj * (iH + 1)  # the dimension of the state vector
    f = KalmanFilter(dim_x=dim, dim_z=dim, dim_u=2)
    initial_state = np.zeros((dim, 1))  # TODO: change this
    f.x = initial_state
    f.F = np.eye(dim)  # state transition matrix
    f.H = np.eye(dim)  # the measurement function
    B = np.zeros((dim, 2))
    for i in range(num_obj * num_max):
        start_index = i * (iH + 1)
        x_index = start_index + iX
        y_index = start_index + iY
        B[x_index][0] = 1
        B[y_index][1] = 1
    f.B = B  # control transition matrix
    f.predict(u=np.array([[2], [3]]))
    obs = np.zeros(dim)
    obs[275] = 0.88
    obs[276] = 200
    obs[277] = 300
    obs[278] = 63
    obs[279] = 27
    f.update(z=obs)


def five_iter():
    num_max = 5  # the number of maximum objects assumed to be detected in each frame
    num_obj = 80  # the total number of classes of objects that can be recognized by the recognition alg
    iC = 0  # relative index of c, the confidence score of the object recognized, taking value in [0, 1]
    iX = 1  # relative index of x, the x-position of the center of the bounding box
    iY = 2  # relative index of y, the y-position of the center of the bounding box
    iW = 3  # relative index of w, the width of the bounding box
    iH = 4  # relative index of h, the height of the bounding box
    dim = num_max * num_obj * (iH + 1)  # the dimension of the state vector
    f = KalmanFilter(dim_x=dim, dim_z=dim, dim_u=2)
    initial_state = np.zeros((dim, 1))  # TODO: change this
    f.x = initial_state
    f.F = np.eye(dim)  # state transition matrix
    f.H = np.eye(dim)  # the measurement function
    B = np.zeros((dim, 2))
    for i in range(num_obj * num_max):
        start_index = i * (iH + 1)
        x_index = start_index + iX
        y_index = start_index + iY
        B[x_index][0] = 1
        B[y_index][1] = 1
    f.B = B  # control transition matrix
    f.predict(u=np.array([[2], [3]]))
    obs = np.zeros(dim)
    obs[275] = 0.88
    obs[276] = 200
    obs[277] = 300
    obs[278] = 63
    obs[279] = 27
    f.update(z=obs)


def twenty_iter():
    num_max = 20  # the number of maximum objects assumed to be detected in each frame
    num_obj = 80  # the total number of classes of objects that can be recognized by the recognition alg
    iC = 0  # relative index of c, the confidence score of the object recognized, taking value in [0, 1]
    iX = 1  # relative index of x, the x-position of the center of the bounding box
    iY = 2  # relative index of y, the y-position of the center of the bounding box
    iW = 3  # relative index of w, the width of the bounding box
    iH = 4  # relative index of h, the height of the bounding box
    dim = num_max * num_obj * (iH + 1)  # the dimension of the state vector
    f = KalmanFilter(dim_x=dim, dim_z=dim, dim_u=2)
    initial_state = np.zeros((dim, 1))  # TODO: change this
    f.x = initial_state
    f.F = np.eye(dim)  # state transition matrix
    f.H = np.eye(dim)  # the measurement function
    B = np.zeros((dim, 2))
    for i in range(num_obj * num_max):
        start_index = i * (iH + 1)
        x_index = start_index + iX
        y_index = start_index + iY
        B[x_index][0] = 1
        B[y_index][1] = 1
    f.B = B  # control transition matrix
    f.predict(u=np.array([[2], [3]]))
    obs = np.zeros(dim)
    obs[275] = 0.88
    obs[276] = 200
    obs[277] = 300
    obs[278] = 63
    obs[279] = 27
    f.update(z=obs)

if __name__ == "__main__":
    start = timer()
    one_iter()
    print("400 matrix: ", timer() - start)

    start = timer()
    two_iter()
    print("800 matrix: ", timer() - start)

    start = timer()
    three_iter()
    print("1200 matrix: ", timer() - start)

    start = timer()
    four_iter()
    print("1600 matrix: ", timer() - start)

    start = timer()
    five_iter()
    print("2000 matrix: ", timer() - start)

    # start = timer()
    # twenty_iter()
    # print("8000 matrix: ", timer() - start)
