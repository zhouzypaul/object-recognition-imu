import numpy as np
from numba import jit


# offsets of each variable in the state vector
iC = 0  # the confidence level
iX = 1  # the x coordinate of the center of the bounding box
iY = 2  # the y coordinate of the center of the bounding box
iW = 3  # the width of the bounding box
iH = 4  # the height of the bounding box
NUMVARS = iH + 1  # the number of variables needed to describe one recognized object
NUMMAX = 20  # the maximum number of each objects assumed to be in a single frame
NUMOBJS = 80  # the number of different types of objects an algorithm can recognize
DIM = NUMOBJS * NUMVARS * NUMMAX

# the initial state of the state vector, all the class tags are initialized to
initial_state = np.zeros((DIM, 1))


class KF:
    """
    This version of KF assumes that each object can occur at most NUMMAX times in one frame
    the calculations of the Kalman Filter are as follows:

    prediction:
    x = A x + B u[n]
    P = A P At + Q

    compute the Kalman Gain
    S = H P Ht + R
    K = P Ht np.linalg.pinv(S)

    update the estimate via Z
    Z = m x[n]
    y = Z - H x
    x = x + K y

    update the error covariance
    P = (I - K H) P
    """
    @jit
    def __init__(self, init_state: np.array = np.zeros((DIM, 1)), state_var: float = 1) -> None:
        # mean of state GRV, all initialized to 0
        self._x = init_state

        # covariance of GRV, initialized to be identity
        self._P = np.eye(DIM)
        if state_var is not 1:
            for i in range(DIM):
                self._P[i, i] = state_var

    @jit
    def predict(self, delta_x: float, delta_y: float, process_noise_var: float = 1) -> None:
        # x = A x + B u[n]
        # P = A P At + Q     (Q = G Gt a)

        # A should be the identity matrix, so it can be occluded
        new_x = self._x
        for i in range(NUMOBJS * NUMMAX):
            x_index = NUMVARS * i + iX
            y_index = NUMVARS * i + iY
            new_x[x_index] += delta_x
            new_x[y_index] += delta_y  # we are assuming that head movement at from t-1 to t is the same at t to t+1
            # which means we are assuming the head movement to be smooth

        # the process noise covariance matrix
        Q = np.eye(DIM)
        if process_noise_var is not 1:
            for i in range(DIM):
                Q[i, i] = process_noise_var
        new_P = self._P + Q

        self._x = new_x
        self._P = new_P

    @jit
    def update(self, meas_value: np.array, meas_variance: float = 1) -> None:
        # compute the Kalman Gain
        # S = H P Ht + R
        # K = P Ht np.linalg.pinv(S)

        # update the estimate via Z
        # Z = m x[n]
        # y = Z - H x
        # x = x + K y

        # update the error covariance
        # P = (I - K H) P

        H = np.eye(DIM)
        R = np.eye(DIM)  # the measurement noise covariance matrix
        if meas_variance is not 1:
            for i in range(DIM):
                R[i, i] = meas_variance
        z = meas_value  # TODO

        S = H.dot(self._P).dot(H.T) + R
        K = self._P.dot(H.T).dot(np.linalg.pinv(S))
        y = z - H.dot(self._x)

        new_x = self._x + K.dot(y)
        new_P = (np.eye(DIM) - K.dot(H)).dot(self._P)

        self._x = new_x
        self._P = new_P

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x

