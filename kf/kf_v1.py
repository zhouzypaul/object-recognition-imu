import numpy as np

# prediction:
# x = A x + B u[n]
# P = A P At + Q

# compute the Kalman Gain
# S = H P Ht + R
# K = P Ht np.linalg.pinv(S)

# update the estimate via Z
# Z = m x[n]
# y = Z - H x
# x = x + K y

# update the error covariance
# P = (I - K H) P

# offsets of each variable in the state vector
iC = 0
iX = 1
iY = 2
iW = 3
iH = 4
NUMVARS = iH + 1
NUMOBJS = 80
TOTOALNUM = NUMOBJS * NUMVARS


class KF:
    def __init__(self) -> None:
        # mean of state GRV, all initialized to 0
        self._x = np.zeros((TOTOALNUM, 1))

        # covariance of GRV, initialized to be identity  TODO: possible change
        self._P = np.eye(TOTOALNUM)

    def predict(self, delta_x: float, delta_y: float, process_noise_var: float) -> None:
        # x = A x + B u[n]
        # P = A P At + Q     (Q = G Gt a)

        # A should be the identity matrix, so it can be occluded
        new_x = self._x
        for i in range(NUMOBJS):
            x_index = NUMVARS * i + iX
            y_index = NUMVARS * i + iY
            new_x[x_index] += delta_x
            new_x[y_index] += delta_y  # we are assuming that head movement at from t-1 to t is the same at t to t+1
            # which means we are assuming the head movement to be smooth

        # the process noise covariance matrix
        Q = np.eye(TOTOALNUM)
        for i in range(TOTOALNUM):
            Q[i, i] = process_noise_var
        new_P = self._P + Q

        self._x = new_x
        self._P = new_P

    def update(self, meas_value: np.array, meas_variance: float) -> None:
        # compute the Kalman Gain
        # S = H P Ht + R
        # K = P Ht np.linalg.pinv(S)

        # update the estimate via Z
        # Z = m x[n]
        # y = Z - H x
        # x = x + K y

        # update the error covariance
        # P = (I - K H) P

        H = np.ones((TOTOALNUM, TOTOALNUM))
        R = np.eye(TOTOALNUM)  # the measurement noise covariance matrix
        for i in range(TOTOALNUM):
            R[i, i] = meas_variance
        z = meas_value  # TODO

        S = H.dot(self._P).dot(H.T) + R
        K = self._P.dot(H.T).dot(np.linalg.pinv(S))
        y = z - H.dot(self._x)

        new_x = self._x + K.dot(y)
        new_P = (np.eye(TOTOALNUM) - K.dot(H)).dot(self._P)

        self._x = new_x
        self._P = new_P

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x

