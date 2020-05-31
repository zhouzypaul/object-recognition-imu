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
        self._x = np.zeros(TOTOALNUM, 1)

        # covariance of GRV, initialized to be identity  TODO: possible change
        self._P = np.eye(TOTOALNUM)

    def predict(self, delta_x: float, delta_y: float) -> None:
        # x = A x + B u[n]
        # P = A P At + Q     (Q = G Gt a)

        # A should be the identity matrix, so it can be occluded
        new_x = self._x
        for i in range(NUMOBJS):
            x_index = NUMVARS * i + iX
            y_index = NUMVARS * i + iY
            new_x[x_index] += delta_x
            new_x[y_index] += delta_y

        G = np.zeros((2, 1))  # TODO: pick up
        G[iX] = 0.5 * dt**2
        G[iV] = dt
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance

        self._x = new_x
        self._P = new_P

    def update(self, meas_value: float, meas_variance: float) -> None:
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P
        H = np.zeros((1, NUMVARS))
        H[0, iX] = 1
        z = np.array([meas_value])
        R = np.array([meas_variance])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R
        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_P = (np.eye(NUMVARS) - K.dot(H)).dot(self._P)

        self._x = new_x
        self._P = new_P

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def pos(self) -> float:
        return self._x[iX]

    @property
    def vel(self) -> float:
        return self._x[iV]
