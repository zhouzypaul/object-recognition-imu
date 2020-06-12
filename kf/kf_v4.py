from filterpy.kalman import KalmanFilter
import numpy as np


# the indices in the state vector
num_max = 2  # the number of maximum objects assumed to be detected in each frame
num_obj = 80  # the total number of classes of objects that can be recognized by the recognition alg
iC = 0  # relative index of c, the confidence score of the object recognized, taking value in [0, 1]
iX = 1  # relative index of x, the x-position of the center of the bounding box
iY = 2  # relative index of y, the y-position of the center of the bounding box
iW = 3  # relative index of w, the width of the bounding box
iH = 4  # relative index of h, the height of the bounding box
dim = num_max * num_obj * (iH + 1)  # the dimension of the state vector

"""
This version of KF assumes that each object can occur at most num_max times in one frame
the calculations of the Kalman Filter are as follows:

prediction:
x = F x + B u[n] 
P = F P Ft + Q

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

# create the KF
f = KalmanFilter(dim_x=dim, dim_z=dim, dim_u=2)
initial_state = np.zeros((dim, 1))   # TODO: change this
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
