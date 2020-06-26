from filterpy.kalman import KalmanFilter
import numpy as np


debug = False

# the indices in the state vector
num_max = 2  # the number of maximum objects assumed to be detected in each frame
num_obj = 80  # the total number of classes of objects that can be recognized by the recognition alg
iC = 0  # relative index of c, the confidence score of the object recognized, taking value in [0, 1]
iX = 1  # relative index of x, the x-position of the center of the bounding box
iY = 2  # relative index of y, the y-position of the center of the bounding box
iW = 3  # relative index of w, the width of the bounding box
iH = 4  # relative index of h, the height of the bounding box
num_var = iH + 1
dim = num_max * num_obj * num_var  # the dimension of the state vector

"""
This version of KF assumes that each object can occur at most num_max times in one frame

the state space x is organized as follows:
column vector  ([obj1_instance1, obj1_instance2, ..., obj1_instance_num_max, 
                 obj2_instance1, ... , 
                                 ...       obj_num_obj_instance_num_max])
                where obj_instance = C, X, Y, W, H 

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
input_dim = 2 * num_obj * num_max  # need dx, dy for each object
f = KalmanFilter(dim_x=dim, dim_z=dim, dim_u=input_dim)
if debug: print('------created kf')
f.F = np.eye(dim)  # state transition matrix
Fi = np.eye(dim)  # state transition matrix for if confidence is to be increased
for i in range(num_max * num_obj):
    Fi[i * num_var][i * num_var] = 1.2
f.H = np.eye(dim)  # the measurement function
B = np.zeros((dim, input_dim))
col_index = 0
for instance_index in range(num_max * num_obj):
    start_row = instance_index * num_var
    x_row = start_row + iX
    y_row = start_row + iY
    B[x_row][col_index] = 1
    col_index += 1
    B[y_row][col_index] = 1
    col_index += 1
# for i in range(num_obj * num_max):
#     start_index = i * num_var
#     x_index = start_index + iX
#     y_index = start_index + iY
#     B[x_index][0] = 1
#     B[y_index][1] = 1
f.B = B  # control transition matrix
if debug: print('------finished creating kf')
