import h5py
import csv
import numpy as np
import math
from config import *

# this file holds all the variables that need to be changed before using the object recognition-IMU API


# to load the data, do:
# 1. Read the raw data from the file
# 2. Correct the timestamps by using the following function:
#      time_new = time - time[0] - C['time_offset']
# 3. Apply acceleration bias by:
#      acc_new = acc - B
#    where
#      B = (C['abias_x'], C['abias_y'], C['abias_z'])
#    Similarly apply gyroscope bias on 'gyro' using 'gbias_x', 'gbias_y', and 'gbias_z'.
# 4. Compute the IMU-to-camera rotation R from the axis-angle vector
#     r = (C['rot_x'], C['rot_y'], C['rot_z'])

# 1. read in the csv file
time = []
quaternion = []
gyro = []  # [[vx, vy, vz]]
acc = []  # [[ax, ay, az]]
frame_timestamp = []  # [[frame number, timestamp]]

with open(raw_imu_path, 'r') as file:
    read = csv.reader(file, delimiter=',')
    for row in read:
        time.append(row[0])
        quaternion.append([row[4], row[1], row[2], row[3]])
        gyro.append([row[5], row[6], row[7]])
        acc.append([row[11], row[12], row[13]])
with open(raw_timestamp_path, 'r') as file:
    read = csv.reader(file, delimiter=',')
    for row in read:
        frame_timestamp.append([row[0], row[1]])

time.pop(0)  # get rid of the first item, the string heading
quaternion.pop(0)
gyro.pop(0)
acc.pop(0)
frame_timestamp.pop(0)
time = list(map(int, time))  # turn string into float
quaternion = list(map(lambda x: [float(x[0]), float(x[1]), float(x[2]), float(x[3])], quaternion))
gyro = list(map(lambda x: [float(x[0]), float(x[1]), float(x[2])], gyro))
acc = list(map(lambda x: [float(x[0]), float(x[1]), float(x[2])], acc))
frame_timestamp = list(map(lambda x: [int(x[0]), float(x[1])], frame_timestamp))

time = np.array(time)  # turn list into numpy array
quaternion = np.array(quaternion)
gyro = np.array(gyro)
acc = np.array(acc)
frame_timestamp = np.array(frame_timestamp)


# uncomment the following if you are using an hdf5 file as input format
# # 1. read in the hdf file
# hf = h5py.File(raw_imu_path, 'r')
#
# # get the data sets
# time = hf.get('time')
# time = np.array(time)
#
# gyro = hf.get('gyro')
# gyro = np.array(gyro)
# gyro = np.transpose(gyro)  # in the shape of gyro[time index][axis index]
#
# acc = hf.get('acc')
# acc = np.array(acc)
# acc = np.transpose(acc)  # in the shape of acc[time index][axis index]
#
# hf.close()


# 2. correct the time stamp
time = time - time_offset
# sweep the NaN values
for i in range(len(time)):
    if np.isnan(time[i]):
        time[i] = 0.5 * (time[i - 1] + time[i + 1])


# 3. correct the gyro & acc bias
gyro = gyro - np.array([gbias_x, gbias_y, gbias_z])
acc = acc - np.array([abias_x, abias_y, abias_z])


# 4. convert the IMU frame to the camera frame
def imu_to_camera_frame(v: np.array) -> np.array:
    """
    change the angular speed from the imy frame to the camera frame
    input: array([vx, vy, vz]), the rotational speed in IMU coordinates
    output: the speed in camera coordinates
    """
    r = np.array([rot_x, rot_y, rot_z])  # rotation axis
    theta = math.sqrt(rot_x**2 + rot_y**2 + rot_z**2)  # rotation angle in radians
    if theta > 0:
        e = r / theta  # rotation axis unit vector
        v_rot = rotate_vector(v, e, theta)  # angular velocity after rotation
        # map the new angular velocity vector to axis
        e_x, e_y, e_z = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])  # unit vectors
        new_vx = np.dot(v_rot, e_x)
        new_vy = np.dot(v_rot, e_y)
        new_vz = np.dot(v_rot, e_z)
        return np.array([new_vx, new_vy, new_vz])


def rotate_vector(v: np.array, e: np.array, theta: float) -> np.array:
    """
    rotate a 3D vector v around an axis for an angle theta
    input: v: a 3D vector np.array([x, y, z]) to be rotated
           e: the rotation axis, a unit vector
           theta: the rotation angle, in radian
    output:  v_rot, a 3D vector after rotation
    """
    # Rodrigues' Rotation formula
    # v_rot = v * cos(theta) + sin(theta) * e x v + ( 1 - cos(theta))(e * v) e
    v_rot = v * math.cos(theta) + \
            np.cross(e, v) * math.sin(theta) + \
            np.dot(e, v) * v * (1 - math.cos(theta))
    return v_rot


for i in range(len(gyro)):
    if imu_to_camera_frame(gyro[i]) is not None:
        gyro[i] = imu_to_camera_frame(gyro[i])
    if imu_to_camera_frame(acc[i]) is not None:
        acc[i] = imu_to_camera_frame(acc[i])


# 5. save to csv files
def make_csv():
    # save to csv file
    np.savetxt(gyro_path, gyro, delimiter=',')
    np.savetxt(acc_path, acc, delimiter=',')
    np.savetxt(imu_time_path, time, delimiter=',')
    np.savetxt(quaternion_path, quaternion, delimiter=',')
    np.savetxt(img_time_path, frame_timestamp, delimiter=',')


# some methods used
def rotate_orientation(v: np.array, axis: np.array) -> np.array:
    """
    rotate a vector from one coordinate orientation to another
    input: array([vx, vy, vz]), the rotational speed in IMU coordinates
    output: the speed in camera coordinates
    """
    # theta = math.sqrt(math.radians(axis[0])**2 + math.radians(axis[1])**2 + math.radians(axis[2])**2)  # rotation angle in radians
    # e = axis / theta  # rotation axis unit vector
    # v_rot = rotate_vector(v, e, theta)  # angular velocity after rotation
    # return v_rot
    theta_x, theta_y, theta_z = math.radians(axis[0]), math.radians(axis[1]), math.radians(axis[2])
    v_rot = rotate_vector(v=v, e=np.array([1, 0, 0]), theta=theta_x)
    v_rot = rotate_vector(v=v_rot, e=np.array([0, 1, 0]), theta=theta_y)
    v_rot = rotate_vector(v=v_rot, e=np.array([0, 0, 1]), theta=theta_z)
    return v_rot


def find_nearest_index(t: float) -> int:
    """
    find the nearest index for a given time
    input: t: the current time, a float
    output: the index of the item in time array that's closest to the input
    """
    estimated_index = int((t + time_offset) * imu_rate)
    if time[estimated_index] <= t:  # the index we seek is larger than estimated
        j = estimated_index
        while time[j] < t:
            j += 1
        return j
    if time[estimated_index] > t:  # the index we seek is smaller than estimated
        j = estimated_index
        while time[j] > t:
            j -= 1
        return j
