import h5py
import numpy as np
import math

fps = 29.97  # the fps of the video
imu_rate = 1000.0  # the sampling rate of the imu
abias_x, abias_y, abias_z = 0.0, 0.0, 0.0
gbias_x, gbias_y, gbias_z = -0.004873082820443066, -0.0022431677727018503, -0.003143394115523099  # gyroscope bias
rot_x, rot_y, rot_z = -3.0535298646328917, 0.07979593555081343, -0.12905816022369937  # IMU-to-camera rotation
time_offset = 2.3280178898330735  # time offset between the camera and the imu

START_FRAME = 480
END_FRAME = 550
PATH = '../input/imu/imu.h5'  # the path to the imu data file

debug = False

# to load the data, do:
# 1. Read the raw data from the imu.h5 file
# 2. Correct the timestamps by using the following function:
#      time_new = time - time[0] - C['time_offset']
# 3. Apply acceleration bias by:
#      acc_new = acc - B
#    where
#      B = (C['abias_x'], C['abias_y'], C['abias_z'])
#    Similarly apply gyroscope bias on 'gyro' using 'gbias_x', 'gbias_y', and 'gbias_z'.
# 4. Compute the IMU-to-camera rotation R from the axis-angle vector
#     r = (C['rot_x'], C['rot_y'], C['rot_z'])


# 1. read in the hdf file
hf = h5py.File(PATH, 'r')

# get the data sets
time = hf.get('time')
time = np.array(time)

gyro = hf.get('gyro')
gyro = np.array(gyro)
gyro = np.transpose(gyro)  # in the shape of gyro[time index][axis index]

acc = hf.get('acc')
acc = np.array(acc)
acc = np.transpose(acc)  # in the shape of acc[time index][axis index]

hf.close()

# 2. correct the time stamp
init_time = time[0]
time = time - init_time - time_offset
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
    v_x, v_y, v_z = v[0], v[1], v[2]  # the magnitude of angular velocities
    vx, vy, vz = np.array([v_x, 0, 0]), np.array([0, v_y, 0]), np.array([0, 0, v_z])  # angular velocity in vector form
    # rotation around x-axis
    vy = rotate_vector(v=vy, e=np.array([1, 0, 0]), theta=rot_x)
    vz = rotate_vector(v=vz, e=np.array([1, 0, 0]), theta=rot_x)
    # rotation around y-axis
    vx = rotate_vector(v=vx, e=np.array([0, 1, 0]), theta=rot_y)
    vz = rotate_vector(v=vz, e=np.array([0, 1, 0]), theta=rot_y)
    # rotation around z-axis
    vx = rotate_vector(v=vx, e=np.array([0, 0, 1]), theta=rot_z)
    vy = rotate_vector(v=vy, e=np.array([0, 0, 1]), theta=rot_z)
    # max the three new angular velocity vectors to axis
    x, y, z = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])  # unit vectors
    new_vx = np.dot(vx, x) + np.dot(vy, x) + np.dot(vz, x)
    new_vy = np.dot(vx, y) + np.dot(vy, y) + np.dot(vz, y)
    new_vz = np.dot(vx, z) + np.dot(vy, z) + np.dot(vz, z)
    assert (vx + vy + vz)[0] == new_vx, "error on x"
    assert (vx + vy + vz)[1] == new_vy, "error on y"
    assert (vx + vy + vz)[2] == new_vz, "error on z"
    return np.array([new_vx, new_vy, new_vz])


def rotate_vector(v: np.array, e: np.array, theta: float) -> np.array:
    """
    rotate a 3D vector v around an axis for an angle theta
    input: v: a 3D vector np.array([x, y, z]) to be rotated
           e: the rotation axis, a unit vector
           theta: the rotation angle, in degrees
    output:  v_rot, a 3D vector after rotation
    """
    # Rodrigues' Rotation formula
    # v_rot = v * cos(theta) + sin(theta) * e x v + ( 1 - cos(theta))(e * v) e
    v_rot = v * math.cos(math.radians(theta)) + \
            np.cross(e, v) * math.sin(math.radians(theta)) + \
            np.dot(e, v) * v * (1 - math.cos(math.radians(theta)))
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


# time_array = np.array([])
gyro_array = np.array([0, 0, 0])
for frame_index in range(START_FRAME, END_FRAME):
    current_time = frame_index / fps
    current_angular_speed = gyro[find_nearest_index(current_time)]
    # time_array = np.append(time_array, time[find_nearest_index(current_time)])
    gyro_array = np.append(gyro_array, imu_to_camera_frame(current_angular_speed))
    gyro_array = gyro_array.reshape(-1, 3)

# save to csv file
np.savetxt('gyro_data.csv', gyro_array, delimiter=',')

if debug:
    print(gyro_array)
    # print(time_array)
