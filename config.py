# this file holds all the variables that need to be changed before using the object recognition-IMU API

# debugger
debug = True  # set this to true to print informative messages


# compare results
get_original = True
saveImage = False  # save the image with bounding boxes


# raw data
image_directory = "./input/image"  # used in kf_main.py & iou_main.py
imu_directory = 'imu/gyro_data.csv'
raw_imu_path = '../input/imu/imu.h5'  # the path to the imu data file, used in imu/raw_data.py


# camera info
default_depth = 3  # the distance between the camera and the object seen, in meters
fps = 29.97  # frame rate of camera
dt = 1 / fps
width_angle = 118.2  # in degrees, the width angle of view from the RGB camera
height_angle = 69.5  # in degrees, the height angle of view from the RGB camera
focus = 0.01  # the distance between the camera eye and the screen where picture is formed. in meters


# image info
pixel_width = 1920  # the length of a single picture, in pixel units
pixel_height = 1080  # the height of a single picture, in pixel units
START_FRAME = 480  # index number of the starting frame
END_FRAME = 550


# imu info
imu_rate = 1000.0  # the sampling rate of the imu


# raw data bias
abias_x, abias_y, abias_z = 0.0, 0.0, 0.0  # accelerometer bias
gbias_x, gbias_y, gbias_z = -0.004873082820443066, -0.0022431677727018503, -0.003143394115523099  # gyroscope bias
rot_x, rot_y, rot_z = -3.0535298646328917, 0.07979593555081343, -0.12905816022369937  # axis-angle vector of
# IMU-to-camera rotation
time_offset = 2.3280178898330735  # time offset between the camera and the imu


# kf
increase_confidence = True  # set to true if you want the underlying Kalman Filter to increase the confidence of
# objects at every "predicting" stage, while moving the objects to a new location according to the IMU input
# the indices in the Kalman Filter state vector
num_max = 2  # the number of maximum objects assumed to be detected in each frame
num_obj = 80  # the total number of classes of objects that can be recognized by the recognition alg
iC = 0  # relative index of c, the confidence score of the object recognized, taking value in [0, 1]
iX = 1  # relative index of x, the x-position of the center of the bounding box
iY = 2  # relative index of y, the y-position of the center of the bounding box
iW = 3  # relative index of w, the width of the bounding box
iH = 4  # relative index of h, the height of the bounding box
num_var = iH + 1
dim = num_max * num_obj * num_var  # the dimension of the state vector


# iou
giou_thresh = 0.55  # generalized iou score threshold for increasing confidence
iou_thresh = 0.6  # iou score threshold for increasing confidence
get_iou = True
