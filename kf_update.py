from darknet.darknet import performDetect
from observation_parser import parse_yolo_output
from kf.kf_v4 import f, input_dim
from object_dict import object_name_to_index
from kf.make_observation import observation_to_nparray_v4, nparray_to_observation_v4, make_u, make_u_quaternion
from config import *
import os
import numpy as np
from update_functions import get_img_path
from pyquaternion import Quaternion


# get the images from input
img_path_ls = get_img_path(image_directory)
# img_path_ls = []  # a list of image paths
# for image in os.scandir(image_directory):
#     if image.path.endswith('.jpg') or image.path.endswith('.png') and image.is_file():
#         img_path_ls.append(image.path)
# img_path_ls.sort()


# incorporate IMU info
gyro_ls = np.loadtxt(gyro_path, delimiter=',')
acc_ls = np.loadtxt(acc_path, delimiter=',')
quaternion_ls = np.loadtxt(quaternion_path, delimiter=',')
imu_time_ls = np.loadtxt(imu_time_path, delimiter=',')
img_time_ls = np.loadtxt(img_time_path, delimiter=',')
gyro_ls = list(gyro_ls)  # turn np.array into lists
acc_ls = list(acc_ls)
quaternion_ls = list(quaternion_ls)
imu_time_ls = list(imu_time_ls)
img_time_ls = list(img_time_ls)


# process single result form darknet
def process_img(img_path: str) -> []:
    """
    process a single result from darknet and parse it into a list
    input: a str, absolute path to the img
    output: a list of objs, [obj1, obj2, ....],
            where obj = ('tag', confidence, (x, y, w, h))
            The X and Y coordinates are from the center of the bounding box, w & h are width and height of the box
    """
    if not os.path.exists(img_path):
        raise ValueError("Invalid image path: " + img_path)
    detect_result: {} = performDetect(imagePath=img_path, thresh=0.10,
                                      metaPath="./darknet/cfg/kf_coco.data", showImage=saveImage)
    parsed_result: [] = parse_yolo_output(detect_result)
    for j in range(len(parsed_result)):
        parsed_result[j] = get_max_con_class(parsed_result[j])  # convert the full distribution to single distribution
    return parsed_result


def img_to_array(obj_ls: []) -> (np.array, []):
    """
    take the input from process_img(), and turn those that can fit into KF into a numpy array
    input: obj_ls: output from process_img()
    output1: a numpy array representing those objects that can fit into KF
    output2: a list of unprocessed objects
    """
    indexed_result: [] = object_name_to_index(obj_ls)
    result_array, unprocessed_objects = observation_to_nparray_v4(indexed_result)  # convert the result to a numpy array
    return result_array, unprocessed_objects


def get_max_con_class(full_distr: []) -> ():
    """
    get the class with greatest confidence in an object with full probability distribution \
    input: [class1, class2, ... , class80]
            where class = ('tag', confidence, (x, y, w, h))
    output: classN, the class with the greatest confidence
    """
    sorted_list = sorted(full_distr, key=lambda x: -x[1])
    return sorted_list[0]


# get the first image's result
def first_state():
    """
    use the first pic to first update KF's first state
    """
    first_obj_ls = process_img(img_path_ls[0])
    first_result, _ = img_to_array(first_obj_ls)
    if debug: print('------got first img')
    f.x = first_result
    if debug: print('------changed first state of kf')


def strip_overconfidence(objs_ls: []):
    """
    update the objects list so that none of the confidence will reach over 1
    """
    for i in range(len(objs_ls)):
        obj = objs_ls[i]
        if obj[1] > 1:  # confidence is greater than 1
            new_obj = (obj[0], 1, obj[2])
            objs_ls[i] = new_obj


def img2imu_time(t_img: int):
    """
    given the unix time an image is taken, return the closest unix time where imu has a record
    input: t_img, the unix time an image is taken
    output: the unix time in imu_time_ls that's closest to t_img
    """
    smaller_time = -float('inf')  # initialize
    bigger_time = None  # the two closest time to t_img in imu_time_ls
    for t in imu_time_ls:
        if t <= t_img:
            smaller_time = t
        else:
            bigger_time = t
            break
    if t_img - smaller_time < bigger_time - t_img:
        return smaller_time
    else:
        return bigger_time


def interval_vel_acc_quat(t_imu: int):
    """
    given an imu_time, return all the gyro & acc info from the start of gyro_ls/acc_ls/quaternion_ls till imu_time,
    then remove all things returned from gyro_ls/acc_ls/quaternion_ls, remove all time from the start till imu_time in
    imu_time_ls
    input: t_imu, an imu_time
    output1: a list of angular velocities [[vx, vy, vz]]
    output2: a list of linear accelerations [[ax, ay, az]]
    output3: a list of quaternions [[w, i, j, k]]
    """
    angular_vel_ls = []
    lin_acc_ls = []
    quat_ls = []
    count = 0
    for time in imu_time_ls:  # TODO: this is wrong, the very first ones aren't in video
        if time <= t_imu:
            count += 1
        else:
            break
    for i in range(count):
        gyro = gyro_ls[i]
        acc = acc_ls[i]
        quat = quaternion_ls[i]
        angular_vel_ls.append(gyro)
        lin_acc_ls.append(acc)
        quat_ls.append(quat)
    del gyro_ls[:count]
    del acc_ls[:count]
    del quaternion_ls[:count]
    del imu_time_ls[:count]
    return angular_vel_ls, lin_acc_ls, quat_ls


# loop YOLO and KF
def update() -> ():
    """
    this is where the bulk of the computation happens, using IMU info and past recognition results to update the current
    recognition, using a Kalman Filter
    output: original_obser_ls: the recognition result outputed by YOLO
            updated_obser_ls: the recognition result after being updated with IMU info using KF
    """
    original_obser_ls = []
    updated_obser_ls = []
    prev_orientation = None
    for i in range(len(img_path_ls)):
        if debug: print('------start loop')

        # load image
        path = img_path_ls[i]
        if debug: print('got image path: ', path)

        # process img
        objs_ls = process_img(path)
        img_array, unprocessed = img_to_array(objs_ls)
        if debug: print('------processed img')

        # load imu
        # img_time = img_time_ls[i][1]
        # if debug: print('------the image time stamp is ', img_time)
        # imu_time = img2imu_time(img_time)
        # if debug: print('------the closest imu time is ', imu_time)
        # angular_vel_ls, lin_acc_ls, quat_ls = interval_vel_acc_quat(imu_time)
        # assert len(angular_vel_ls) == len(lin_acc_ls), 'length of angular vel and linear acc not the same'
        # assert len(angular_vel_ls) == len(quat_ls), 'length of angular vel and quaternion not the same'
        # if debug: print('------loaded imu data during this interval. ', 'number of data: ', len(angular_vel_ls))
        angular_vel_ls = [[0, 0, 0]]  # TODO: delete this
        interval_time_ls = [1 / fps]

        # predict
        u = np.zeros((input_dim, 1))
        for av in angular_vel_ls:
            du = make_u(img_array, av[0], av[1], av[2])
            # u = u + du  # TODO: change this
        # q = quat_ls[-1]
        # if debug: print("------the current quaternion is ", q)
        # cur_orientation = Quaternion(q[0], q[1], q[2], q[3])
        # delta_ori = Quaternion() if prev_orientation is None else cur_orientation * prev_orientation.inverse
        # prev_orientation = cur_orientation
        # u = make_u_quaternion(img_array, delta_ori)
        if debug: print('------got input array u')
        f.predict(u=u)
        if debug: print('------predicted')

        # update
        f.update(z=img_array)
        if debug: print('------updated')

        # output
        updated_obser: [] = nparray_to_observation_v4(f.x, unprocessed)
        if debug: print('------change back to observation')
        strip_overconfidence(objs_ls)
        strip_overconfidence(updated_obser)
        original_obser_ls.append(objs_ls)
        if debug: print('------added to original_obser_ls')
        updated_obser_ls.append(updated_obser)
        if debug: print('------added to updated_obser_ls')
        if debug: print('------finished loop')
    return original_obser_ls, updated_obser_ls
