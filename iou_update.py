import os
import numpy as np
from pyquaternion import Quaternion
from darknet.darknet import performDetect
from observation_parser import parse_yolo_output
from iou.compute import compute_iou, compute_giou
from iou.increase_confidence import percent_increase, first_time_decrease
from iou.move_object import move_object
from imu.displacement import compute_displacement_pr, compute_displacement_quaternion
from imu.image_info import get_angle, get_distance_center
from imu.raw_data import rotate_orientation
from config import *


# get the images from input
img_path_ls = []  # a list of image paths
for image in os.scandir(image_directory):
    if image.path.endswith('.jpg') or image.path.endswith('.png') and image.is_file():
        img_path_ls.append(image.path)
img_path_ls.sort()


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
    input: a str, absolute path to the img
    output: a list of objs, [obj1, obj2, ....],
            where obj = [class1, class2, ... , class80]
            where class = ('tag', confidence, (x, y, w, h))
            The X and Y coordinates are from the center of the bounding box, w & h are width and height of the box
    """
    if not os.path.exists(img_path):
        raise ValueError("Invalid image path: " + img_path)
    detect_result: {} = performDetect(imagePath=img_path, thresh=0.10,
                                      metaPath="./darknet/cfg/kf_coco.data", showImage=saveImage)
    parsed_result: [] = parse_yolo_output(detect_result)
    return parsed_result


def get_max_con_class(full_distr: []) -> ():
    """
    get the class with greatest confidence in an object with full probability distribution \
    input: [class1, class2, ... , class80]
            where class = ('tag', confidence, (x, y, w, h))
    output: classN, the class with the greatest confidence
    """
    sorted_list = sorted(full_distr, key=lambda x: -x[1])
    return sorted_list[0]


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
    interval_time_ls = []
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
        if i + 1 < len(imu_time_ls):  # make sure index not out of range
            interval_time_ls.append((imu_time_ls[i + 1] - imu_time_ls[i]) / 1000)
        else:
            interval_time_ls.append(1 / imu_rate)
    del gyro_ls[:count]
    del acc_ls[:count]
    del quaternion_ls[:count]
    del imu_time_ls[:count]
    return angular_vel_ls, lin_acc_ls, quat_ls, interval_time_ls


# loop YOLO and iou
# the outputs shall all be of single distribution, as opposed to the darknet.py output
def update(giou: bool) -> ():
    """
    this is where the bulk of the computation happens, using IMU info and past recognition results to update the current
    recognition
    input: True if we use generalized IoU to update the result
           False if we use the regular IoU
    output: original_obser_ls: the recognition result outputed by YOLO
            updated_obser_ls: the recognition result after being updated with IMU info
            iou_ls: the list of top IOU for each obj
    """
    global current_obj_max_iou
    original_obser_ls = []
    updated_obser_ls = []
    iou_ls = []
    previous_objects: [] = []
    seen_objects: [] = []  # tags of objects already seen in the video sequence
    prev_orientation = None
    cur_vel = np.array([0, 0, 0])  # current velocity of the IMU, the first integration of acceleration
    for i in range(len(img_path_ls)):
        if debug: print("------start loop")

        # load image
        img_path = img_path_ls[i]
        if debug: print("------got path: ", img_path)

        # load imu
        img_time = img_time_ls[i][1]
        if debug: print('------the image time stamp is ', img_time)
        imu_time = img2imu_time(img_time)
        if debug: print('------the closest imu time is ', imu_time)
        angular_vel_ls, lin_acc_ls, quat_ls, interval_time_ls = interval_vel_acc_quat(imu_time)
        assert len(angular_vel_ls) == len(lin_acc_ls), 'length of angular vel and linear acc not the same'
        assert len(angular_vel_ls) == len(quat_ls), 'length of angular vel and quaternion not the same'
        assert len(angular_vel_ls) == len(interval_time_ls), 'length of angular vel and interval time not the same'
        if debug: print('------loaded imu data during this interval. ', 'number of data: ', len(angular_vel_ls))

        # process image
        objects: [] = process_img(img_path)
        if debug: print("------processed img: ", len(objects), "objects detected")

        # move objects in previous frame
        moved_objs = []  # the list for old objs after they've been moved to the new predicted location
        for prev_obj in previous_objects:
            if debug: print("--------previous object is :", prev_obj)
            # q = quat_ls[-1]  # get the current quaternion as orientation
            # if debug: print("--------the current quaternion is ", q)
            # cur_orientation = Quaternion(q[0], q[1], q[2], q[3])
            # delta_ori = Quaternion() if prev_orientation is None else cur_orientation * prev_orientation.inverse
            # prev_orientation = cur_orientation
            dx, dy = 0, 0  # initialize dx, dy
            for k in range(len(angular_vel_ls)):  # integrate the imu info between two frames to get actual displacement
                vx, vy, vz = angular_vel_ls[k][0], angular_vel_ls[k][1], angular_vel_ls[k][2]
                # ax, ay, az = lin_acc_ls[k][0], lin_acc_ls[k][1], lin_acc_ls[k][2]
                # cur_orientation = Quaternion(quat_ls[k][0], quat_ls[k][1], quat_ls[k][2], quat_ls[k][3])

                delta_dx, delta_dy = compute_displacement_pr(vx, vy, vz, get_distance_center(prev_obj[2]),
                                                             get_angle(prev_obj[2]), delta_t=interval_time_ls[k])
                dx += delta_dx
                dy += delta_dy

                # rotation_axis = np.array([-vx, -vy, -vz]) * (1 / imu_rate)
                # rotated_acc = rotate_orientation(v=np.array([ax, ay, az]), axis=rotation_axis)

                # delta_orientation = Quaternion() if prev_orientation is None else cur_orientation * prev_orientation.inverse
                # prev_orientation = cur_orientation
                # rotated_acc = (delta_orientation * Quaternion(0, ax, ay, az) * delta_orientation.inverse).axis
                # dx += cur_vel[0] * (1 / imu_rate) + 0.5 * rotated_acc[0] * (1 / imu_rate)**2
                # dy += cur_vel[1] * (1 / imu_rate) + 0.5 * rotated_acc[1] * (1 / imu_rate)**2
                # cur_vel = cur_vel + np.array(rotated_acc) * (1 / imu_rate)
                # cur_vel = (delta_orientation.inverse * Quaternion(0, cur_vel[0], cur_vel[1], cur_vel[2]) * delta_orientation).axis
                # cur_vel = np.array(cur_vel)
                # cur_vel = rotate_orientation(v=cur_vel, axis=-rotation_axis)

                # dx += cur_vx * (1 / imu_rate) + 0.5 * ax * (1 / imu_rate)**2
                # dy -= cur_vy * (1 / imu_rate) + 0.5 * ay * (1 / imu_rate)**2
                # cur_vx += ax * (1 / imu_rate)
                # cur_vy += ay * (1 / imu_rate)

            # dx, dy = compute_displacement_quaternion(delta_ori)
            if debug: print("--------displacement dx, dy: ", dx, dy)
            moved_obj = move_object(prev_obj, dx, dy)
            if debug: print("--------moved previous obj to: ", moved_obj)
            moved_objs.append(moved_obj)

        # process current frame
        processed_objs = []  # the list for objs after confidence are increased
        unprocessed_objs = []  # the list for objs as YOLO detects them
        frame_ious = []  # the top iou for each object in current frame
        for current_obj in objects:
            max_con_class = get_max_con_class(current_obj)
            if debug: print("------current most likely object: ", max_con_class)
            # if debug: print("--------current object with full distribution: ", current_obj)
            if get_original:
                unprocessed_objs.append(max_con_class)
                if debug: print("------original object is", max_con_class)
            current_obj_ious = {}  # keys: class tags, values: iou score for that class, for the current object
            if max_con_class[0] not in seen_objects:  # if the object is not seen during previous part of the video
                first_time_decrease(current_obj, max_con_class[0], percent=0.2)
                seen_objects.append(max_con_class[0])
            for old_obj in moved_objs:
                # iou_score = compute_iou(old_obj[2], current_obj[0][2])
                if giou:
                    iou_score = compute_giou(old_obj[2], current_obj[0][2])
                    if debug: print("------computed giou: ", iou_score)
                else:
                    iou_score = compute_iou(old_obj[2], current_obj[0][2])
                    if debug: print("------computed iou: ", iou_score)
                if get_iou:
                    if old_obj[0] in current_obj_ious:
                        current_obj_ious[old_obj[0]] = max(iou_score, current_obj_ious[old_obj[0]])
                    else:
                        current_obj_ious[old_obj[0]] = iou_score
                if iou_score >= iou_thresh:
                    percent_increase(current_obj, old_obj[0], percent=0.5)  # TODO: can change increase method here
            new_max_con_class = get_max_con_class(current_obj)
            processed_objs.append(new_max_con_class)
            if debug: print('------adding object: ', new_max_con_class)
            if get_iou:
                if new_max_con_class[0] in current_obj_ious:
                    max_con_class_iou = current_obj_ious[new_max_con_class[0]]
                else:
                    max_con_class_iou = 0
                frame_ious.append((new_max_con_class[0], max_con_class_iou))
                if debug: print('------adding iou ', new_max_con_class[0], max_con_class_iou)
        if debug: print("------increased all confidence possible")
        previous_objects = processed_objs
        if debug: print("------saved to previous objects")
        updated_obser_ls.append(processed_objs)
        if debug: print("------saved to processed objects")
        if get_original:
            original_obser_ls.append(unprocessed_objs)
            if debug: print("------added original unprocessed items")
        if get_iou:
            iou_ls.append(frame_ious)
            if debug: print("------added iou of the frame")
        if debug: print("------end loop")
    return original_obser_ls, updated_obser_ls, iou_ls
