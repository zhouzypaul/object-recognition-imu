import os
import csv
import json
import math
import numpy as np
from darknet.darknet import performDetect
from observation_parser import parse_yolo_output
from iou.compute import compute_iou, compute_giou
from iou.increase_confidence import percent_increase, first_time_decrease
from iou.move_object import move_objects, move_object
from imu.displacement import compute_displacement_pr  # TODO: change the parameters there before executing main
from imu.image_info import get_angle, get_distance_center
from config import *


# get the images from input
img_path_ls = []  # a list of image paths
for image in os.scandir(image_directory):
    if image.path.endswith('.jpg') or image.path.endswith('.png') and image.is_file():
        img_path_ls.append(image.path)
img_path_ls.sort()


# incorporate IMU and depth info
imu_ls = np.loadtxt(imu_directory, delimiter=',')

assert len(img_path_ls) == len(imu_ls), "Length of IMU input should match length of camera input"


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


# loop YOLO and iou
# the outputs shall all be of single distribution, as opposed to the darknet.py output
def update() -> ():
    """
    this is where the bulk of the computation happens, using IMU info and past recognition results to update the current
    recognition
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
    for i in range(len(img_path_ls)):

        # load info
        if debug: print("------start loop")
        img_path = img_path_ls[i]
        if debug: print("------got path: ", img_path)
        angular_speed = imu_ls[i]
        vx, vy, vz = angular_speed[0], angular_speed[1], angular_speed[2]
        if debug: print("------got angular speed: ", vx, vy, vz)
        objects: [] = process_img(img_path)
        if debug: print("------processed img: ", len(objects), "objects detected")

        # move objects in previous frame
        moved_objs = []  # the list for old objs after they've been moved to the new predicted location
        for prev_obj in previous_objects:
            if debug: print("--------previous object is :", prev_obj)
            dx, dy = compute_displacement_pr(vx, vy, vz, get_distance_center(prev_obj[2]), get_angle(prev_obj[2]))
            moved_obj = move_object(prev_obj, dx, dy)
            if debug: print("--------moved previous obj to: ", moved_obj)
            moved_objs.append(moved_obj)

        # process current frame
        processed_objs = []  # the list for objs after confidence are increased
        unprocessed_objs = []  # the list for objs as YOLO detects them
        frame_ious = []  # the top iou for each object in current frame
        for current_obj in objects:
            max_con_class = get_max_con_class(current_obj)
            if debug: print("--------current most likely object: ", max_con_class)
            # if debug: print("--------current object with full distribution: ", current_obj)
            if get_original:
                unprocessed_objs.append(max_con_class)
                if debug: print("--------original object is", max_con_class)
            current_obj_ious = {}  # keys: class tags, values: iou score for that class, for the current object
            if max_con_class[0] not in seen_objects:  # if the object is not seen during previous part of the video
                first_time_decrease(current_obj, max_con_class[0], percent=0.2)
                seen_objects.append(max_con_class[0])
            for old_obj in moved_objs:
                iou_score = compute_giou(old_obj[2], current_obj[0][2])  # TODO: iou/giou
                if debug: print("------computed iou: ", iou_score)
                if get_iou:
                    if old_obj[0] in current_obj_ious:
                        current_obj_ious[old_obj[0]] = max(iou_score, current_obj_ious[old_obj[0]])
                    else:
                        current_obj_ious[old_obj[0]] = iou_score
                if iou_score >= iou_thresh:
                    percent_increase(current_obj, old_obj[0], percent=0.5)  # TODO: increase method
            new_max_con_class = get_max_con_class(current_obj)
            processed_objs.append(new_max_con_class)
            if debug: print('------adding object: ', new_max_con_class)
            if get_iou:
                if new_max_con_class[0] in current_obj_ious:
                    max_con_class_iou = current_obj_ious[new_max_con_class[0]]
                else:
                    max_con_class_iou = 0  # TODO: 0 for iou and -1 for giou
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


# see the result
if __name__ == '__main__':
    print("------------------main--------------------")
    original, updated, iou = update()

    if get_original:
        with open('./iou_output/original_store_giou.csv', 'w') as f:
            json.dump(original, f, indent=2)

        original_observation = open('./iou_output/original_read_giou.csv', 'w', newline='')
        with original_observation:
            write = csv.writer(original_observation)
            write.writerows(original)

    if get_iou:
        with open('./iou_output/giou.csv', 'w') as f:
            json.dump(iou, f, indent=2)

    with open('./iou_output/updated_store_giou.csv', 'w') as f:
        json.dump(updated, f, indent=2)
    updated_observation = open('./iou_output/updated_read_giou.csv', 'w', newline='')
    with updated_observation:
        write = csv.writer(updated_observation)
        write.writerows(updated)
    print("-------------------main-------------------")
