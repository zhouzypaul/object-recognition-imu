import os
import csv
import json
import numpy as np
from darknet.darknet import performDetect
from observation_parser import parse_yolo_output
from iou.compute import compute_iou, compute_giou
from iou.increase_confidence import percent_increase
from iou.move_object import move_objects, move_object
from imu.displacement import compute_displacement_pr  # TODO: change the parameters there before executing main
from imu.image_info import get_angle, get_distance_center


debug = True
get_original = True
iou_thresh = 0.6  # the thresh hold of iou, if iou > thresh, two pics are considered close to each other
image_directory = "./input/image"
imu_directory = 'imu/gyro_data.csv'


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
    detect_result: {} = performDetect(imagePath=img_path, thresh=0.10,
                                      metaPath="./darknet/cfg/kf_coco.data", showImage=False)
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
    """
    original_obser_ls = []
    updated_obser_ls = []
    previous_objects: [] = []
    for i in range(len(img_path_ls)):
        if debug: print("------start loop")
        img_path = img_path_ls[i]
        if debug: print("------got path: ", img_path)
        angular_speed = imu_ls[i]
        vx, vy, vz = angular_speed[0], angular_speed[1], angular_speed[2]
        if debug: print("------got angular speed: ", vx, vy, vz)
        objects: [] = process_img(img_path)
        if debug: print("------processed img: ", len(objects), "objects detected")
        moved_objs = []  # the list for old objs after they've been moved to the new predicted location
        for prev_obj in previous_objects:
            if debug: print("--------previous object is :", prev_obj)
            dx, dy = compute_displacement_pr(vx, vy, vz, get_distance_center(prev_obj[2]), get_angle(prev_obj[2]))
            moved_obj = move_object(prev_obj, dx, dy)
            if debug: print("--------moved previous obj to: ", moved_obj)
            moved_objs.append(moved_obj)
        processed_objs = []  # the list for objs after confidence are increased
        unprocessed_objs = []  # the list for objs as YOLO detects them
        for current_obj in objects:
            if get_original:
                original_obj = get_max_con_class(current_obj)
                unprocessed_objs.append(original_obj)
                if debug: print("--------original object is", original_obj)
            increased = False  # keep track of whether the current_obj has already been increased
            if debug: print("--------current most likely object: ", get_max_con_class(current_obj))
            if debug: print("--------current object with full distribution: ", current_obj)
            for old_obj in moved_objs:
                if debug: print("------about to compute iou: ", compute_iou(old_obj[2], current_obj[0][2]))
                if compute_iou(old_obj[2], current_obj[0][2]) >= iou_thresh:  # TODO: iou/giou
                    if not increased:
                        increased = True
                        percent_increase(current_obj, old_obj[0], percent=0.5)  # TODO: increase method
                        new_obj = get_max_con_class(current_obj)
                        processed_objs.append(new_obj)
                        if debug: print("----------adding increased obj: ", new_obj)
                        break
            if not increased:
                processed_objs.append(get_max_con_class(current_obj))
                if debug: print("----------adding unincreased obj: ", get_max_con_class(current_obj))
        if debug: print("------increased all confidence possible")
        previous_objects = processed_objs
        if debug: print("------saved to previous objects")
        updated_obser_ls.append(processed_objs)
        if debug: print("------saved to processed objects")
        if get_original:
            original_obser_ls.append(unprocessed_objs)
            if debug: print("------added original unprocessed items")
        if debug: print("------end loop")
    return original_obser_ls, updated_obser_ls


# see the result
if __name__ == '__main__':
    print("--------------------------------------")
    # print("image path is: ", img_path_ls)
    original, updated = update()
    for item in updated:
        print(item)
    if get_original:
        with open('./output/original_store_iou.csv', 'w') as f:
            json.dump(original, f, indent=2)

        original_observation = open('./output/original_read_iou.csv', 'w', newline='')
        with original_observation:
            write = csv.writer(original_observation)
            write.writerows(original)

    with open('./output/updated_store_iou.csv', 'w') as f:
        json.dump(updated, f, indent=2)
    updated_observation = open('./output/updated_read_iou.csv', 'w', newline='')
    with updated_observation:
        write = csv.writer(updated_observation)
        write.writerows(updated)
    print("--------------------------------------")
