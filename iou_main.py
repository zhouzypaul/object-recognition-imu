import os
import numpy as np
from darknet.darknet import performDetect
from observation_parser import parse_yolo_output
from iou.compute import compute_iou, compute_giou
from iou.increase_confidence import percent_increase
from iou.move_object import move_objects, move_object
from imu.displacement import compute_displacement  # TODO: change the parameters there before executing main
from imu.image_info import get_angle, get_distance_center


# debugger, set to true to see debug results
debug = False


# get the images from input
directory = "/home/h2r/VP/input"
img_path_ls = []  # a list of image paths
for image in os.scandir(directory):
    if image.path.endswith('.jpg') or image.path.endswith('.png') and image.is_file():
        img_path_ls.append(image.path)
img_path_ls.sort()


# incorporate IMU and depth info
imu_ls = [(0, 0, 0)]
# TODO: put the IMU data in /input, and import it here


# process single result form darknet
def process_img(img_path: str) -> []:
    """
    input: a str, absolute path to the img
    output: a list of objs, [obj1, obj2, ....],
            where obj = [class1, class2, ... , class80]
            where class = ('tag', confidence, (x, y, w, h))
            The X and Y coordinates are from the center of the bounding box, w & h are width and height of the box
    """
    detect_result: {} = performDetect(imagePath=img_path, thresh=0.70,
                                      metaPath="./darknet/cfg/kf_coco.data", showImage=True)
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
iou_thresh = 0.7  # the thresh hold of iou, if iou > thresh, two pics are considered close to each other
updated_obser_ls = []
previous_objects: [] = []
for i in range(len(img_path_ls)):
    if debug: print("------start loop")
    img_path = img_path_ls[i]
    if debug: print("------computed path: ", img_path)
    angular_speed = imu_ls[i]
    vx, vy, vz = angular_speed[0], angular_speed[1], angular_speed[2]
    if debug: print("------computed angular speed: ", vx, vy, vz)
    objects: [] = process_img(img_path)
    if debug: print("------processed img: ", objects)
    processed_objs = []  # the list for objs after confidence are increased
    for current_obj in objects:
        increased = False  # keep track of whether the current_obj has already been increased
        if debug: print("--------current object with full distribution: ", current_obj)
        for old_obj in previous_objects:
            if debug: print("--------old object is: ", old_obj)
            dx, dy = compute_displacement(vx, vy, vz, get_distance_center(old_obj[2]), get_angle(old_obj[2]))
            moved_obj = move_object(old_obj, dx, dy)
            if debug: print("--------moved old object to: ", moved_obj)
            if compute_giou(moved_obj[2], current_obj[0][2]) >= iou_thresh:
                if not increased:
                    increased = True
                    # increased_objs.append(current_obj)
                    percent_increase(current_obj, moved_obj[0])  # change the increase method here
                    new_obj = get_max_con_class(current_obj)
                    processed_objs.append(new_obj)
                    if debug: print("----------adding increased obj: ", new_obj)
                    break
        if not increased:
            processed_objs.append(get_max_con_class(current_obj))
            if debug: print("----------adding unincreased obj: ", current_obj)
    if debug: print("------increased all confidence possible")
    previous_objects = processed_objs
    if debug: print("------saved to previous objects")
    updated_obser_ls.append(processed_objs)
    if debug: print("------end loop")


# see the result
if __name__ == '__main__':
    print("--------------------------------------")
    print("image path is: ", img_path_ls)
    print(updated_obser_ls[0])
    # for obj in updated_obser_ls[0]:
    #     print(obj)
    #     for instance in obj:
    #         print(instance)
    # print(len(updated_obser_ls[0]))
    print("--------------------------------------")
