import os
import numpy as np
from darknet.darknet import performDetect
from observation_parser import parse_yolo_output
from iou.compute import compute_iou, compute_giou
from iou.increase_confidence import percent_increase
from iou.move_object import move_objects
from imu.displacement import compute_displacement  # TODO: change the parameters there before executing main


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
imu_ls = [(0, 0)]
# TODO: need more info


# process single result form darknet
def process_img(img_path: str) -> []:
    """
    input: a str, absolute path to the img
    output: a list of objs, [obj1, obj2, ....], where each obj is:
    ('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px))
    The X and Y coordinates are from the center of the bounding box.
    """
    detect_result: {} = performDetect(imagePath=img_path, thresh=0.70,
                                      metaPath="./darknet/cfg/kf_coco.data", showImage=False)
    parsed_result: [] = parse_yolo_output(detect_result)
    return parsed_result


# loop YOLO and iou
thresh = 0.7  # the thresh hold of iou, if iou > thresh, two pics are considered close to each other
updated_obser_ls = []
previous_objects: [] = None
for i in range(len(img_path_ls)):
    if debug: print("------start loop")
    img_path = img_path_ls[i]
    delta_x = imu_ls[i][0]
    delta_y = imu_ls[i][1]
    if debug: print("------computed path, x, y")
    objects: [] = process_img(img_path)
    if debug: print("------processed img: ", objects)
    moved_objs = move_objects(objs=previous_objects, dx=delta_x, dy=delta_y)
    if debug: print("------moved objects according to dx, dy")
    processed_objs = []  # the list for objs after confidence are increased
    increased_objs = []  # keep track of objs that have already been increased, so that they don't get increased again
    for current_obj in objects:
        if debug: print("--------current object is: ", current_obj)
        for old_obj in moved_objs:
            if debug: print("--------old object is: ", old_obj)
            if (old_obj[0] == current_obj[0]) and compute_giou(old_obj[2], current_obj[2]) >= thresh:
                if current_obj not in increased_objs:
                    increased_objs.append(current_obj)
                    new_obj = percent_increase(current_obj)  # change the increase method here
                    processed_objs.append(new_obj)
                    if debug: print("----------adding increased obj: ", new_obj)
                    break
        if current_obj not in increased_objs:
            processed_objs.append(current_obj)
            if debug: print("----------adding unincreased obj: ", current_obj)
    if debug: print("------increased all con possible")
    previous_objects = processed_objs
    if debug: print("------saved to previous objects")
    updated_obser_ls.append(processed_objs)
    if debug: print("------end loop")


# see the result
if __name__ == '__main__':
    print("--------------------------------------")
    print("image path is: ", img_path_ls)
    print(updated_obser_ls)
    print("--------------------------------------")
