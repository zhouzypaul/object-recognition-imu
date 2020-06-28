from darknet.darknet import performDetect
from observation_parser import parse_yolo_output
from kf.kf_v4 import f, Fi
from object_dict import object_name_to_index
from kf.make_observation import observation_to_nparray_v4, nparray_to_observation_v4, make_u
from config import *
from imu.image_info import get_angle, get_distance_center
from imu.displacement import compute_displacement_pr
import os
import numpy as np
import json
import csv


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
    for i in range(len(img_path_ls)):
        if debug: print('------start loop')

        # load info
        path = img_path_ls[i]
        if debug: print('got image path: ', path)
        angular_speed = imu_ls[i]
        vx, vy, vz = angular_speed[0], angular_speed[1], angular_speed[2]
        if debug: print("------got angular speed: ", vx, vy, vz)

        # process img
        objs_ls = process_img(path)
        img_array, unprocessed = img_to_array(objs_ls)
        if debug: print('------processed img')

        # predict
        u = make_u(img_array, vx, vy, vz)
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


# process result of darknet from detect_image, this gives the full probability distribution
# image = cv2.imread("darknet/data/person.jpg")
# detect_result = detect(net="./darknet/cfg/yolov3.cfg", meta="./darknet/cfg/kf_coco.data", image=image, thresh=.5,
#                        hier_thresh=.5, nms=.45)

# process batch result from darknet
"""
the result is in the form of ([[batch_boxes]], [[batch_scores]], [[batch_classes]])
"""
# batch_detect_result = performBatchDetect(thresh=0.70,
#                                          configPath="./darknet/cfg/yolov3.cfg",
#                                          weightPath="darknet/yolov3.weights",
#                                          metaPath="./darknet/cfg/kf_coco.data",
#                                          batch_size=3,
#                                          input_images=['darknet/data/person.jpg', 'darknet/data/horses.jpg', 'darknet/data/dog.jpg'])
# parsed_batch_result: [] = parse_yolo_batch_output(batch_detect_result)
# batch_result_array = observation_to_nparray_v2(parsed_batch_result)  # convert the result to a numpy array


# see the results
if __name__ == "__main__":
    print("-------------------kf maim-------------------")
    if increase_confidence:
        f.F = Fi
    first_state()
    original, updated = update()

    if get_original:
        with open('./kf_output/original_store.csv', 'w') as f:
            json.dump(original, f, indent=2)

        original_observation = open('./kf_output/original_read.csv', 'w', newline='')
        with original_observation:
            write = csv.writer(original_observation)
            write.writerows(original)

    with open('./kf_output/updated_store.csv', 'w') as f:
        json.dump(updated, f, indent=2)
    updated_observation = open('./kf_output/updated_read.csv', 'w', newline='')
    with updated_observation:
        write = csv.writer(updated_observation)
        write.writerows(updated)
    print("-------------------kf main-------------------")
