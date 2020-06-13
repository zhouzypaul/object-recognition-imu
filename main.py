from darknet.darknet import performDetect, performBatchDetect, detect_image, detect
from darknet.darknet_video import YOLO
from observation_parser import parse_yolo_batch_output, parse_yolo_output
from kf.kf_v1 import KF
from kf.object_dict import object_name_to_index
from kf.make_observation import observation_to_nparray_v4
import cv2
import os
import numpy as np


# get the images from input
directory = "/home/h2r/VP/input"
img_path_ls = []  # a list of image paths
for image in os.scandir(directory):
    if image.path.endswith('.jpg') or image.path.endswith('.png') and image.is_file():
        img_path_ls.append(image.path)


# process single result form darknet
def process_img(img_path: str) -> (np.array, []):
    """
    ('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px))
    The X and Y coordinates are from the center of the bounding box.
    """
    detect_result: {} = performDetect(imagePath=img_path, thresh=0.70,
                                      metaPath="./darknet/cfg/kf_coco.data", showImage=True)
    parsed_result: [] = parse_yolo_output(detect_result)
    indexed_result: [] = object_name_to_index(parsed_result)
    result_array, unprocessed_objects = observation_to_nparray_v4(indexed_result)  # convert the result to a numpy array
    return result_array, unprocessed_objects


# get the first image's result
first_result, _ = process_img(img_path_ls[0])

# loop YOLO and KF
# TODO


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

# incorporate IMU and depth info
# TODO: need more info

# see the results
if __name__ == "__main__":
    print("------------------------------------")
    print(first_result)
    print(img_path_ls)
    print("------------------------------------")
