from darknet.darknet import performDetect, performBatchDetect
from observation_parser import parse_yolo_batch_output
from kf.kf_v1 import KF
from kf.object_dict import *
from kf.make_observation import observation_to_nparray


# experiement with batch detect
"""
the result is in the form of ([[batch_boxes]], [[batch_scores]], [[batch_classes]])
"""
batch_detect_result = performBatchDetect(thresh=0.70,
                                         configPath="./darknet/cfg/yolov3.cfg",
                                         weightPath="darknet/yolov3.weights",
                                         metaPath="./darknet/cfg/kf_coco.data",
                                         batch_size=3,
                                         input_images=['darknet/data/person.jpg', 'darknet/data/horses.jpg', 'darknet/data/dog.jpg'])
print(type(batch_detect_result))

# process the result form YOLO
"""
('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px))
The X and Y coordinates are from the center of the bounding box. 
"""
# detect_result: [] = performDetect(imagePath="darknet/data/horses.jpg", thresh=0.70, metaPath="./darknet/cfg/kf_coco.data")
# detect_result_str: str = str(detect_result)
# parsed_result: [] = parse_yolo_output(detect_result_str)
# change_obejct_name_to_index(parsed_result)

# convert the result to a numpy array
# result_array = observation_to_nparray(parsed_result)

# incorporate IMU and depth info
# TODO: need more info

# apply the Kalman Filter
kf = KF()

# see the results
print("------------------------------------")
print(batch_detect_result)
print(batch_detect_result[1])
print("------------------------------------")
