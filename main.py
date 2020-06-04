from darknet.darknet import performDetect
from observation_parser import parse_yolo_output
from kf.kf_v1 import KF
from kf.object_dict import *


"""
('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px))
The X and Y coordinates are from the center of the bounding box. 
"""
detect_result = performDetect(imagePath="darknet/data/dog.jpg", thresh=0.70)
detect_result_str = str(detect_result)
parsed_result = parse_yolo_output(detect_result_str)
print("--------------")
print(parsed_result)
print("---------------")
