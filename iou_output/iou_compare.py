import json
import math
import numpy as np
from matplotlib import pyplot as plt
from .rename import rename
from config import *


# rename the output pictures
rename()

# load the iou files
with open(iou_output_path + 'original_store_iou.csv', 'r') as f:
    original = json.load(f)
with open(iou_output_path + 'updated_store_iou.csv', 'r') as f:
    updated = json.load(f)
with open(iou_output_path + 'iou.csv', 'r') as f:
    iou = json.load(f)
gyro: np.array = np.loadtxt(gyro_path, delimiter=',')


# get absolute gyro speed
speed = []
for i in gyro:
    speed.append(math.sqrt(i[0]**2 + i[1]**2 + i[2]**2))


# process things in IOU
def is_tvmonitor(obj: []):
    """
    see if an object is a bicycle
    """
    if obj[0] == 'person':
        return True
    else:
        return False


def create_single_item_ls(from_list: [], func, con=0) -> []:
    """
    from a list of objects, select a specific class (such as 'bycicle's)
    """
    single_item_ls = []
    for pic in from_list:
        contain_obj = False
        for obj in pic:
            if func(obj):
                contain_obj = True
                single_item_ls.append(obj)
                break
        if not contain_obj:
            single_item_ls.append([None, con, [None, None, None, None]])
    return single_item_ls


# create single item lists
original_bicycle_ls = create_single_item_ls(original, is_tvmonitor)
updated_bicycle_ls = create_single_item_ls(updated, is_tvmonitor)
iou_bicycle_ls = create_single_item_ls(iou, is_tvmonitor)
"""
in the form of [pic1, pic2, .. ] where pic = [] or ['bicycle', con, [x, y, w, h]]
"""


def compare():
    # draw the data
    plt.ion()
    plt.figure()

    plt.title('IOU - person confidence')
    plt.plot([obj[1] for obj in original_bicycle_ls], 'r', label='YOLO confidence')
    plt.plot([obj[1] for obj in updated_bicycle_ls], 'b', label='IOU model confidence')
    plt.plot([obj[1] for obj in iou_bicycle_ls], 'y', label='IOU score')
    plt.plot([iou_thresh for i in range(90)], 'k', label='IOU threshold')
    # plt.plot(speed, 'g', label='speed of user')
    plt.legend()

    plt.show()
    plt.ginput(1, timeout=3000)
