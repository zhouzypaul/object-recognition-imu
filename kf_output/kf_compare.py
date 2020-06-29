import json
import math
import numpy as np
from matplotlib import pyplot as plt
from config import *
from mpl_toolkits.mplot3d import Axes3D
from .rename import rename


# rename the output pictures
# rename()

# load the iou files
with open(kf_output_path + 'original_store.csv', 'r') as f:
    original = json.load(f)
with open(kf_output_path + 'updated_store.csv', 'r') as f:
    updated = json.load(f)
gyro: np.array = np.loadtxt(imu_directory, delimiter=',')


# get absolute gyro speed
speed = []
for i in gyro:
    speed.append(math.sqrt(i[0]**2 + i[1]**2 + i[2]**2))


# process things in IOU
def is_bicycle(obj: []):
    """
    see if an object is a bicycle
    """
    if obj[0] == 'bicycle':
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
original_bicycle_ls = create_single_item_ls(original, is_bicycle)
updated_bicycle_ls = create_single_item_ls(updated, is_bicycle)
"""
in the form of [pic1, pic2, .. ] where pic = [] or ['bicycle', con, [x, y, w, h]]
"""


def compare():
    # draw the data
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # plt.title('Kalman Filter increase - bicycle confidence')
    # plt.plot([obj[1] for obj in original_bicycle_ls], 'r', label='YOLO confidence')
    # plt.plot([obj[1] for obj in updated_bicycle_ls], 'b', label='IOU model confidence')
    # # plt.plot(speed, 'g', label='speed of user')
    # plt.legend()

    plt.title('Kalman Filter - bicycle location')
    x = np.array([obj[2][0] for obj in original_bicycle_ls])
    y = np.array([obj[2][1] for obj in original_bicycle_ls])
    t = np.array([i + 1 for i in range(len(original_bicycle_ls))])
    ax.scatter3D(x, y, t, c=t, cmap='Reds', label='old')
    x = np.array([obj[2][0] for obj in updated_bicycle_ls])
    y = np.array([obj[2][1] for obj in updated_bicycle_ls])
    ax.scatter3D(x, y, t, c=t, cmap='Blues', label='new')
    ax.set_xlabel('box x coordinates')
    ax.set_ylabel('box y coordinates')
    ax.set_zlabel('frame')
    plt.legend()

    plt.show()
    plt.ginput(1)
