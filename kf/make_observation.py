import numpy as np
from kf.kf_v1 import TOTOALNUM, NUMVARS


def observation_to_nparray(observ: []) -> np.array:
    # observ is a list of recognized object, which are lists themselves
    final_array = np.zeros((TOTOALNUM, 1))
    for recognized_objects in observ:
        object_index = recognized_objects[0]
        start_index = object_index * NUMVARS
        final_array[start_index] = recognized_objects[1]
        final_array[start_index + 1] = recognized_objects[2]
        final_array[start_index + 2] = recognized_objects[3]
        final_array[start_index + 3] = recognized_objects[4]
        final_array[start_index + 4] = recognized_objects[5]
    return final_array
