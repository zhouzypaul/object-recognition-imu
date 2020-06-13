import numpy as np
from kf.kf_v1 import TOTOALNUM, NUMVARS
from kf.kf_v4 import dim, num_max, num_obj, num_var


def observation_to_nparray_v1(observ: []) -> np.array:
    """
    this method is for kf_v1
    """
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


def observation_to_nparray_v4(observ: []) -> (np.array, []):
    """
    this method is for kf_v4
    input: [obj1, obj2, obj3, ...], where obj is (tag: int, con, (x, y, w, h))
    output1: np.array of shape (dim, 1)
            [[obj1 occurrence 1],
             [obj1 occurrence 2],
             ...
             [obj1 occurrence num_max],
             [obj2 occurrence 1],
             ... ]
    output2: a list of unprocessed objects: [obj1, obj2, ...], where obj is (tag: int, con, (x, y, w, h))
    """
    visited = {}  # keys: indices (objects) already encountered;  values: the number of times encountered
    unprocessed = []
    final_array = np.zeros((dim, 1))
    for object in observ:
        index = object[0]

        if index in visited and visited[index] >= num_max:
            unprocessed.append(object)
        else:
            if index in visited:
                visited[index] += 1
            if index not in visited:
                visited[index] = 1
            current_occurrence = visited[index]
            start_index = index * num_max * num_var + num_var * (current_occurrence - 1)
            final_array[start_index] = object[1]  # con
            final_array[start_index + 1] = object[2][0]  # x
            final_array[start_index + 2] = object[2][1]  # y
            final_array[start_index + 3] = object[2][2]  # w
            final_array[start_index + 4] = object[2][3]  # h
    return final_array, unprocessed
