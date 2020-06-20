import math
from object_dict import object_to_index


# several different ways of increasing the confidence


def percent_increase(obj: [], tag: str, percent: float = 0.2):
    """
    increase by a certain percentage, but don't exceed 1.00
    set percent = iou/giou if you wish to increase confidence proportionally to iou scores
    input: obj = [class1, class2, ... , class80], where class = ('tag', con, (x, y, w, h))
           tag: a string, the class tag, only increase the confidence of this class
           percent: the increase percentage, default to be 20%
    output: None, update obj, where the 'tag' class's confidence is increased
    """
    class_index = object_to_index(tag)
    con = obj[class_index][1]
    new_con = con + (1 - con) * percent
    new_obj = (obj[class_index][0], new_con, obj[class_index][2])
    obj[class_index] = new_obj  # update the obj list


def distance_increase(current_obj: [], old_obj: (), percent: float = 0.2):
    """
    the smaller the distance between two box centers, the greater the increase
    input: current_obj = [class1, class2, ... , class80], where class = ('tag', con, (x, y, w, h))
           old_obj: ('tag', con, (x, y, w, h))
           percent: the increase percentage, default to be 20%
    output: None, update current_obj, only increase the class confidence of the old_obj's class
    """
    class_tag: str = old_obj[0]
    class_index: int = object_to_index(class_tag)
    current_class: () = current_obj[class_index]
    x_cur, y_cur = current_class[2][0], current_class[2][1]
    x_old, y_old = old_obj[2][0], old_obj[2][1]
    d = math.sqrt((x_cur - x_old)**2 + (y_cur - y_old)**2)
    diag_cur = math.sqrt((current_class[2][2]) ** 2 + (current_class[2][3]) ** 2)  # diagonal length of box current
    d_percent = d / (diag_cur / 2)  # it's considered to be very far away if two pics are half a diagonal away
    if d_percent > 1:
        d_percent = 1  # keep the percentage to be between [0, 1]
    percent_increase(obj=current_obj, tag=class_tag, percent=percent * (1 - d_percent))


def exponential_increase(obj: ()) -> ():
    """
    increase in an exponential manner
    """
    # TODO:


def decrease_others(obj: []) -> []:
    """
    decrease the confidence of those that aren't increased
    """
    # TODO:
