import math


# several different ways of increasing the confidence


def percent_increase(obj: (), percent: float = 0.2) -> ():
    """
    increase by a certain percentage, but don't exceed 1.00
    default percentage is 20%
    set percent = iou/giou if you wish to increase confidence proportionally to iou scores
    """
    con = obj[1]
    new_con = con + (1 - con) * percent
    return obj[0], new_con, obj[2]


def distance_increase(currnet_obj: (), old_obj: (), percent: float = 0.2) -> ():
    """
    the smaller the distance between two box centers, the greater the increase
    """
    x_cur, y_cur = currnet_obj[2][0], currnet_obj[2][1]
    x_old, y_old = old_obj[2][0], old_obj[2][1]
    d = math.sqrt((x_cur - x_old)**2 + (y_cur - y_old)**2)
    diag_cur = math.sqrt((currnet_obj[2][2])**2 + (currnet_obj[2][3])**2)  # diagonal length of box current
    d_percent = d / (diag_cur / 2)  # it's considered to be very far away if two pics are half a diagonal away
    if d_percent > 1:
        d_percent = 1  # keep the percentage to be between [0, 1]
    return percent_increase(obj=currnet_obj, percent=percent * (1 - d_percent))


def exponential_increase(obj: ()) -> ():
    """
    increase in an exponential manner
    """
    # TODO:
