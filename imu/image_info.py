import math


# get certain info from the processed image (a list of object tuples)


def get_distance_center(box: ()) -> float:
    """
    calculate the distance from the center of the box to the center of the image
    input: box: a bounding box
    output: the distance
    """
    return math.sqrt((box[0])**2 + (box[1])**2)


def get_angle(box: ()) -> float:
    """
    calculate the polar angle of an object center
    input: box: the bounding box of an object
    output: the polar angle from the center of box to the center of image, ranging from [0, 2pi)
    """
    x, y = box[0], box[1]
    if x > 0 and y > 0:
        return math.atan(y / x)
    if x < 0:
        return math.pi + math.atan(y / x)
    if x > 0 and y < 0:
        return 2 * math.pi + math.atan(y / x)
