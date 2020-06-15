import math


class Box:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def compute_iou(box1: (), box2: ()) -> float:
    """
    input: box1 and box2, where each box is a tuple (x, y, w, h)
    output: the intersection over union score of box1 and box2
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_intersect_left = max(x1 - 0.5 * w1, x2 - 0.5 * w2)
    x_intersect_right = min(x1 + 0.5 * w1, x2 + 0.5 * w2)
    y_intersect_up = min(y1 + 0.5 * h1, y2 + 0.5 * h2)
    y_intersect_bottom = max(y1 - 0.5 * h1, y2 - 0.5 * h2)
    if x_intersect_right <= x_intersect_left or y_intersect_up <= y_intersect_bottom:  # No overlap
        return 0
    I = (x_intersect_right - x_intersect_left) * (y_intersect_up - y_intersect_bottom)
    U = w1 * h1 + w2 * h2 - I  # Union = Total Area - I
    return I / U


def compute_giou(box1: (), box2: ()) -> float:
    """
    input: box1 and box2, where each box is a tuple (x, y, w, h)
    output: the GENERALIZED intersection over union score of box1 and box2
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_intersect_left = max(x1 - 0.5 * w1, x2 - 0.5 * w2)
    x_intersect_right = min(x1 + 0.5 * w1, x2 + 0.5 * w2)
    y_intersect_up = min(y1 + 0.5 * h1, y2 + 0.5 * h2)
    y_intersect_bottom = max(y1 - 0.5 * h1, y2 - 0.5 * h2)
    if x_intersect_right <= x_intersect_left or y_intersect_up <= y_intersect_bottom:  # No overlap
        I = 0
    else:
        I = (x_intersect_right - x_intersect_left) * (y_intersect_up - y_intersect_bottom)
    U = w1 * h1 + w2 * h2 - I  # Union = Total Area - I
    # find the smallest enclosing box C
    x_right, x_left = max(x1 + 0.5 * w1, x2 + 0.5 * w2), min(x1 - 0.5 * w1, x2 - 0.5 * w2)
    y_top, y_bottom = max(y1 + 0.5 * h1, y2 + 0.5 * h2), min(y1 - 0.5 * h1, y2 - 0.5 * h2)
    C = (x_right - x_left) * (y_top - y_bottom)  # area of box C
    # GIoU = IoU - (C - U)/C
    return I / U - (C - U) / C
