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
    x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
    x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[3]
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0 or h_intersection <= 0:  # no intersection
        return 0
    I = w_intersection * h_intersection
    U = w1 * h1 + w2 * h2 - I  # Union = total area - I
    return I / U


def compute_giou(box1: (), box2: ()) -> float:
    """
    input: box1 and box2, where each box is a tuple (x, y, w, h)
    output: the GENERALIZED intersection over union score of box1 and box2
    """
    x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
    x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[3]
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0 or h_intersection <= 0:  # no intersection
        return 0
    I = w_intersection * h_intersection
    U = w1 * h1 + w2 * h2 - I  # Union = total area - I
    # find the smallest enclosing box C
    x_right, x_left = max(x1 + 0.5 * w1, x2 + 0.5 * w2), min(x1 - 0.5 * w1, x2 - 0.5 * w2)
    y_top, y_bottom = max(y1 + 0.5 * h1, y2 + 0.5 * h2), min(y1 - 0.5 * h1, y2 - 0.5 * h2)
    C = (x_right - x_left) * (y_top - y_bottom)  # area of box C
    # GIoU = IoU - (C - U)/C
    return I / U - (C - U) / C
