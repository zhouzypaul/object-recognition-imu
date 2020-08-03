from darknet.darknet import performDetect
from observation_parser import parse_yolo_output
from config import *
import os


def get_img_path(dir: str):
    """
    get the list of image paths from a certain directory dir
    """
    img_path_ls = []
    for image in os.scandir(dir):
        if image.path.endswith('.jpg') or image.path.endswith('.png') or image.path.endswith(
                '.JPEG') and image.is_file():
            img_path_ls.append(image.path)
    img_path_ls.sort()
    return img_path_ls


# process single result form darknet
def process_img(img_path: str) -> []:
    """
    input: a str, absolute path to the img
    output: a list of objs, [obj1, obj2, ....],
            where obj = [class1, class2, ... , class80]
            where class = ('tag', confidence, (x, y, w, h))
            The X and Y coordinates are from the center of the bounding box, w & h are width and height of the box
    """
    if not os.path.exists(img_path):
        raise ValueError("Invalid image path: " + img_path)
    detect_result: {} = performDetect(imagePath=img_path, thresh=0.10,
                                      metaPath="./darknet/cfg/kf_coco.data", showImage=saveImage)
    parsed_result: [] = parse_yolo_output(detect_result)
    return parsed_result


def get_max_con_class(full_distr: []) -> ():
    """
    get the class with greatest confidence in an object with full probability distribution \
    input: [class1, class2, ... , class80]
            where class = ('tag', confidence, (x, y, w, h))
    output: classN, the class with the greatest confidence
    """
    sorted_list = sorted(full_distr, key=lambda x: -x[1])
    return sorted_list[0]


def in_prev_n_frames(tag: str, ls: []) -> bool:
    """
    return true the tag is present in all of the previous n frames, false otherwise
    input: tag: the string tag of an object class
           ls: the list of containing tags of objects detected in previous n frames
    """
    for frame in ls:
        if tag not in frame:
            return False
    return True


def eligibility_score(tag: str, prev_ls: [], init_weight: float = 0.1, weight_increase: float = 0.05):
    """
    the eligibility score determines whether the confidence of an object is to be increased or decreased
    the score is the percent increase/decrease of the confidence

    if tag is in an item in prev_ls, it contributes positively to the eligibility score
    if tag is NOT in an item in prev_ls, it contributes negatively to the eligibility score
    the earlier items in prev_ls contributes to the eli_score with a higher weight
    """
    if not all(prev_ls):  # if all the sublists are empty
        return 0
    score = 0  # initialize the score
    count = 0  # count of the number of frames we have interated through
    for frame in reversed(prev_ls):  # iterate starting with the later frames, with have a higher weight
        tag_to_con = {}
        for obj in frame:
            if obj[0] in tag_to_con:
                tag_to_con[obj[0]] = max(tag_to_con[obj[0]], obj[1])
            else:
                tag_to_con[obj[0]] = obj[1]
        if tag in tag_to_con:
            score += init_weight + count * weight_increase * tag_to_con[tag]
        else:
            score -= init_weight + count * weight_increase
        count += 1
    return score

