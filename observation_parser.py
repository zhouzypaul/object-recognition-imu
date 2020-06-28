import re


# the observation output is in the form of
# [('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px))]
def parse_yolo_output(output) -> []:
    """
    when showImage is True:
    input: {'detections: [obj1, obj2, obj3, ...],   'image': ...,    'caption': ...}
            where obj = (tag, con, (x, y, w, h))
    when showImage is False:
    input: [obj1, obj2, obj3, ... ]

    output: [obj1, obj2, obj3, ... ]
    """
    if type(output) == dict:
        return output.get('detections')
    elif type(output) == list:
        return output


# the batch observation is in the form of
# ([[batch_boxes]], [[batch_scores]], [[batch_classes]])
def parse_yolo_batch_output(output: ()) -> []:
    """
    input: result of perform_batch_detect from darknet.py
    output: [ [pic1], [pic2], .... ]
            [pic] = [ [obj1], [obj2], ... ]
            [obj] = [ class, con, x, y, w, h ]
    """
    boxes: [[]] = output[0]
    scores: [[]] = output[1]
    classes: [[]] = output[2]

    num_pic = len(classes)  # the number of pics being processed

    pic_list = []
    for index in range(num_pic):
        current_boxes = boxes[index]
        current_scores = scores[index]
        current_classes = classes[index]
        num_obj = len(current_boxes)  # the number of objects being recognized in the current pic
        current_pic = []
        for obj_index in range(num_obj):
            box = current_boxes[obj_index]
            score = current_scores[obj_index]
            tag = current_classes[obj_index]
            left = box[0]
            bottom = box[1]
            right = box[2]
            top = box[3]
            x = 0.5 * (left + right)
            y = 0.5 * (top + bottom)
            w = right - left
            h = top - bottom
            current_obj = [tag, score, x, y, w, h]
            current_pic.append(current_obj)
        pic_list.append(current_pic)

    return pic_list


# the observation output is in the form of
# [('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px))]
def parse_yolo_str_output(output: str) -> []:
    """
    this assumes the outpuut is a string. Need to convert the output into a string first. DON'T USE THIS
    Original output is a list
    """
    recognized_objects_ls = []
    split_output = output.split(')), ')
    for object_with_bb in split_output:
        single_items = object_with_bb.split(',')
        current_object = []
        for single_item in single_items:
            # strip the single_item of chars and
            single_item = single_item.replace("(", "")
            single_item = single_item.replace(")", "")
            single_item = single_item.replace("[", "")
            single_item = single_item.replace("]", "")
            single_item = single_item.replace("'", "")
            if single_item.startswith(" "):
                single_item = float(single_item)
            current_object.append(single_item)
        recognized_objects_ls.append(current_object)
    return recognized_objects_ls


# if the batch observation is a string
def parse_yolo_batch_str_output(output: str) -> []:
    """
    There really is no need for this function

    input: result of perform_batch_detect from darknet.py
    output: [ [pic1], [pic2], .... ]
            [pic] = [ [obj1], [obj2], ... ]
            [obj] = [ class, con, x, y, w, h ]
    """

    # break down the output string into 3 parts, boxes, scores, and classes
    boxes_scores_classes = []
    for i in re.finditer('\[\[(.*?)\]\]', output):
        boxes_scores_classes.append(i.group())
    boxes = boxes_scores_classes[0]
    scores = boxes_scores_classes[1]
    classes = boxes_scores_classes[2]

    num_pic = 0  # the total number of pictures in the series
    for i in re.finditer('\[(.*?)\]', classes):
        num_pic += 1

    # process/rearrange the string in order of pictures
    boxes = boxes[1: len(boxes) - 1]  # strip away the outer []
    scores = scores[1: len(scores) - 1]
    classes = classes[1: len(classes) - 1]

    box_iter = re.finditer('\[(.*?)\]', boxes)  # get the span of every sub []
    box_span_ls = []
    for i in box_iter:
        box_span_ls.append(i.span())

    score_iter = re.finditer('\[(.*?)\]', scores)
    score_span_ls = []
    for i in score_iter:
        score_span_ls.append(i.span())

    class_iter = re.finditer('\[(.*?)\]', classes)
    class_span_ls = []
    for i in class_iter:
        class_span_ls.append(i.span())

    ordered_ls = []  # a list of list of strings ([[pic1], [pic2]), where things in the same pic are groups together
    for pic_index in range(num_pic):
        pic = []  # a list for each individual pic, containing box, score, class, in that order
        box_span = box_span_ls[pic_index]
        score_span = score_span_ls[pic_index]
        class_span = class_span_ls[pic_index]
        pic.append(boxes[box_span[0]: box_span[1]])
        pic.append(scores[score_span[0]: score_span[1]])
        pic.append(classes[class_span[0]: class_span[1]])
        ordered_ls.append(pic)

    # give the final output
    pic_list = []
    for index in range(num_pic):
        current_pic = ordered_ls[index]
        current_boxes: str = current_pic[0]
        current_scores: str = current_pic[1]
        current_classes: str = current_pic[2]
        num_current = 0  # the number of objects detected in the current pic
        for i in re.finditer('\((.*?)\)', current_boxes):
            num_current += 1
        boxes_ls: [[]] = []
        iter = re.finditer('\((.*?)\)', current_boxes)
        for i in iter:
            boxes_ls.append(str_to_ls(i.group()))
        scores_ls: [] = str_to_ls(current_scores)
        classes_ls: [] = str_to_ls(current_classes)

        pic = []
        for obj_index in range(num_current):
            left = boxes_ls[obj_index][0]
            bottom = boxes_ls[obj_index][1]
            right = boxes_ls[obj_index][2]
            top = boxes_ls[obj_index][3]
            x = 0.5 * (left + right)
            y = 0.5 * (top + bottom)
            w = right - left
            h = top - bottom
            current_obj = [int(classes_ls[obj_index]), scores_ls[obj_index], x, y, w, h]
            pic.append(current_obj)

        pic_list.append(pic)

    return pic_list


def str_to_ls(s: str) -> []:
    """
    transform a string of the form '[num1, num2, ...]' or '(num1, num2, ...)' to a list of those numbers
    """
    stripped_s = s[1: len(s) - 1]
    str_ls = stripped_s.split(", ")
    num_ls = list(map(lambda x: float(x), str_ls))
    return num_ls
