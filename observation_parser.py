import re


# the observation output is in the form of
# [('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px))]
def parse_yolo_output(output: str) -> []:
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


# output = "[('dog', 0.9977676868438721, (221.8644561767578, 383.2872009277344, 196.4304656982422, 319.7508544921875)), " \
#          "('bicycle', 0.9898741245269775, (343.3896484375, 278.4660949707031, 451.69219970703125, " \
#          "308.5605773925781)), ('truck', 0.9315087795257568, (582.4560546875, 126.83545684814453, 216.3033447265625, " \
#          "78.77030181884766))] "
# out = parse_yolo_output(output)
# for i in out:
#     print(i)



