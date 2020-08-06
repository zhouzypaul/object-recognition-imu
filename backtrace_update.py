from iou.compute import compute_iou, compute_giou
from iou.increase_confidence import percent_increase, percent_decrease
from update_functions import *
from object_dict import object_to_index
from config import *


# get the images from input
img_path_ls = get_img_path(image_directory)


# loop YOLO and iou
# the outputs shall all be of single distribution, as opposed to the darknet.py output
def update(giou: bool) -> ():
    """
    this is where the bulk of the computation happens, using IMU info and past recognition results to update the current
    recognition
    input: True if we use generalized IoU to update the result
           False if we use the regular IoU
    output: original_obser_ls: the recognition result outputed by YOLO
            updated_obser_ls: the recognition result after being updated with IMU info
            iou_ls: the list of top IOU for each obj
    """
    original_obser_ls = []
    updated_obser_ls = []
    iou_ls = []
    previous_objects: [] = []
    seen_objects: [] = []  # tags of objects already seen in the video sequence
    prev_n_objects: [] = []  # list of tuples (tag, con), for the previous n frames
    prev_scores: {} = {} # key: object full distribution tuple, value: sub-dict of tag -> bootstrapped eligibility score
    for i in range(len(img_path_ls)):
        if debug: print("------start loop")

        # load image
        img_path = img_path_ls[i]
        if debug: print("------got path: ", img_path)

        # process image
        objects: [] = process_img(img_path)
        if debug: print("------processed img: ", len(objects), "objects detected")

        # process current frame
        processed_objs = []  # the list for objs after confidence are increased
        unprocessed_objs = []  # the list for objs as YOLO detects them
        frame_ious = []  # the top iou for each object in current frame
        cur_scores: {} = {}  # key: object full distribution, value: sub-dict of tag -> bootstrapped eligibility score

        # if there are no detections in objects
        if not objects and previous_objects:
            old_max_obj = max(previous_objects, key=lambda x: x[1])
            eligi_score = eligibility_score(old_max_obj[0], prev_n_objects)
            if eligi_score > 0:
                simulated_obj = []
                for tag in object_to_index:
                    simulated_obj.append([tag, 0, old_max_obj[2]])
                percent_increase(simulated_obj, old_max_obj[0], percent=eligi_score)
                objects = objects.append(simulated_obj)

        # if there are indeed detections in the current frame
        if objects:
            for current_obj in objects:

                # initialize the score dict
                obj_scores: {} = {}  # keys: tag  values: bootstrapped eligibility score
                for class_tag in object_to_index:
                    obj_scores[class_tag] = 0

                # get original current class & IoU
                max_con_class = get_max_con_class(current_obj)
                if debug: print("------current most likely object: ", max_con_class)
                if get_original:
                    if max_con_class[1] > detection_thresh:
                        unprocessed_objs.append(max_con_class)
                        if debug: print("------original object is", max_con_class)
                current_obj_ious = {}  # keys: class tags, values: iou score for that class, for the current object
                if max_con_class[0] not in seen_objects:  # if the object is not seen during previous part of the video
                    percent_decrease(current_obj, max_con_class[0], percent=0.2)
                    seen_objects.append(max_con_class[0])

                # INCREASE CONFIDENCE
                # if not previous_objects:  # if there are no objects detected in the previous frame
                #     eligi_score = eligibility_score(max_con_class[0], prev_n_objects)
                #     if eligi_score > 0:
                #         percent_increase(current_obj, max_con_class[0], percent=eligi_score)
                #     else:
                #         percent_decrease(current_obj, max_con_class[0], percent=-eligi_score)

                # for tag in object_to_index:
                #     eligi_score = eligibility_score(tag=tag, prev_ls=prev_n_objects)
                #     # TODO: we want to check condition before increase
                #     if eligi_score >= 0:
                #         percent_increase(current_obj, tag, percent=eligi_score)
                #         if debug: print("--increased confidence by " + str(eligi_score))
                #     if eligi_score < 0:
                #         percent_decrease(current_obj, tag, percent=-eligi_score)
                #         if debug: print("--decreased confidence by " + str(-eligi_score))

                eligi_score = eligibility_score(tag=max_con_class[0], prev_ls=prev_n_objects)  # TODO: the baseline
                if eligi_score >= 0:
                    percent_increase(current_obj, max_con_class[0], percent=eligi_score)
                    if debug: print("--increased confidence by " + str(eligi_score))
                if eligi_score < 0:
                    percent_decrease(current_obj, max_con_class[0], percent=-eligi_score)
                    if debug: print("--decreased confidence by " + str(-eligi_score))

                # for old_obj in previous_objects:
                #     old_obj = tuple(old_obj)  # turn into tuple so it's hashable
                #
                #     # get IOU score
                #     if giou:
                #         iou_score = compute_giou(old_obj[2], current_obj[0][2])
                #         if debug: print("------computed giou: ", iou_score)
                #     else:
                #         iou_score = compute_iou(old_obj[2], current_obj[0][2])
                #         if debug: print("------computed iou: ", iou_score)
                #     if get_iou:
                #         if old_obj[0] in current_obj_ious:
                #             current_obj_ious[old_obj[0]] = max(iou_score, current_obj_ious[old_obj[0]])
                #         else:
                #             current_obj_ious[old_obj[0]] = iou_score
                #
                #     # update the dict
                #     if iou_score >= moved_iou_thresh:
                #         old_obj_scores = prev_scores[old_obj]
                #         for class_tag in object_to_index:
                #             if class_tag == old_obj[0]:
                #                 obj_scores[class_tag] = old_obj_scores[class_tag] * decay_rate + delta_score * old_obj[1]
                #             else:
                #                 obj_scores[class_tag] = old_obj_scores[class_tag] * decay_rate - delta_score

                    # adjust the confidence according to the score
                    # if obj_scores[max_con_class[0]] > 0:
                    #     percent_increase(current_obj, max_con_class[0], percent=obj_scores[max_con_class[0]])
                    # if obj_scores[max_con_class[0]] < 0:
                    #     percent_decrease(current_obj, max_con_class[0], percent=-obj_scores[max_con_class[0]])
                    # for class_tag in obj_scores:
                    #     if obj_scores[class_tag] > 0:
                    #         percent_increase(current_obj, class_tag, percent=obj_scores[class_tag])
                    #     if obj_scores[class_tag] < 0:
                    #         percent_decrease(current_obj, class_tag, percent=-obj_scores[class_tag])

                    # if iou_score >= iou_thresh:
                        # eligi_score = eligibility_score(tag=old_obj[0], prev_ls=prev_n_objects)
                        # if eligi_score >= 0:
                        #     percent_increase(current_obj, old_obj[0], percent=eligi_score)
                        #     if debug: print("--increased confidence by " + str(eligi_score))
                        # if eligi_score < 0:
                        #     percent_decrease(current_obj, old_obj[0], percent=-eligi_score)
                        #     if debug: print("--decreased confidence by " + str(-eligi_score))
                    # else:  # if IOU doesn't work out
                    #     eligi_score = eligibility_score(tag=max_con_class[0], prev_ls=prev_n_objects)
                    #     if eligi_score >= 0:
                    #         percent_increase(current_obj, max_con_class[0], percent=eligi_score)
                    #         if debug: print("--increased confidence by " + str(eligi_score))
                    #     if eligi_score < 0:
                    #         percent_decrease(current_obj, max_con_class[0], percent=-eligi_score)
                    #         if debug: print("--decreased confidence by " + str(-eligi_score))

                # WRAPPING UP
                new_max_con_class = get_max_con_class(current_obj)
                cur_scores[tuple(new_max_con_class)] = obj_scores
                if new_max_con_class[1] > detection_thresh:
                    processed_objs.append(new_max_con_class)
                    if debug: print('------adding object: ', new_max_con_class)
                if get_iou:
                    if new_max_con_class[0] in current_obj_ious:
                        max_con_class_iou = current_obj_ious[new_max_con_class[0]]
                    else:
                        max_con_class_iou = 0
                    frame_ious.append((new_max_con_class[0], max_con_class_iou))
                    if debug: print('------adding iou ', new_max_con_class[0], max_con_class_iou)

        if debug: print("------increased all confidence possible")
        previous_objects = processed_objs
        if debug: print("------saved to previous objects")
        prev_scores = cur_scores
        if debug: print("------saved to previous scores")
        tags_cons = []
        for obj in processed_objs:
            tags_cons.append((obj[0], obj[1]))
        prev_n_objects.append(tags_cons)  # recording the tags and cons of prev_n
        if len(prev_n_objects) > N:
            prev_n_objects.pop(0)  # if prev_n is too long
        if debug: print("------saved to prev_n_objects")
        updated_obser_ls.append(processed_objs)
        if debug: print("------saved to processed objects")
        if get_original:
            original_obser_ls.append(unprocessed_objs)
            if debug: print("------added original unprocessed items")
        if get_iou:
            iou_ls.append(frame_ious)
            if debug: print("------added iou of the frame")
        if debug: print("------end loop")
    return original_obser_ls, updated_obser_ls, iou_ls
