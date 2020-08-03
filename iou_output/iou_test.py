from config import *
from iou.compute import compute_iou
import json
import numpy as np
from sklearn.metrics import average_precision_score as ap, mean_squared_error as mse, precision_recall_curve as prcurve
import matplotlib.pyplot as plt


# load files
with open(iou_output_path + 'original_store_iou.json', 'r') as f:
    original = json.load(f)
with open(iou_output_path + 'updated_store_iou.json', 'r') as f:
    updated = json.load(f)
with open(iou_output_path + 'ground_truth.json', 'r') as f:
    ground_truth = json.load(f)
assert len(original) == len(updated), 'length of original and updated not the same'
assert len(original) == len(ground_truth), 'length of original and ground truth not the same'


# def take_first(x):
#     if not x:
#         return []
#     else:
#         return [x[0]]
# original = list(map(take_first, original))
# updated = list(map(take_first, updated))


# true positive: IoU > 0.5 and correct classification
def is_true_positive(preds: [[]], tru: []):
    true_label, true_box = tru[0], tru[2]
    for pred in preds:
        pred_label, pred_con, pred_box = pred[0], pred[1], pred[2]
        if compute_iou(pred_box, true_box) > test_thresh and pred_label == true_label:
            return [pred_con]
    return []


# duplicated bounding box or IoU < 0.5, or IoU > 0.5 but wrong label
def is_false_positive(preds: [[]], tru: []):
    true_label, true_box = tru[0], tru[2]
    seen = []  # confidences of labels already seen
    pred_cons = []  # toReturn
    for pred in preds:
        pred_label, pred_con, pred_box = pred[0], pred[1], pred[2]
        if pred_label in seen:
            pred_cons.append(pred_con)
        else:
            seen.append(pred_label)
            if compute_iou(true_box, pred_box) < test_thresh:
                pred_cons.append(pred_con)
            if compute_iou(true_box, pred_box) >= test_thresh and pred_label != true_label:
                pred_cons.append(pred_con)
    return pred_cons


# No detection of existing object
def is_false_negative(preds: [[]], tru: []):
    count = 0
    true_label, true_box = tru[0], tru[2]
    seen = False
    for pred in preds:
        pred_label, pred_con, pred_box = pred[0], pred[1], pred[2]
        if pred_label == true_label:
            seen = True
    if not seen:
        count += 1
    return count
    # all_objs_detected = True  # assume there's not false negative
    # count = 0  # number of objects that goes undetected
    # detected_objs = list(map(lambda x: x[0], preds))
    # for test_obj in test_objs:
    #     if test_obj not in detected_objs:
    #         all_objs_detected = False
    #         count += 1
    # return not all_objs_detected, count


def create_ap_array(ls: []):
    """
    args: ls is either original or updated
    """
    y_true = np.array([])
    y_pred = np.array([])
    for i in range(len(ls)):
        tru = ground_truth[i]
        preds = ls[i]
        tp_cons = is_true_positive(preds, tru)
        fp_cons = is_false_positive(preds, tru)
        fn_count = is_false_negative(preds, tru)
        for con in tp_cons:
            y_true = np.append(y_true, 1)
            y_pred = np.append(y_pred, con)
        for con in fp_cons:
            y_true = np.append(y_true, 0)
            y_pred = np.append(y_pred, con)
        for j in range(fn_count):
            y_true = np.append(y_true, 1)
            y_pred = np.append(y_pred, 0)
    assert len(y_pred) == len(y_true), 'length of y_true, y_pred is different'
    return y_true, y_pred


def area_under_curve(x_arr: np.array, y_arr: np.array):
    length = len(x_arr)
    area = 0
    for i in range(length - 1):
        area += y_arr[i] * (x_arr[i] - x_arr[i + 1])
    return area


# def compute_loss_score(obser: []):
#     loss = 0
#     true_positive = 0
#     false_positive = 0
#     false_negative = 0
#     for frame in obser:
#         if is_false_negative(frame)[0]:
#             loss += is_false_negative(frame)[1]
#             false_negative += is_false_negative(frame)[1]
#         for detected_obj in frame:
#             if is_true_positive(detected_obj):
#                 loss += (1 - detected_obj[1])**2
#                 true_positive += (1 - detected_obj[1])**2
#             if is_false_positive(detected_obj):
#                 loss += detected_obj[1]**2
#                 false_positive += detected_obj[1]**2
#     print('true positive score: ', true_positive)
#     print('false positive score: ', false_positive)
#     print('false negative score: ', false_negative)
#     return loss / NUM_FRAME


def test():
    ori_true, ori_pred = create_ap_array(original)
    up_true, up_pred = create_ap_array(updated)
    # for i in range(len(ori_true)):
    #     if not ori_true[i] == up_true[i]:
    #         print('not the same!!!', i, ' ori_true is ', ori_true[i], ' up_true is ', up_true[i])
    # print('the length is: ', len(ori_true))
    # print('the length is: ', len(up_true))

    # ori = []
    # for i in range(len(ori_true)):
    #     ori.append((ori_true[i], ori_pred[i]))
    # ori.sort(key=lambda x: x[1], reverse=True)
    # for i in range(len(ori)):
    #     ori_true[i] = ori[i][0]
    #     ori_pred[i] = ori[i][1]
    # up = []
    # for j in range(len(up_true)):
    #     up.append((up_true[j], up_pred[j]))
    # up.sort(key=lambda x: x[1], reverse=True)
    # for j in range(len(up)):
    #     up_true[j] = up[j][0]
    #     up_pred[j] = up[j][1]
    print('ori_true is: ', ori_true)
    print('ori_pred is: ', ori_pred)
    print('up_true is: ', up_true)
    print('up_pred is: ', up_pred)
    print('the original AP score is: ', "%.3f" % ap(ori_true, ori_pred))
    print('the updated AP score is: ', "%.3f" % ap(up_true, up_pred))

    ori_precision, ori_recall, ori_thresh = prcurve(ori_true, ori_pred)
    # ori_recall = ori_recall[::-1]
    assert len(ori_precision) == len(ori_recall), 'precision, recall not the same length'
    up_precision, up_recall, up_thresh = prcurve(up_true, up_pred)
    # up_recall = up_recall[::-1]
    assert len(up_precision) == len(up_recall), 'precision, recall not the same length'

    # print('precision is: ', up_precision)
    # print('recall is: ', up_recall)
    # print('thresh is: ', ori_thresh)

    # print('original area under curve is: ', area_under_curve(ori_recall, ori_precision))
    # print('updated area under curve is: ', area_under_curve(up_recall, up_precision))

    plt.ion()
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.step(ori_recall, ori_precision, where='post')
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.title('updated')
    plt.step(up_recall, up_precision, where='post')
    plt.show()
    plt.ginput(timeout=3000)

    # print('the original MSE is: ', mse(ori_true, ori_pred))
    # print('the updated MSE is: ', mse(up_true, up_pred))

    # print('the original loss score is: ', compute_loss_score(original))
    # print()
    # print('the updated loss score is: ', compute_loss_score(updated))
