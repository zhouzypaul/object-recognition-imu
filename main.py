import json
import csv
import sys
from config import *


def run(args: []):
    """
    execute the main program and update the object recognition results using IMU info
    """
    if len(args) == 2 and args[1] == 'kf':
        print('running the Kalman Filter model ...')
        from kf.kf_v4 import Fi
        import kf_update
        print("-------------------kf maim-------------------")
        if increase_confidence:
            kf_update.f.F = Fi
        kf_update.first_state()
        original, updated = kf_update.update()

        if get_original:
            with open(kf_output_path + 'original_store.csv', 'w') as f:
                json.dump(original, f, indent=2)

            original_observation = open(kf_output_path + 'original_read.csv', 'w', newline='')
            with original_observation:
                write = csv.writer(original_observation)
                write.writerows(original)

        with open(kf_output_path + 'updated_store.csv', 'w') as f:
            json.dump(updated, f, indent=2)
        updated_observation = open(kf_output_path + 'updated_read.csv', 'w', newline='')
        with updated_observation:
            write = csv.writer(updated_observation)
            write.writerows(updated)
        print("-------------------kf main-------------------")

    elif len(args) == 2 and args[1] == 'iou':
        print('running the Intersection Over Union model')
        import iou_update
        print("--------------------iou main--------------------")
        original, updated, iou = iou_update.update(giou=False)

        if get_original:
            with open(iou_output_path + 'original_store_giou.csv', 'w') as f:
                json.dump(original, f, indent=2)

            original_observation = open(iou_output_path + 'original_read_giou.csv', 'w', newline='')
            with original_observation:
                write = csv.writer(original_observation)
                write.writerows(original)

        if get_iou:
            with open(iou_output_path + 'giou.csv', 'w') as f:
                json.dump(iou, f, indent=2)

        with open(iou_output_path + 'updated_store_giou.csv', 'w') as f:
            json.dump(updated, f, indent=2)
        updated_observation = open(iou_output_path + 'updated_read_giou.csv', 'w', newline='')
        with updated_observation:
            write = csv.writer(updated_observation)
            write.writerows(updated)
        print("--------------------iou main-------------------")

    elif len(args) == 3 and args[1] == 'iou' and args[2] == '--giou':
        print('running the Intersection Over Union model')
        import iou_update
        print("--------------------iou main--------------------")
        original, updated, iou = iou_update.update(giou=True)

        if get_original:
            with open(iou_output_path + 'original_store_giou.csv', 'w') as f:
                json.dump(original, f, indent=2)

            original_observation = open(iou_output_path + 'original_read_giou.csv', 'w', newline='')
            with original_observation:
                write = csv.writer(original_observation)
                write.writerows(original)

        if get_iou:
            with open(iou_output_path + 'giou.csv', 'w') as f:
                json.dump(iou, f, indent=2)

        with open(iou_output_path + 'updated_store_giou.csv', 'w') as f:
            json.dump(updated, f, indent=2)
        updated_observation = open(iou_output_path + 'updated_read_giou.csv', 'w', newline='')
        with updated_observation:
            write = csv.writer(updated_observation)
            write.writerows(updated)
        print("--------------------iou main-------------------")

    else:
        print("Not a valid argument. Please use one of: ")
        print("python main.py kf")
        print("python main.py iou [--giou]")


if __name__ == '__main__':
    run(sys.argv)
