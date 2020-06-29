import json
import csv
import sys
import argparse
from config import *


def run(model: str, use_giou: bool):
    """
    execute the main program and update the object recognition results using IMU info
    """
    if model == 'kf':
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

    if model == 'iou':
        print('running the Intersection Over Union model')
        import iou_update
        print("--------------------iou main--------------------")
        if use_giou:
            print('using generalized IoU')
        original, updated, iou = iou_update.update(giou=use_giou)

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


def parse_arguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("model", help="model type, either kf or iou", type=str)

    # Optional arguments
    parser.add_argument("--giou", help="use generalized iou", action='store_true', default=False)

    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')

    # Parse arguments
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    run(args.model, args.giou)
