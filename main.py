import argparse
import csv
import json

from config import *


def run(model: str, use_giou: bool, compare: bool, test: bool):
    """
    execute the main program and update the object recognition results using IMU info
    store the updated results in csv files
    input: model: either kf or iou
           use_giou: whether to use generalized iou in the iou model
           compare: True to compare the results of updated and original recognition output
    """
    if model == 'kf':
        if compare:  # if we only compare to see results, don't run update function
            from kf_output import kf_compare
            kf_compare.compare()
        elif test:
            from kf_output import parse_xml
            parse_xml.make_csv()
            from kf_output import kf_test
            kf_test.test()
        else:
            print('running the Kalman Filter model ...')
            from kf.kf_v4 import Fi
            import kf_update
            print("-------------------kf maim-------------------")
            if increase_confidence:
                kf_update.f.F = Fi
            kf_update.first_state()
            original, updated = kf_update.update()

            if get_original:
                with open(kf_output_path + 'original_store.json', 'w') as f:
                    json.dump(original, f, indent=2)

                original_observation = open(kf_output_path + 'original_read.csv', 'w', newline='')
                with original_observation:
                    write = csv.writer(original_observation)
                    write.writerows(original)

            with open(kf_output_path + 'updated_store.json', 'w') as f:
                json.dump(updated, f, indent=2)
            updated_observation = open(kf_output_path + 'updated_read.csv', 'w', newline='')
            with updated_observation:
                write = csv.writer(updated_observation)
                write.writerows(updated)
            print("-------------------kf main-------------------")

    if model == 'iou':
        if compare:  # if we only compare to see results, don't run update function
            if not use_giou:
                from iou_output import iou_compare
                iou_compare.compare()
            if use_giou:
                from iou_output import giou_compare
                giou_compare.compare()
        elif test:  # if we only test the results, don't run update function
            from iou_output import parse_xml
            parse_xml.make_csv()
            if not use_giou:
                from iou_output import iou_test
                iou_test.test()
            if use_giou:
                from iou_output import giou_test
                giou_test.test()
        else:
            print('running the Intersection Over Union model')
            if use_giou:
                print('using generalized IoU')
                import iou_update
                print("--------------------giou main--------------------")
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
                print("--------------------giou main-------------------")
            else:
                import iou_update, backtrace_update  # TODO: change this simplification
                print("--------------------iou main--------------------")
                original, updated, iou = iou_update.update(giou=use_giou)

                if get_original:
                    with open(iou_output_path + 'original_store_iou.json', 'w') as f:
                        json.dump(original, f, indent=2)

                    original_observation = open(iou_output_path + 'original_read_iou.csv', 'w', newline='')
                    with original_observation:
                        write = csv.writer(original_observation)
                        write.writerows(original)

                if get_iou:
                    with open(iou_output_path + 'iou.csv', 'w') as f:
                        json.dump(iou, f, indent=2)

                with open(iou_output_path + 'updated_store_iou.json', 'w') as f:
                    json.dump(updated, f, indent=2)
                updated_observation = open(iou_output_path + 'updated_read_iou.csv', 'w', newline='')
                with updated_observation:
                    write = csv.writer(updated_observation)
                    write.writerows(updated)
                print("--------------------iou main-------------------")
    if model == 'csv':
        from imu import raw_data
        raw_data.make_csv()
        print('made csv')


def parse_arguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("model", help="model type, either kf or iou", type=str)

    # Optional arguments
    parser.add_argument("-g", "--giou", help="use generalized iou", action='store_true', default=False)
    parser.add_argument("-c", "--compare", help="compare the results", action='store_true', default=False)
    parser.add_argument("-t", "--test", help="compute the loss function to test how good the model performs", action='store_true', default=False)

    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')

    # Parse arguments
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    run(args.model, args.giou, args.compare, args.test)
