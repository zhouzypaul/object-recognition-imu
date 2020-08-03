from config import ground_truth_directory, OBJ_NAME, kf_output_path
import os
import json
from xml.etree import ElementTree as ET


# load files
truth_ls = []
for gt in os.scandir(ground_truth_directory):
    if gt.path.endswith('.xml') and gt.is_file():
        truth_ls.append(gt.path)
truth_ls.sort()


def extract_xml(file_name):
    tree = ET.parse(file_name)
    root = tree.getroot()
    xmax = int(root.find("./object/bndbox/xmax").text)
    xmin = int(root.find("./object/bndbox/xmin").text)
    ymax = int(root.find("./object/bndbox/ymax").text)
    ymin = int(root.find("./object/bndbox/ymin").text)
    x = 0.5 * (xmax + xmin)
    y = 0.5 * (ymax + ymin)
    w = xmax - xmin
    h = ymax - ymin
    return [OBJ_NAME, 1, [x, y, w, h]]


def make_csv():
    print('begin to parse xml file ...')
    final_list = []
    for gt_file in truth_ls:
        final_list.append(extract_xml(gt_file))
    with open(kf_output_path + 'ground_truth.json', 'w') as f:
        json.dump(final_list, f, indent=2)
    print('made json file')
    print('-----------------------------------')
