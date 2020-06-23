import json
from matplotlib import pyplot as plt


# load the iou files
with open('original_store_iou.csv', 'r') as f:
    original = json.load(f)
with open('updated_store_iou.csv', 'r') as f:
    updated = json.load(f)


# process things in IOU
def is_bicycle(obj: []):
    """
    see if an object is a bicycle
    """
    if obj[0] == 'bicycle':
        return True
    else:
        return False


def create_single_item_ls(from_list: [], func) -> []:
    single_item_ls = []
    for pic in from_list:
        contain_obj = False
        for obj in pic:
            if func(obj):
                contain_obj = True
                single_item_ls.append(obj)
                break
        if not contain_obj:
            single_item_ls.append([None, 0, [None, None, None, None]])
    return single_item_ls


# create single item lists
original_bicycle_ls = create_single_item_ls(original, is_bicycle)
updated_bicycle_ls = create_single_item_ls(updated, is_bicycle)
"""
in the form of [pic1, pic2, .. ] where pic = [] or ['bicycle', con, [x, y, w, h]]
"""


# draw the data
plt.ion()
plt.figure()

plt.title('bicycle confidence')
plt.plot([obj[1] for obj in original_bicycle_ls], 'r')
plt.plot([obj[1] for obj in updated_bicycle_ls], 'b')

plt.show()
plt.ginput(timeout=300)
