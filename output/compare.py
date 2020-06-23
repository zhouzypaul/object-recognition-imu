import json


# load the iou files
with open('original_store_iou.csv', 'r') as f:
    original_iou = json.load(f)
with open('updated_store_iou.csv', 'r') as f:
    updated_iou = json.load(f)

# load the giou files
with open('original_store_giou.csv', 'r') as f:
    original_giou = json.load(f)
with open('updated_store_giou.csv', 'r') as f:
    updated_giou = json.load(f)

for i in original_iou:
    print(i)
