import json


# load the giou files
with open('original_store_giou.csv', 'r') as f:
    original_giou = json.load(f)
with open('updated_store_giou.csv', 'r') as f:
    updated_giou = json.load(f)