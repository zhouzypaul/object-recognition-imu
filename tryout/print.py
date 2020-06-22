import csv
import pickle
import json


ls = [[('car', 0.9, (1, 2, 3, 4)), ('person', 0.8, (2, 3, 4, 5))], [('car', 0.3, (1, 2, 3, 4))]]
# file = open('data.csv', 'w', newline='')
# with file:
#     write = csv.writer(file)
#     write.writerows(ls)


# loaded = []
# with open('data.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         loaded.append(row[0])
#         print(type(row[0]))
# print(loaded)




# with open('data.csv', 'wb') as fp:
#     pickle.dump(ls, fp)
#
# with open('data.csv', 'wb') as fp:
#     b = pickle.load(fp)
#
# print(b)
# print(type(b))



with open('data.csv', 'w') as f:
    json.dump(ls, f, indent=2)

with open('data.csv', 'r') as f:
    loaded = json.load(f)

print(loaded)
