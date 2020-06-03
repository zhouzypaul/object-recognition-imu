from darknet.darknet import performDetect
from kf.kf_v1 import KF


detect_result = performDetect(imagePath="darknet/data/dog.jpg", thresh=0.70)
print(detect_result)
