
<a href="https://paulzhou69.github.io/"><img src="https://i.pinimg.com/236x/fb/e4/ae/fbe4ae602785635af1b72789ab2d5c34--owl-anime.jpg" title="VP" alt="VP"></a>

# Post-Processing Model of Video Object Recognition using IMU

> a post-processing model using Inertial Measurement Unit (IMU) to enhance accuracy of object recognition 

> uses Kalman Filter and Intersection over Union model to update the recognition results with IMU information, compatible with any existing object recognition algorithms such as `YOLO` or `Detectron` 

> used on Vision Prosthetic in Paradiso Lab, Brown University




[![INSERT YOUR GRAPHIC HERE](https://github.com/paulzhou69/object-recognition-imu/blob/master/info/hardware.jpg)]()

<!-- [![HitCount](http://hits.dwyl.com/paulzhou69/object-recognition-imu.svg)](http://hits.dwyl.com/paulzhou69/object-recognition-imu) -->
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fpaulzhou69%2Fobject-recognition-imu)](https://hits.seeyoufarm.com)

---

## Table of Contents 

- [Installation](#installation)
- [Features](#features)
- [Use](#use)
- [Team](#team)
- [FAQ](#faq)


---


## Installation

- Clone this repo to your local machine using `https://github.com/paulzhou69/object-recognition-imu.git`
- install the dependencies using `requirements.txt`

```shell script
$ git clone https://github.com/paulzhou69/object-recognition-imu.git
$ pip install -r requirements.txt
```

- If you are using YOLO as your object detection algorithm, please follow the <a href="https://github.com/paulzhou69/object-recognition-imu/blob/master/darknet/README.md"> README here </a> to set up YOLO. 
Please note that the YOLO in this repo is a bit different from the <a href="https://github.com/AlexeyAB/darknet.git"> official release </a>, 
as the YOLO in this repo is better integrated with the post-processing model. 

- If you are nto using YOLO as your object detection algorithm, please build the algorithm of your choice under this folder, and change the follwing scripts: 
    1. `object_dict.py`. change the `object_to_index` dictionary to fit the classes of your detection algorithm
    2. `iou_update.py` & `kf_update.py`. Change the `process_img()` function to use your algorithm's detection function 
    3. `observation.py`. Please parse the detection result of a single frame into the form 
    ```
        a list of objects: [obj1, obj2, obj3, ..., obj N], where N is the total number of bounding boxes in the frame
        an object is a list of full probability distribution over all classes: obj = [class1, class2, ..., class M], where M is the total number of classes your algorithm can recognize
        a class is a tuple: class = (confidence, (x, y, w, h)), 
        where confidence is the confidence of the object being that class, x, y is the x, y coordinates of the center of the bounding box of the object, w, h are the width and heighht of the bounding box 
    ``` 

- If you are not using a conventional IMU/camera coordinate system orientation, please adjust the function `compute_displacement_pr()` in `imu/displacement.py` 


---

## Features

- updates the object recognition result (class confidences & bounding box locations) using 
either Kalman Filter model or a (generalized) Intersection over Union model

- save the updated results in a new file

- included functions to draw plots to compare the confidences/bounding box location before 
and after the post-processing model is applied 


---
## Usage 

> first install your own object recognition algorithm 

> then, specify your camera, IMU, and other parameters in `config.py`

> make sure to put your images and IMU raw data in the `input/image` and `input/imu` folder. The default file format for IMU data is csv. If not, please change the script `imu/raw_data.py` to take in the file format of your choosing. Then run `python main.py csv` to process the raw data

> the bulk of the model can be access by running main.py 

to see what arguments main.py takes in, run `python main.py --help`


###Example 1: 

running the Kalman Filter model and process input images and IMU info
```shell script
$ python main.py kf
```
- you should see the terminal loading the images and the processing steps if you set `debug = True` in `config.py`
[![INSERT YOUR GRAPHIC HERE](https://github.com/paulzhou69/object-recognition-imu/blob/master/info/debug.png)]()

the results of the updated observation will be saved in files in `kf_output/`, where the `read.csv`
files are for human reading the outputs, and the `store.csv` files are stored in json format
and available for further processing



###Example 2:

running the IoU model and process input images and IMU info
```shell script
$ python main.py iou
```
to run the IoU with generalized IoU, use the tag `--giou` or `-g`
```shell script
$ python main.py iou --giou
```



###Example 3:
to compare the updated results with original results output by your object detection 
algorithm, use the `--compare` or `-c` tag, but remember to specify the model you used
```shell script
$ python main.py iou --compare
$ python main.py kf --compare
```
- you should see an image pop up:
[![INSERT YOUR GRAPHIC HERE](https://github.com/paulzhou69/object-recognition-imu/blob/master/info/compare.png)]()

to customized your own compare settings, please edit `iou_output/iou_compare.py`, `iou_output/giou_compare.py`, and 
`kf_output/kf_compare.py`


---

## Contributing

> To get started...

### Step 1

- **Option 1**
    - 🍴 Fork this repo!

- **Option 2**
    - 👯 Clone this repo to your local machine using `https://github.com/paulzhou69/object-recognition-imu.git`

### Step 2

- **HACK AWAY!** 🔨🔨🔨

### Step 3

- 🔃 Create a new pull request using <a href="https://github.com/paulzhou69/object-recognition-imu/compare" target="_blank">`https://github.com/paulzhou69/object-recognition-imu/compare` </a>.

---

## Team (Paradiso Lab, Brown University)

> Contributors/People 


---

## FAQ

- **Feel free to open issues :))**

---

## Support

Reach out to me at one of the following places!

- Leave comments on my website: <a href="https://paulzhou69.github.io/" target="_blank">`https://paulzhou69.github.io/` </a>



---

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
