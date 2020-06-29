
<a href="https://paulzhou69.github.io/"><img src="https://i.pinimg.com/236x/fb/e4/ae/fbe4ae602785635af1b72789ab2d5c34--owl-anime.jpg" title="VP" alt="VP"></a>

# Post-Processing Model of Video Object Recognition using IMU

> a post-processing model using Inertial Measurement Unit (IMU) to enhance accuracy of object recognition 

> uses Kalman Filter and Intersection over Union model to update the recognition results with IMU information, compatible with any existing object recognition algorithms such as `YOLO` or `Detectron` 

> used on Vision Prosthetic in Paradiso Lab, Brown University




[![INSERT YOUR GRAPHIC HERE](https://github.com/paulzhou69/object-recognition-imu/blob/master/info/hardware.jpg)]()



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

> make sure to put your images and IMU raw data in the `input/image` and `input/imu` folder. The default file format for IMU data is csv. If not, please change the script `imu/raw_data.py` to take in the file format of your choosing

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

- Website at <a href="https://paulzhou69.github.io/" target="_blank">`https://paulzhou69.github.io/` </a>
- Email at `paul_zhou@brown.edu`



---

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2020 © Zhiyuan Zhou