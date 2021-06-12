# Accurate AprilGroup Tracking

[![Accurate AprilGroup Tracking](https://i.imgur.com/9NRvmJ4.jpg)]()

## Table of Contents

- [Accurate AprilGroup Tracking Milestones](#accurate-aprilgroup-tracking-milestones)
    - [High Level Goals](#high-level-goals)
- [Usage](#usage)
    - [Requirements](#requirements) 
        - [Software](#software)
        - [Hardware](#hardware)
    - [Project Installation](#project-installation)


## Accurate AprilGroup Tracking Milestones

### High Level Goals

Improve the accuracy and tracking of apriltags over typical pose estimation techniques. Algorithms that have shown promising results in this area, such as the ones found in this [paper](https://research.fb.com/wp-content/uploads/2017/09/uist2017_pen.pdf) ([video demonstration](https://www.youtube.com/watch?v=7Xczpq4VkHM), [website](http://media.ee.ntu.edu.tw/research/DodecaPen/)) will be implemented. By doing so, an understanding of pose estimation, optical flow and non-linear least squares, among others, will be gained.

To get a breakdown of the milestones for this project, [click here!](https://docs.google.com/document/d/1mbGgtIESmOyPC7zV751N53poYPESlf1Rtbdir4BS704/edit?usp=sharing)

## Usage

### Requirements

#### Software

This project was implemented and tested on Ubuntu. 
For this project, you will need:
- [Python3](https://phoenixnap.com/kb/how-to-install-python-3-ubuntu)
- [OpenCV](https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html)
- The [apriltag library](https://github.com/swatbotics/apriltag)
- The other necessary libraries are listed in requirements.txt, installations are found under [Project Installation.](#project-installation)

#### Hardware

You will need:
- [Chessboard](https://www.researchgate.net/publication/330317635/figure/fig1/AS:713873762050051@1547212176704/Calibration-Boards-a-Opencv-9-6-checkerboard-b-Opencv-asymmetric-circle.ppm), which will be used to [calibrate your camera](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html).
- AprilTags which can be downloaded through [here](https://github.com/AprilRobotics/apriltag-imgs/), or you can create your own using [this package](https://github.com/AprilRobotics/apriltag-generation).
- [Calibrated Dodecahedron with the AprilTags attached](http://media.ee.ntu.edu.tw/research/DodecaPen/)

### Project Installation

**Step 1**\
To clone and run this application, you'll need [Git](https://git-scm.com) installed on your computer. From your command line:

```bash
# Clone this repository 
git clone https://github.com/Virtana/accurate-aprilgroup-tracking.git

# Go into the repository
cd accurate-aprilgroup-tracking

```

**Step 2**\
Set Up Programming Environment [If you already have this setup, you can skip this.]

- Update and Upgrade
```bash
$ sudo apt update
$ sudo apt -y upgrade

```

- Check your python version
```bash
$ python3 -V
```

- Install pip
```bash
$ sudo apt install -y python3-pip
```

**Step 3**\
Create new Python virtual environment [Optional but best practice]
- To get the commands to create your virtual environment on Ubuntu, [click here.](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-programming-environment-on-ubuntu-20-04-quickstart)


```bash
# Upon activating the new virtual environment, at root directory, run
$ pip3 install -r requirements.txt
```

**Step 4**\
Install [OpenCV](https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html) and the [AprilTag library](https://github.com/swatbotics/apriltag)


**Step 5**\
Execute the program
- Take images of your chessboard with your USB camera and store these images under the folder: ***aprilgroup_tracking/calibration/images/***
- Calibrate you camera by running

```bash
$ python3 aprilgroup_tracking/calibration/calibrate_camera.py
```

This will store the camera intrinsic parameters and any logs created under the ***logs*** folder.


**N.B: More details and patches coming soon...**
