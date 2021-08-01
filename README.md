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
- The [AprilTag library](https://github.com/swatbotics/apriltag)
- The other necessary libraries are listed in requirements.txt, installations are found under [Project Installation.](#project-installation)

#### Hardware

You will need:
- [Chessboard](https://www.researchgate.net/publication/330317635/figure/fig1/AS:713873762050051@1547212176704/Calibration-Boards-a-Opencv-9-6-checkerboard-b-Opencv-asymmetric-circle.ppm), which will be used to [calibrate your camera](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html).
- AprilTags which can be downloaded through [here](https://github.com/AprilRobotics/apriltag-imgs/), or you can create your own using [this package](https://github.com/AprilRobotics/apriltag-generation).
- [Calibrated Dodecahedron with the AprilTags attached](http://media.ee.ntu.edu.tw/research/DodecaPen/)
    - The Calibrated Dodecahedron comes with measured tag sizes, rotation and translation vectors, <br/>
    that described how all the tags attached to the dodecahedron form an AprilGroup. <br/>
    This information is needed to run this project, but has been ommited due to confidentiality. 
- USB Camera 

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

**Step 3 [Optional]**\
Create new Python virtual environment [Optional but best practice]
- To get the commands to create your virtual environment on Ubuntu, [click here.](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-programming-environment-on-ubuntu-20-04-quickstart)

**Step 4**\
Install the necessary libraries:

```bash
# At root directory, run
$ pip3 install -r requirements.txt
```

**If you are using the Python virtual environment, activate it and the run the above command.**

**Step 4**\
Install [OpenCV](https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html) and the [AprilTag library](https://github.com/swatbotics/apriltag)

**If you are using the Python virtual environment, link these libraries to the environment:**

```bash
cd ~/venv/lib/python3.[your python3 version number]/site-packages/
# Linking OpenCV library
ln -s /usr/lib/python3/dist-packages/cv2.cpython-38-x86_64-linux-gnu.so cv2.so
# Linking AprilTag library
ln -s /usr/local/lib/libapriltag.so
```
These were the locations the libraries downloaded for me, but they typically download under ***/usr/local/lib/python3.[your python3 version number]***. <br/><br/>
See [here for more details regarding OpenCV library.](https://stackoverflow.com/questions/37188623/ubuntu-how-to-install-opencv-for-python3)
After linking, add the ***apriltag.py*** file obtained from installation, to the same directory: ***venv/lib/python3.8/site-packages/***

**Step 5**\
Execute the program
- Take images of your chessboard with your USB camera 
- Create a folder named "***images***" under the directory: ***aprilgroup_tracking/calibration/***
- Store your images under this folder
- Obtain the Calibrated Dodecahedron and it's information (measured tag sizes, rotation and translation vectors) as a ***.json*** file
- Store this ***.json*** file under the directory: ***aprilgroup_tracking/aprilgroup_pose_estimation/april_group.json***
- There are Bash options available to run parts of the program as described below:

| Bash Options | Argument | Description |
|--------------|----------|-------------|
| `--no-enhanceape` | `-` | No enhancement on APE would be completed, meaning that `cv:solvePnP()` would be called with no extrinsic guesses. |
| `--enhanceape` | `-` | APE will be enhance via using the calculated predicted pose as an extrinsic guess for `cv:solvePnP()`. |
| `--no-opticalflow` | `-` | Optical flow will not be used to obtain the pose of the object. |
| `--opticalflow` | `‑` | Lucas Kanade pyramidal optical flow algorithm will be used to obtain more image points for pose estimation. |
| `--outliermethod` | `opencv` | Once optical flow is used, the [OpenCV](https://github.com/opencv/opencv/blob/master/samples/python/lk_track.py) outlier method will be implemented. |
| `--outliermethod` | `velocity_vector` | The velocity vector outlier method from the [paper](https://research.fb.com/wp-content/uploads/2017/09/uist2017_pen.pdf) will be implemented. |

- Calibrate, Detect and Estimate the pose of the Dodecahedron by running:

```bash
# Assumming you are in the directory: accurate-aprilgroup-tracking
# The following command line arguements are used on the terminal:

# An example:
$ python3 aprilgroup_tracking/main.py --enhanceape --opticalflow --outliermethod opencv
```

> This will search for the camera intrinsic parameters, if found, it will store them, 
> if not found, it will run the calibration using the images you took.
> After calibration, two windows will open up. 
> Move your calibrated dodecahedron in front of your camera, or vice versa.
> The pose points will be overlaid onto your dodecahedron, and the pose drawing 
> will be displayed on the other window.
> Any logs created would be stored under the ***logs*** folder.


**N.B: More details and patches coming soon...**
