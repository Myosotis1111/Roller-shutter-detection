# Contents Overview

This folder contains extra functions that are used to optimize the usability of the software. **Related codes are not used for implementation and evaluation in the bachelor thesis.**  

# Instructions

Copy all contents in this folder to the [detection_platform](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/detection_platform) folder and replace the old [model.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/detection_platform/model.py), [view.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/detection_platform/view.py) and [evaluation.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/detection_platform/evaluation.py). The other two text files named "[dingcam02.txt](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/extra%20functions/dingcam02.txt)" and "[dingcam03.txt](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/extra%20functions/dingcam03.txt)" are just for reference of the window locations from different camera positions.

# Extra Function 1:  Auto Calibration

Now the detection platform supports uploading original images from dingcam02 and dingcam03 and will transform them into undistorted images accordingly before detection.

**Make sure that the original images fulfills the following requirements:**

1. The file name of images **includes "dingcam02" or "dingcam03"** 

1. The file name of images **doesn't include "undistorted"**

If any of the requirements are not fulfilled, no calibration step will be made on the images. If you **upload undistorted images and do not want to calibrate them**, please make sure you obtained these images by [calibration.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/preprocessing/calibration.py) in folder [preprocessing](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/preprocessing), so that the image file name would contain "undistorted".

# Extra Function 2: Fixed Window Location

**If we can assure that the camera is not moved during data collection**, then there is no need to prepare a unique window location text file for each image. We can use **a fixed window location text file** for every camera position.

This function can be activated or deactivated by modifying the following codes in **Line 24-31** of [view.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/extra%20functions/view.py):

        # # The following three lines can be enabled if you want to use a fixed window location of each image
        # self.m.txt_mode = "fixed"
        # # if txt_mode = "fixed", cropping of images will follow the same text file for each position.
        # # else by default, cropping of images will follow the corresponding text file of each image.
        # # the path of text file including window position for dingcam02.
        # self.m.cam_dingcam02_txt_path = "dingcam02.txt"
        # # the path of text file including window position for dingcam03.
        # self.m.cam_dingcam03_txt_path = "dingcam03.txt"

These codes are **by default deactivated**. If you want to use the fixed window location, please select these codes and press **"ctrl + /**" to activate them. **Also, please configure the path of window location text file, one for dingcam02 and another for dingcam03:**

    self.m.cam_dingcam02_txt_path = "your_path"
    self.m.cam_dingcam03_txt_path = "your_path"

The window location text file can be obtained via [localize.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/preprocessing/localize.py) in folder [preprocessing](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/preprocessing).

- **Warning:** Fixed window location can be only applied when
the file name of images **includes "dingcam02" or "dingcam03"**, which means other images will still apply a independent window location.
