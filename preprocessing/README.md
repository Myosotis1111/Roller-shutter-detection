# Contents Overview

This folder contains all the related preprocessing steps before labeling including image calibration, window localizing, and cropping.

Before moving onto the Instruction, please **make sure the filename of each image is standard!**

e.g., **image_dingcam02_2023-05-16_00-00-00.jpg** is the filename of the original image taken from dingcam02. **The information of camera position in filename** is essential for preprocessing.

Also, these preprocessing steps and existing camera matrix are **only valid for the current dingcam02 and dingcam03**. If new cameras are added, codes need to be modified.



# Instruction: Calibration

**Usage**: turning the **original images from dingcam** into **undistorted images**.

1. Copy all the original images into folder [original_images](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/preprocessing/original_images).

1. run [calibration.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/preprocessing/calibration.py)

1. all the undistorted images will be stored in folder [undistorted_images](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/preprocessing/undistorted_images).

# Instruction: Window localization

**Usage**: obtaining a text file containing all the window location information. Here **a manual mapping** is applied based on Tao's thesis, as the automatic one is not designed for dingcam.

1. Changing the path to **the undistorted image** which you want to localize windows for. The path is in **Line 10** of [mapping_manual.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/preprocessing/mapping_manual.py)

1. run [localize.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/preprocessing/localize.py)

1. **Selecting feature points** on the target images, press **Enter** to finish selecting. Then selecting corresponding feature points on the template image (construction plan) , press **Enter** to finish selecting. Keep pressing **Enter** until process finished. Detailed manual feature point selection can be referred to Tao, Liu's thesis. Usually we choose four corner points on the building face.

1. The window location **text file** will **be saved under the root directory**, the same filename with the undistorted image selected.

# Instruction: Cropping

**Usage**: Cropping **undistorted images** **into window images** according to coordinate in text file.

1. Make sure there are images in folder [undistorted_images](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/preprocessing/undistorted_images). These images can either come from **Calibration** or pasted from other repositories. Please **make sure they are undistorted**.

1. Configure the text file path. There are two path need to be configured: **dingcam02** text file and **dingcam03** text file. They are in **Line 28 and 36** of [crop.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/preprocessing/crop.py) respectively. Changing it into the text file paths that you want to use as the cropping coordinates for each camera position, which can be obtained from previous **Window localization**.

1. run [crop.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/preprocessing/crop.py)

1. The script will crop all images in the folder according to the coordinates and camera position. The cropped images will then be saved in folder [cropped_images](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/preprocessing/cropped_images).

**Note:** The batch processing is based on the premise that **the positions of two video cameras are fixed**. If it is moved even slightly during the time period of these undistorted images, the same text file may not accurately localize windows of all images in the folder.

