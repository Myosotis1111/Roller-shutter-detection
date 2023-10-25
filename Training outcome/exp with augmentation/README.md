# Building management Machine Vision

# Author：Yang Xinchen

# Bachelor thesis:

**(Updated on 04.06.2023)**

First draft of the complete thesis uploaded.

Detailed evaluation result on the test dataset uploaded.
               

# Python project：

The python project （see folder “**detection_platform**”） is packed using conda. Users can download the folder (please make sure there is **at least 8GB** available in the disk Anaconda installed), import to IDE and follow three steps to configure the environment:

**(1) Use Conda to create a new environment**. Run the following command in a terminal or command prompt:

    conda env create -f environment.yaml

This will create a new Conda environment using the configuration in the environment.yaml file.

**(2)** After the environment is created successfully, **activate the newly created environment.** Run the following command:

    conda activate rs_detection

**(3)** Finally, **change the Python interpreter** of the program to the interpreter in the conda environment you just created. The project can then be run via **main.py**.

# Dataset

The dataset used for training and evaluation can be access via the link below:

https://cloud.th-luebeck.de/index.php/apps/files/?dir=/Bachelor_ECUST/20_Dataset&fileid=4892230

# Executable Software (no python environment needed):

For common users, an executable software version is provided with all the features. It can be downloaded via the link below:

https://cloud.th-luebeck.de/index.php/apps/files/?dir=/Bachelor_ECUST/30_GUI&fileid=4120537

**(1)** To install it, unzip the zip file to any path except desktop (unknown error may occur). 

**(2)** To run it, run the .exe file in the folder "detection_platform".

**Note:** Folder "test images" contains images as samples to upload and detect.
