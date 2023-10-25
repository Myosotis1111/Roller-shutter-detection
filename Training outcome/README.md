# Content

This folder contains weights and data obtained from the training process. 

- # [exp with augmentation](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/Training%20outcome/exp%20with%20augmentation)

The training outcome of running 200 epoches using yolov5m with data augmentation (brightness, cut-off) applied. The training set was thus doubled to 7652 images.

- # [exp without augmentation](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/Training%20outcome/exp%20without%20augmentation)

The training outcome of running 200 epoches using yolov5m without any data augmentation. The training set only contains the original 3826 cropped window images.

# Training Hardware

- GPU: **NIVIDA GeForce RTX 2060 (6 GB)**
- CPU: **AMD Ryzen 7 4800H with Radeon Graphics 2.90 GHz**
- RAM: **16 GB**

The training has **a maximum batch size of 5** with hardware above, and takes 4.61 GB of the VRAM during the training process. The whole training process would last **approximate 27 hours** (8 minutes for each epoch with batch size 5, 200 epoches, yolov5m) to finish the training process of the model with data augmentation.


