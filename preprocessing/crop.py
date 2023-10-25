import os
import shutil
import cv2
import numpy as np

# source folder, containing original building images need to be cropped
folder_path = 'undistorted_images/'

# result folder, containing cropped window images
output_folder = 'cropped_images'

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]

# batch processing
for original_image_path in image_paths:
    filename = os.path.splitext(os.path.basename(original_image_path))[0]
    img = cv2.imread(original_image_path)

    # we assume the cameras are not moved, so we can use one location text for all the images from the same dingcam

    if "dingcam02" in filename:

        # txt obtained from localize.py
        txt_path = 'image_dingcam02_2023-05-16_06-59-59_undistorted.txt'
        if not os.path.exists(txt_path):
            print(f"No txt file found for {original_image_path}!")
            continue

    if "dingcam03" in filename:

        # txt obtained from localize.py
        txt_path = 'image_dingcam03_2023-05-16_05-00-16_undistorted.txt'
        if not os.path.exists(txt_path):
            print(f"No txt file found for {original_image_path}!")
            continue

    # read coordinates for each window
    with open(txt_path, 'r') as f:
        index = 0
        for line in f:
            parts = line.strip().split(' ')
            coord_upper_left = eval(parts[1])
            coord2_upper_right = eval(parts[2])
            coord3_lower_right = eval(parts[3])
            coord4_lower_left = eval(parts[4])

            # Coordinates of the four points in the source image: top-left, top-right, bottom-right, and bottom-left
            src_points = np.float32([np.float32(coord_upper_left), np.float32(coord2_upper_right),
                                     np.float32(coord3_lower_right), np.float32(coord4_lower_left)])

            # Coordinates of the four points in the destination image
            dst_points = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]])

            # Compute the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_points, dst_points)

            # Apply the perspective transformation to the source image
            dst_img = cv2.warpPerspective(img, M, (640, 640), flags=cv2.INTER_NEAREST)

            cv2.imwrite(f"cropped_images/{filename}_{index}.png", dst_img)

            index += 1
