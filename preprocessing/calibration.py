import os
import shutil
import cv2
import numpy as np

# source folder of original images
folder_path = 'original_images/'

# obtain image path of the original jpg from dingcam
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]

# clear previous calibration results
if os.path.exists("undistorted_images"):
    shutil.rmtree("undistorted_images")

# batch processing
for original_image_path in image_paths:
    basename = os.path.basename(original_image_path)
    img = cv2.imread(original_image_path)

    # calibration parameters for dingcam02
    if "dingcam02" in basename:
        camera_matrix = np.array([[2.6187847790515211e+03, 0., 1.9247825281366602e+03],
                                  [0., 2.6190662010826636e+03, 1.1240250895276588e+03],
                                  [0., 0., 1.]])

        distortion_coefficients = np.array([-6.0551270198576423e-01, 4.0605814517988853e-01,
                                            -1.2875590324980141e-04, -3.0328317729940276e-04,
                                            -1.2530184280072734e-01])

    # calibration parameters for dingcam03
    if "dingcam03" in basename:
        camera_matrix = np.array([[2.7330367475619405e+03, 0., 1.8818942467729551e+03],
                                  [0., 2.7348774707016569e+03, 1.1274349017283473e+03],
                                  [0., 0., 1.]])

        distortion_coefficients = np.array([-6.2029341264444593e-01, 4.0042950999969296e-01,
                                            -9.8063629504388256e-04, -3.1248567772822359e-05,
                                            -1.1272460463581432e-01])

    original_filename, extension = os.path.splitext(os.path.basename(original_image_path))

    h, w = img.shape[:2]
    image_size = (w, h)

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, image_size, 0, image_size)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, new_camera_matrix, image_size, cv2.CV_32FC1)

    img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    if not os.path.exists("undistorted_images"):
        os.mkdir("undistorted_images")

    # save undistorted images
    undistorted_filename = original_filename + '_undistorted.png'
    cv2.imwrite(os.path.join('undistorted_images/', undistorted_filename), img)

