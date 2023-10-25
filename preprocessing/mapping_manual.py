import os
import cv2
import numpy as np

# Define a list to store the coordinates of the clicks
img_coords_list = []
tep_coords_list = []

# path of the undistorted images need to be localized
undistorted_image_path = "image_dingcam02_2023-05-16_06-59-59_undistorted.png"

filename = os.path.basename(undistorted_image_path)
basename = os.path.splitext(filename)[0]


def cv_show(img, name):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, img)


# Define mouse click event handling functions
def get_img_coords(event, x, y, flags, param):
    """Get the coordinates of the mouse click and mark them in red"""
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        img_coords_list.append((x, y))
        cv2.imshow("image", img)


def get_tmp_coords(event, x, y, flags, param):
    # Get the parameters
    tep_coords_list = param
    """Get the coordinates of the mouse click on the second image and mark it green"""
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img2, (x, y), 5, (0, 255, 0), -1)
        tep_coords_list.append((x, y))
        cv2.imshow("template", img2)


# Open image and display
img = cv2.imread(undistorted_image_path)
cv_show(img, 'image')
# Bind mouse click events and get coordinates on click
cv2.setMouseCallback("image", get_img_coords)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Open the second image and display
img2 = cv2.imread("template.png")

rows, cols = img2.shape[:2]
pt_tl = [cols * 0.02, rows * 0.02]
pt_tr = [cols * 0.84, rows * 0.02]
pt_br = [cols * 0.84, rows * 0.48]
pt_bl = [cols * 0.02, rows * 0.48]

width = int((cols * 0.84 - cols * 0.02))
height = int((rows * 0.48 - rows * 0.02))
x_start = int(cols * 0.02)
y_start = int(rows * 0.02)
x_end = x_start + width
y_end = y_start + height
img2 = img2[y_start:y_end, x_start:x_end]


# Create a window and set the window size to the image size
cv2.namedWindow('template', cv2.WINDOW_NORMAL)
cv2.resizeWindow('template', width, height)


cv_show(img2, 'template')
# Bind the second image mouse click event and get the coordinates on click
cv2.setMouseCallback("template", get_tmp_coords, tep_coords_list)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the list of obtained coordinates
img_coords_list = np.array(img_coords_list, dtype='float32')
tep_coords_list = np.array(tep_coords_list, dtype='float32')
print("Coords of image:", img_coords_list)
print("Coords of template:", tep_coords_list)

# ********************************* Homography

h, w = img.shape[:2]
image_size = (w, h)

M = cv2.getPerspectiveTransform(img_coords_list, tep_coords_list)
# M = cv2.findHomography(tep_coords_list, img_coords_list, cv2.RANSAC, ransacReprojThreshold=4.0)
# M = cv2.findHomography(img_coords_list, tep_coords_list, cv2.RANSAC, ransacReprojThreshold=4.0)

print(M)

result = cv2.warpPerspective(img, M, (img2.shape[1], img2.shape[0]))

alpha = 0.5
beta = 1 - alpha
dst = cv2.addWeighted(img2, alpha, result, beta, 0)

cv_show(dst, 'dst')
cv2.waitKey(0)
cv2.destroyAllWindows()

