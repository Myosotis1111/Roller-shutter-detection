import cv2
import os
import numpy as np
import mapping_manual
from template_matching_windows import first_row
from template_matching_windows import second_row
from template_matching_windows import third_row

# The name of output txt file is set as the same with the corresponding image
output_txt_filename = mapping_manual.basename + '.txt'
if os.path.exists(output_txt_filename):
    os.remove(output_txt_filename)


def cv_show(img, name):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


new_img = mapping_manual.img.copy()
windows_first_row = np.array(first_row, dtype=np.float32)
windows_second_row = np.array(second_row, dtype=np.float32)
windows_third_row = np.array(third_row, dtype=np.float32)

dst_pts_first_list = []
for i in range(len(first_row)):
    dst_pts = cv2.perspectiveTransform(windows_first_row[i].reshape(-1, 1, 2), cv2.invert(mapping_manual.M)[1])
    dst_pts = dst_pts.reshape(-1, 2)
    dst_pts_first_list.append(dst_pts)
    if [dst_pts.astype(np.int32)]:
        cv2.drawContours(new_img, [dst_pts.astype(np.int32)], -1, (0, 255, 0), 2)
    else:
        print("Error: contours is empty.")

print('***********************')


for point in range(len(dst_pts_first_list)):
    pt_str = ' '.join([f"[{dst_pts_first_list[point][i][0]:.2f},{dst_pts_first_list[point][i][1]:.2f}]" for i in
                       range(len(dst_pts_first_list[point]))])
    print('upper_' + str(point + 1) + ' ' + pt_str)
    with open(output_txt_filename, "a") as f:
        f.write('upper_' + str(point + 1) + ' ' + pt_str+'\n')

dst_pts_second_list = []
for i in range(len(second_row)):
    dst_pts = cv2.perspectiveTransform(windows_second_row[i].reshape(-1, 1, 2), cv2.invert(mapping_manual.M)[1])
    dst_pts = dst_pts.reshape(-1, 2)
    dst_pts_second_list.append(dst_pts)
    if [dst_pts.astype(np.int32)]:
        cv2.drawContours(new_img, [dst_pts.astype(np.int32)], -1, (0, 255, 0), 2)
    else:
        print("Error: contours is empty.")

for point in range(len(dst_pts_second_list)):
    pt_str = ' '.join([f"[{dst_pts_second_list[point][i][0]:.2f},{dst_pts_second_list[point][i][1]:.2f}]" for i in
                       range(len(dst_pts_second_list[point]))])
    print('middle_' + str(point + 1) + ' ' + pt_str)
    with open(output_txt_filename, "a") as f:
        f.write('middle_' + str(point + 1) + ' ' + pt_str + '\n')


dst_pts_third_list = []
for i in range(len(third_row)):
    dst_pts = cv2.perspectiveTransform(windows_third_row[i].reshape(-1, 1, 2), cv2.invert(mapping_manual.M)[1])
    dst_pts = dst_pts.reshape(-1, 2)
    dst_pts_third_list.append(dst_pts)
    if [dst_pts.astype(np.int32)]:
        cv2.drawContours(new_img, [dst_pts.astype(np.int32)], -1, (0, 255, 0), 2)
    else:
        print("Error: contours is empty.")


for point in range(len(dst_pts_third_list)):
    pt_str = ' '.join([f"[{dst_pts_third_list[point][i][0]:.2f},{dst_pts_third_list[point][i][1]:.2f}]" for i in
                       range(len(dst_pts_third_list[point]))])
    print('lower_' + str(point + 1) + ' ' + pt_str)
    with open(output_txt_filename, "a") as f:
        f.write('lower_' + str(point + 1) + ' ' + pt_str + '\n')


