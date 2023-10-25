import cv2
import numpy as np


def cv_show(img, name):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


img = cv2.imread('template.png')

# mask ************************************
rows, cols = img.shape[:2]
pt_tl = [cols * 0.02, rows * 0.02]
pt_tr = [cols * 0.84, rows * 0.02]
pt_br = [cols * 0.84, rows * 0.48]
pt_bl = [cols * 0.02, rows * 0.48]

width = int((cols * 0.84 - cols * 0.02))
height = int((rows * 0.48 - rows * 0.02))
x_start = int(cols * 0.02)  # 计算截取区域的左上角 x 坐标
y_start = int(rows * 0.02)  # 计算截取区域的左上角 y 坐标
x_end = x_start + width  # 计算截取区域的右下角 x 坐标
y_end = y_start + height  # 计算截取区域的右下角 y 坐标
img = img[y_start:y_end, x_start:x_end]  # 截取指定区域

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# template 1a *********************************************************
windows_1a = cv2.imread('windows_template/window1a.png', 0)
h_1a, w_1a = windows_1a.shape[:2]

res = cv2.matchTemplate(img_gray, windows_1a, cv2.TM_CCOEFF_NORMED)
list_window1a = []
threshold = 0.71
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + w_1a, pt[1] + h_1a)
    cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 2)
    list_window1a.append([pt, (pt[0] + w_1a, pt[1]),
                         bottom_right, (pt[0], pt[1] + h_1a)])

# print(list_window1a)
# print(len(list_window1a))

# template 1a1 *********************************************************
windows_1a1 = cv2.imread('windows_template/window1a1.png', 0)
h_1a1, w_1a1 = windows_1a.shape[:2]

res = cv2.matchTemplate(img_gray, windows_1a1, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w_1a1, top_left[1] + h_1a1)
cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

list_window1a1 = [top_left, (top_left[0] + w_1a1, top_left[1]),
                  bottom_right, (top_left[0], top_left[1] + h_1a1)]

# print(list_window1a1)

# template 1a2 *********************************************************
windows_1a2 = cv2.imread('windows_template/window1a2.png', 0)
h_1a2, w_1a2 = windows_1a.shape[:2]

res = cv2.matchTemplate(img_gray, windows_1a2, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w_1a2, top_left[1] + h_1a2)
cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

list_window1a2 = [top_left, (top_left[0] + w_1a2, top_left[1]),
                  bottom_right, (top_left[0], top_left[1] + h_1a2)]

# print(list_window1a2)

# template 1b ***********************************************************
windows_1b = cv2.imread('windows_template/window1b.png', 0)
h_1b, w_1b = windows_1b.shape[:2]

res = cv2.matchTemplate(img_gray, windows_1b, cv2.TM_CCOEFF_NORMED)
list_window1b = []
threshold = 0.6
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + w_1b, pt[1] + h_1b)
    cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 2)
    list_window1b.append([pt, (pt[0] + w_1b, pt[1]),
                         bottom_right, (pt[0], pt[1] + h_1b)])


#  template 1b1 ***********************************************************
windows_1b1 = cv2.imread('windows_template/window1b1.png', 0)
h_1b1, w_1b1 = windows_1b1.shape[:2]

res = cv2.matchTemplate(img_gray, windows_1b1, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w_1b1, top_left[1] + h_1b1)
cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

list_window1b1 = [top_left, (top_left[0] + w_1b1, top_left[1]),
                  bottom_right, (top_left[0], top_left[1] + h_1b1)]

# print(list_window1b1)

#  template 1e ***********************************************************
windows_1e = cv2.imread('windows_template/window1e.png', 0)
h_1e, w_1e = windows_1e.shape[:2]

res = cv2.matchTemplate(img_gray, windows_1e, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w_1e, top_left[1] + h_1e)
cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

list_window1e = [top_left, (top_left[0] + w_1e, top_left[1]),
                 bottom_right, (top_left[0], top_left[1] + h_1e)]

#  template 1f ***********************************************************
windows_1f = cv2.imread('windows_template/window1f.png', 0)
h_1f, w_1f = windows_1f.shape[:2]

res = cv2.matchTemplate(img_gray, windows_1f, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w_1f, top_left[1] + h_1f)
cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

list_window1f = [top_left, (top_left[0] + w_1f, top_left[1]),
                 bottom_right, (top_left[0], top_left[1] + h_1f)]

#  template 1g ***********************************************************
windows_1g = cv2.imread('windows_template/window1g.png', 0)
h_1g, w_1g = windows_1g.shape[:2]

res = cv2.matchTemplate(img_gray, windows_1g, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w_1g, top_left[1] + h_1g)
cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

list_window1g = [top_left, (top_left[0] + w_1g, top_left[1]),
                 bottom_right, (top_left[0], top_left[1] + h_1g)]


#  template 2a ***********************************************************
windows_2a = cv2.imread('windows_template/window2a.png', 0)
h_2a, w_2a = windows_2a.shape[:2]

res = cv2.matchTemplate(img_gray, windows_2a, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w_2a, top_left[1] + h_2a)
cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

list_window2a = [top_left, (top_left[0] + w_2a, top_left[1]),
                 bottom_right, (top_left[0], top_left[1] + h_2a)]


# template 2b ***********************************************************
windows_2b = cv2.imread('windows_template/window2b.png', 0)
h_2b, w_2b = windows_2b.shape[:2]

res = cv2.matchTemplate(img_gray, windows_2b, cv2.TM_CCOEFF_NORMED)
list_window2b = []
threshold = 0.76
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + w_2b, pt[1] + h_2b)
    cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 2)
    list_window2b.append([pt, (pt[0] + w_2b, pt[1]),
                          bottom_right, (pt[0], pt[1] + h_2b)])


# template 2c ***********************************************************
windows_2c = cv2.imread('windows_template/window2c.png', 0)
h_2c, w_2c = windows_2c.shape[:2]

res = cv2.matchTemplate(img_gray, windows_2c, cv2.TM_CCOEFF_NORMED)
list_window2c = []
threshold = 0.71
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + w_2c, pt[1] + h_2c)
    cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 2)
    list_window2c.append([pt, (pt[0] + w_2c, pt[1]),
                          bottom_right, (pt[0], pt[1] + h_2c)])


# print(list_window2c)

# template 2d ***********************************************************
windows_2d = cv2.imread('windows_template/window2d.png', 0)
h_2d, w_2d = windows_2d.shape[:2]

res = cv2.matchTemplate(img_gray, windows_2d, cv2.TM_CCOEFF_NORMED)
list_window2d = []
threshold = 0.8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + w_2d, pt[1] + h_2d)
    cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 2)
    list_window2d.append([pt, (pt[0] + w_2d, pt[1]),
                          bottom_right, (pt[0], pt[1] + h_2d)])

# print(list_window2d)

# template 3a ***********************************************************
windows_3a = cv2.imread('windows_template/window3a.png', 0)
h_3a, w_3a = windows_3a.shape[:2]

res = cv2.matchTemplate(img_gray, windows_3a, cv2.TM_CCOEFF_NORMED)
list_window3a = []
threshold = 0.8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + w_3a, pt[1] + h_3a)
    cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 2)
    list_window3a.append([pt, (pt[0] + w_3a, pt[1]),
                          bottom_right, (pt[0], pt[1] + h_3a)])

# print(list_window3a)

# template 3b ***********************************************************
windows_3b = cv2.imread('windows_template/window3b.png', 0)
h_3b, w_3b = windows_3b.shape[:2]

res = cv2.matchTemplate(img_gray, windows_3b, cv2.TM_CCOEFF_NORMED)
list_window3b = []
threshold = 0.8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + w_3b, pt[1] + h_3b)
    cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 2)
    list_window3b.append([pt, (pt[0] + w_3b, pt[1]),
                          bottom_right, (pt[0], pt[1] + h_3b)])

# print(list_window3b)

#  template 3c ***********************************************************
windows_3c = cv2.imread('windows_template/window3c.png', 0)
h_3c, w_3c = windows_3c.shape[:2]

res = cv2.matchTemplate(img_gray, windows_3c, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w_3c, top_left[1] + h_3c)
cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

list_window3c = [top_left, (top_left[0] + w_3c, top_left[1]),
                 bottom_right, (top_left[0], top_left[1] + h_3c)]

#  template 7 ***********************************************************
windows_7 = cv2.imread('windows_template/window7.png', 0)
h_7, w_7 = windows_7.shape[:2]

res = cv2.matchTemplate(img_gray, windows_7, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w_7, top_left[1] + h_7)
cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
cv_show(img, 'windows')

list_window7 = [top_left, (top_left[0] + w_7, top_left[1]),
                bottom_right, (top_left[0], top_left[1] + h_7)]


# order the windows in 3 rows ************************************************
first_row = [list_window2a, list_window1a[0], list_window2b[0],
             list_window1b[0], list_window1b[1], list_window1b[2],
             list_window1b[3], list_window1b[4], list_window1b[5],
             list_window1b[6], list_window1b[7], list_window1b[8],
             list_window1b[9], list_window1b[10], list_window1b[11]]

second_row = [list_window2b[1], list_window1b[12], list_window2b[2],
              list_window1a[1], list_window1a[2], list_window1a[3],
              list_window1a[4], list_window1a[5], list_window1a[6],
              list_window1a[7], list_window1b1, list_window1a[8],
              list_window1a[9], list_window1a1, list_window1a2]

third_row = [list_window2c[0], list_window7, list_window3c,
             list_window3b[0], list_window3a[0], list_window2d[0],
             list_window1g, list_window3b[1], list_window1e,
             list_window2d[1], list_window3a[1], list_window1f,
             list_window3b[2], list_window2c[1]]

