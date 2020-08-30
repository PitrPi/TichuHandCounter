import imutils
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("sample/camera001.jpg")
if img.shape[0] < img.shape[1]:
    img = cv2.rotate(img, "ROTATE_90_CLOCKWISE")
img_small = cv2.resize(img, (int(img.shape[0]*800/img.shape[1]), 800))
edged = cv2.Canny(img_small, 50, 200)
for i in range(10):
    lines = cv2.HoughLinesP(edged, 1, np.pi/180, 15, minLineLength=i*10, maxLineGap=i*1)
    if lines is not None:
        print(str(i) + str(lines.shape[0]))



# Take longest edge
plt.imshow(img_small, aspect="auto")
for line in lines:
    x1, y1, x2, y2 = line[0]
    plt.plot([x1, x2], [y1, y2], 'k-')
plt.show()

plt.imshow(img, aspect="auto")
plt.show()
plt.imshow(edged, aspect="auto")
plt.show()

matches = v2.matchTemplate(img, template)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)

grey = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
plt.imshow(grey, aspect="auto")
plt.show()
bw = cv2.threshold(grey, 180, 255, cv2.THRESH_BINARY)[1]
plt.imshow(bw[470:530, 50:80], aspect="auto")
plt.show()
edged = cv2.Canny(bw, 50, 200)
for i in range(10):
    lines = cv2.HoughLinesP(edged, 1, np.pi/180, 15, minLineLength=i*10, maxLineGap=i*1)
    if lines is not None:
        print(str(i) + str(lines.shape[0]))



plt.imshow(img_small, aspect="auto")
for line in lines:
    x1, y1, x2, y2 = line[0]
    plt.plot([x1, x2], [y1, y2], 'k-')
plt.show()
