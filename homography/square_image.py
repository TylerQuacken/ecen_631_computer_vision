import cv2
import numpy as np

image = cv2.imread("./images/left_screen.png")

# cv2.imshow("image", image)
# cv2.waitKey(0)

ptsSrc = np.array([[155, 40], [472, 43], [465, 415], [162, 418]])
w = 552
h = 700
ptsDst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

H, _ = cv2.findHomography(ptsSrc, ptsDst)

rectImg = cv2.warpPerspective(image, H, (w, h))

cv2.imwrite("left_screen_rect.png", rectImg)
cv2.imshow("rect", rectImg)
cv2.waitKey(0)
