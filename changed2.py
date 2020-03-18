import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

frame = cv2.imread("im4.jpg")
print('original',frame.shape)
cv2.imshow('original',frame)




yellowLower = (14, 76, 6)
yellowUpper = (34, 255, 255)


print('frame',frame.shape)

# resize the frame, blur it, and convert it to the HSV color space
frame = imutils.resize(frame, width=600)
print('frame_resize',frame.shape)

blurred = cv2.GaussianBlur(frame, (11, 11), 0)
print('blurred',blurred.shape)

hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
print('hsv',hsv.shape)
print()

# create a mask containing 0 and 255 values only
mask = cv2.inRange(hsv, yellowLower, yellowUpper)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
cv2.imshow('mask',imutils.resize(mask,width=250))



# mask=cv2.GaussianBlur(mask,(15,15),0)
# mask=cv2.medianBlur(mask,15)
# mask=cv2.bilateralFilter(mask,5,175,175)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# mask = cv2.filter2D(mask, -1, kernel)

# _, mask = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
# print('thres_binary',mask.shape)
# cv2.imshow('thres_binary',mask)

# mask = cv2.merge((mask, mask, mask))
# result=cv2.bitwise_and(original_image,mask)

# cv2.imshow('result',result)
# cv2.imshow('threshold',mask)

cv2.waitKey(0)
cv2.destroyAllWindows()