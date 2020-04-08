from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from sklearn.metrics import pairwise


size=64
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

yellowLower = (14, 76, 6)
yellowUpper = (34, 255, 255)

blueLower = (109 , 50 , 6)
blueUpper = (130 , 255, 255)

# vs = cv2.VideoCapture('./videos/ball_tracking_example.mp4')
vs = cv2.VideoCapture(0)
time.sleep(2.0)


while True:
	(grabbed, frame) = vs.read()
	print('frame',frame.shape)

	# resize the frame, blur it, and convert it to the HSV color space
	frame = imutils.resize(frame, width=600)
	print('frame_resize',frame.shape)

	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	print('blurred',blurred.shape)

	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	print('hsv',hsv.shape)

	# create a mask containing 0 and 255 values only
	# mask = cv2.inRange(hsv, blueLower, blueUpper)
	mask = cv2.inRange(hsv, yellowLower, yellowUpper)
	# mask = cv2.inRange(hsv, greenLower, greenUpper)

	print("mask",mask.shape)
	print()

	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	cv2.imshow('mask',imutils.resize(mask,width=250))


	(_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)




	cv2.imshow("Frame", frame)


	# End the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

vs.release()
cv2.destroyAllWindows()








