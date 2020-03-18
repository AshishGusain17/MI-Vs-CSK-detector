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
pts = deque(maxlen=size)

vs = cv2.VideoCapture('ball_tracking_example.mp4')
# vs = cv2.VideoCapture(0)
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
	print()

	# create a mask containing 0 and 255 values only
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	cv2.imshow('mask',imutils.resize(mask,width=250))


	(_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	if len(cnts) > 0:

		c = max(cnts, key=cv2.contourArea)
		chull = cv2.convexHull(c)
		extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
		extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
		extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
		extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

		cX = int((extreme_left[0] + extreme_right[0]) / 2)
		cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

		# find the maximum euclidean dist. between the center and the most extreme points
		distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
		radius = int(distance[distance.argmax()])
		center=(cX,cY)

		if radius > 10:
			# draw the circle outside the ball
			cv2.circle(frame, (int(cX), int(cY)), int(radius),(0, 255, 255), 2)

		# update the points queue
		pts.appendleft(center)


	# loop over the set of tracked points
	for i in range(1, len(pts)):
		thickness = int(np.sqrt(size / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	cv2.imshow("Frame", frame)


	# End the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

vs.release()
cv2.destroyAllWindows()