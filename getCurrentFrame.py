import cv2
import numpy as np
import copy
import imutils
import time
from sklearn.metrics import pairwise


def getCurrentFrame(class_ids , confidences , boxes):
	classes = []
	with open("coco.names", "r") as f:
	    classes = [line.strip() for line in f.readlines()]
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	curr_frame=[]
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			# color = colors[i]
			cx,cy = (2*x + w)/2  ,  (2*y + h)/2
			curr_frame.append([x,y,w,h,cx,cy,label])
	return curr_frame , classes