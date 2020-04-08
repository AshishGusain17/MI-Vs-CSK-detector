import cv2
import numpy as np
import copy
import imutils
import time
from sklearn.metrics import pairwise


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('./videos/2.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('MI_V-s_CSK.avi', fourcc, 10.0, (int(cap.get(3)),int(cap.get(4))))

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

yellowLower = (14, 76, 6)
yellowUpper = (34, 255, 255)

blueLower = (107 , 50 , 6)
blueUpper = (132 , 255, 255)


prev_frame=[]
number=0
count=0
dharti=0
while True:
    _,img=cap.read()
    count=count+1
    if count<550:
        continue

    height, width, channels = img.shape


    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                if class_id==0:
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    curr_frame=[]

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            # color = colors[i]
            cx,cy = (2*x + w)/2  ,  (2*y + h)/2
            curr_frame.append([x,y,w,h,cx,cy,label])


    dark_mumbai=np.zeros((height,width))
    dark_chennai=np.zeros((height,width))
    for i in curr_frame:
        x,y,w,h,label = i[0] , i[1] , i[2] , i[3] , i[6] 


        print("img",img.shape)



        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        print("blurred",blurred.shape)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        print("hsv",hsv.shape)

        mask_mumbai = cv2.inRange(hsv, blueLower, blueUpper)
        mask_chennai = cv2.inRange(hsv, yellowLower, yellowUpper)
        print("mask",mask_mumbai.shape)

        mask_mumbai = cv2.erode(mask_mumbai, None, iterations=2)
        mask_mumbai = cv2.dilate(mask_mumbai, None, iterations=2)

        mask_chennai = cv2.erode(mask_chennai, None, iterations=2)
        mask_chennai = cv2.dilate(mask_chennai, None, iterations=2)     

        
        bound_box_mumbai=mask_mumbai[y:y+h,x:x+w]
        # cv2.imshow("bound_box_mumbai",bound_box_mumbai)
        bound_box_chennai=mask_chennai[y:y+h,x:x+w]
        # cv2.imshow("bound_box_chennai",bound_box_chennai)

        dark_mumbai[y:y+h,x:x+w]=bound_box_mumbai
        dark_chennai[y:y+h,x:x+w]=bound_box_chennai
        print("bound_box",bound_box_chennai.shape)


        count_mumbai  = ( (bound_box_mumbai  > 1).sum() )/(w*h)
        count_chennai = ( (bound_box_chennai > 1).sum() )/(w*h)
        print(count_mumbai,count_chennai)
        print()

        fraction=0.17
        if count_mumbai>fraction and count_mumbai>count_chennai:
            text="MI"
            # text=str(w*h) + " "+str(count_mumbai)
            cv2.rectangle(img, (x, y), (x + w, y + h), (204,0,0), 2)
            cv2.putText(img, text, (x, y + 30), font, 3, (204,0,0), 2)
        elif count_chennai>fraction and count_chennai >count_mumbai:
            text="CSK"
            # text=str(w*h)+" "+str(count_chennai)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,255), 2)
            cv2.putText(img, text, (x, y + 30), font, 3, (0,255,255), 2)
        else:
        #     text=str(w*h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,0), 2)
        #     cv2.putText(img, text, (x, y + 30), font, 3, (0,0,0), 2)
        if count_mumbai>fraction and count_chennai>fraction:
            dharti=dharti+1
        

        



    cv2.imshow('dark_mumbai',dark_mumbai)
    cv2.imshow("dark_chennai",dark_chennai)

    cv2.imshow("yolo", img)



    out1.write(img)
    key=cv2.waitKey(100)
    if key & 0xFF == ord("q"):
        break

out1.release()
print("dharti=",dharti)
cap.release()
cv2.destroyAllWindows()