#
# basic_vision.py - file for testing vision
# Project - makerspace Poppy robot
#
# Author: Ilke Dincer
# Revisions:        17/10/19 - initial version
#                   13/11/19 - draws box around face
#

# do normal vision tracking, combine with poppy movements for motion when seeing someone
#
import cv2
import numpy as np

################################################# Functions
TOLERANCE = 100 # in pixels


def find_location(screen_middle, item_x, frane):
    if item_x < screen_middle - TOLERANCE:
        turn_left(frane)
    elif screen_middle + TOLERANCE < item_x:
        turn_right(frane)
    else:
        in_middle(frane)

def turn_right(frane):
    h, w = frane.shape[:2]
    y = int(h / 2)
    x = int(w - 10)
    cv2.circle(frane, (x, y), 8, (0, 0, 255), -1)


def turn_left(frane):
    h, w = frane.shape[:2]
    y = int(h / 2)
    cv2.circle(frane, (10, y), 8, (0, 0, 255), -1)


def in_middle(frane):
    h, w = frane.shape[:2]
    y = int(h / 2)
    x = int(w / 2)
    cv2.circle(frane, (x, y), 8, (0, 0, 255), -1)


# https://pythonprogramming.net/loading-video-python-opencv-tutorial/?completed=/loading-images-python-opencv-tutorial/

################################################# Begin Video Capture
cap = cv2.VideoCapture(0) # begins camera at port 0

# ----- writing video to file
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# out = cv2.VideoWriter("output.avi", fourcc, 20, (640, 480))
# cap.set(15, 0.1)

x_centre = cap.get(3) / 2 # gets video width and divides by 2

isOpen = True

# faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while (isOpen):
    
    if cv2.waitKey(1) & 0xFF == ord("q"): # window closes if 'q' is pressed
        isOpen = False
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # out.write(frame) # writes to file
    faceCascade = cv2.CascadeClassifier("C:\\Users\\Ilke\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.1, 
                                        minNeighbors=5, 
                                        minSize=(30,30))
                                        # flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    
    print("Found {0} faces".format(len(faces)))

    face_mid_x = 0
    for (x, y, w, h) in faces:
        # cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_mid_x = int(x + w / 2)

    # if ((x_centre - ()))\
    if face_mid_x != 0:
        find_location(x_centre, face_mid_x, frame)
    
    cv2.imshow('frame', frame)

cap.release()
# out.release() # releases the output file
cv2.destroyAllWindows()