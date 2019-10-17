#
# basic_vision.py - file for testing vision
# Project - makerspace Poppy robot
#
# Author: Ilke Dincer
# Revisions:       17/10/19 - initial version
#

# do normal vision tracking, combine with poppy movements for motion when seeing someone
#
import cv2
import numpy as np

# https://pythonprogramming.net/loading-video-python-opencv-tutorial/?completed=/loading-images-python-opencv-tutorial/

cap = cv2.VideoCapture(0)
# ----- writing video to file
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# out = cv2.VideoWriter("output.avi", fourcc, 20, (640, 480))

isOpen = True


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while (isOpen):
    
    if cv2.waitKey(1) & 0xFF == ord("q"): # window closes if 'q' is pressed
        isOpen = False
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # out.write(frame) # writes to file
    faces = faceCascade.detectMultiScale(gray)
                                        # scaleFactor=1.1, 
                                        # minNeighbors=5, 
                                        # minSize=(30,30))
    
    print("Found {0} faces".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

cap.release()
# out.release() # releases the output file
cv2.destroyAllWindows()