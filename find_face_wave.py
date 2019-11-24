#
# basic_vision.py - file for testing vision
# Project - makerspace Poppy robot
#
# Author: Ilke Dincer
# Revisions:        17/10/19 - initial version
#                   13/11/19 - draws box around face
#                   24/11/19 - changed face finding method
#

# do normal vision tracking, combine with poppy movements for motion when seeing someone
#
import cv2
import numpy as np

################################################# Constants
TOLERANCE = 100 # in pixels

# focal length = pixel width * distance / actual width
# this variable will change depending on the camera, and is currently a 
# rough estimate using a pixel width of 100cm, distance of 60cm 
# and face width of 15cm
FACE_WIDTH = 15
FOCAL_LENGTH = (100 * 60) / FACE_WIDTH

################################################# Functions

def find_distance(pixel_width):
    """ Finds the distance away from the camera that the face is using focal 
        length. Focal length = pixel_width * distance / actual_width."""
    d = FACE_WIDTH * FOCAL_LENGTH / pixel_width
    return d

def find_angle(w, f_dist):
    """ Calculates angular distance of object using arctan, import opposite and 
        adjacent side length"""
    theta = np.arctan(w / f_dist)
    return np.degrees(theta)

def poppy_turn(facee):
    f_dist = find_distance(facee)
    theta = find_angle(facee.w, f_dist)



# def find_location(screen_middle, item_x, frane):
#     if item_x < screen_middle - TOLERANCE:
#         turn_left(frane)
#     elif screen_middle + TOLERANCE < item_x:
#         turn_right(frane)
#     else:
#         in_middle(frane)

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

# Path to openCV cascade file
faceCascade = cv2.CascadeClassifier("C:\\Users\\Ilke\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")

while (isOpen):
    
    if cv2.waitKey(1) & 0xFF == ord("q"): # window closes if 'q' is pressed
        isOpen = False
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # out.write(frame) # writes to file
    faces = faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.1, 
                                        minNeighbors=5, 
                                        minSize=(30,30))
                                        # flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    
    # print("Found {0} faces".format(len(faces)))
    faces = np.asarray(faces) # converts tuple of faces into array
    # print(faces)

    # finds the centre x coordinate for each face
    # faces_x = [ int(x + w /2) for (x, y, w, h) in faces]
    

    # draws rectangles around all faces
    for (x, y, w, h) in faces:
        # cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "width: " + str(w), 
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        # face_mid_x = int(x + w / 2)

    # if ((x_centre - ()))\
    # if face_mid_x != 0:
    #     find_location(x_centre, face_mid_x, frame)

    # poppy_turn(face)

    if len(faces): # ensures that faces is not empty
        index = np.random.randint(len(faces), size = 1)
        face = faces[index] # chooses only one face 
        face = [val for val in face[0]] # removes extra array dimension
        
        # Format of face: [x, y, w, h]
        # print(face)
        w = face[2]
        x = face[0] + 0.5 * w

        f_dist = find_distance(w)
        theta = find_angle(x - x_centre, f_dist)

        print("angle: {0}".format(theta)) 
        print("distance: {0}".format(f_dist))
        print()

        cv2.putText(frame, "angle: " + str(theta), 
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    # print("------------------------------------------------------------------/n")

cap.release()
# out.release() # releases the output file
cv2.destroyAllWindows()