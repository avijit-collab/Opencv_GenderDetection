# Project - Face Detection - OpenCV - Live Video -  LAPTOP CAMERA CONNECT

# Import Packages
import tensorflow as tf
print("Your Tensorflow version is: " + tf.__version__)

import cv2
print("Your OpenCV version is: " + cv2.__version__)

import numpy as np
print("Your Numpy version is: " + np.__version__)

import keras
print("Your Keras version is: " + keras.__version__)

# Take a video from my own camera
cap = cv2.VideoCapture(0)

# Get the Width and Height of the Video Frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video format, FPS, etc.
# Use : "DIVX" for Windows
# Use : "XVID" for Mac

output = cv2.VideoWriter('SelfVideo.mov',cv2.VideoWriter_fourcc(*'XVID'),20,(width,height))

while True:
    ret, frame = cap.read()

    # Operations
    output.write(frame)

    # For Gray Frame
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', gray)

    # For Colored Frame
    cv2.imshow('frame', frame)
    # 27 == esc button.
    if cv2.waitKey(5) & 0xFF - - 27:
        break

# Dont forget to release and destroy
cap.release()
output.release()
cv2.destroyAllWindows()




