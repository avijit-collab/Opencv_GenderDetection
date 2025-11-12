# Project - Face Detection - OpenCV - Live Video -  OPEN VIDEO FILE


# Import OpenCV
import cv2

# Open the video from script: laptop_camera_connect.py

cap = cv2.VideoCapture(0)

# In case i have wrote the path wrong

if cap.isOpened() == False:
    print('Error!Check your file path')

# Action!
while cap.isOpened():

    ret, frame = cap.read()

    if ret == True:

        # Show it
        cv.imshow('frame', frame)

    # 27 == esc button.
    if cv2.waitKey(15) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()



