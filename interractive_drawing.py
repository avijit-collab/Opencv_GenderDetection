# Project - Face Detection - OpenCV - Live Video -  Interactive Drawing #1


import cv2

# Callback Function for the Mouse, Rectangle
def draw_rectangle(event,
                   x,
                   y,
                   flags,
                   param):


    global pt1, pt2, tope_left_clicked, bottom_right_clicked

    # Create an event for Left Button Down and assigned it to event

    if event == cv2.EVENT_LBUTTONDOWN:

        # Reset Rectangle
        if top_left_clicked == True and bottom_right_clicked == True:
            # Give 0 values to pt1 & pt2

            pt1 = (0.0)
        pt2(0, 0)

        # Give False value to your Trackers
        top_left_clicked = False
        bottom_right_clicked = False

    # Check the top_left_clicked if it's False

    if top_left_clicked == False:
        pt1 = (x, y)
        top_left_clicked = True

    # Check the bottom_right_clicked if it's False

    elif bottom_right_clicked == False:
        pt2 = (x, y)
        bottom_right_clicked = True


# pt1, pt2 Global Variables

pt1 = (0, 0)
pt2 = (0, 0)

# Trackers are False
top_left_clicked = False
bottom_right_clicked = False

# Take a video

cap = cv2.VideoCapture(0)

# Create a Named Window for the Connections

cv2.namedWindow('Test')

# Set a Mouse Callback
cv2.setMouseCallback('Test', draw_rectangle)

while True:

    # Capture the frame
    ret, frame = cap.read()

    # Based on the Global Variables Draw the Frame
    if top_left_clicked == True:
        cv2.circle(frame, center=pt1, radius=5, color=(255, 0, 255), thickness=-1)

        # Draw a Rectangle on the Frame
    if top_left_clicked == True and bottom_right_clicked == True:
        cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 3)

    # Show the Frame
    cv2.imshow('frame', frame)


    if cv2.waitKey(3) & 0xFF - -27:
        break

cap.release()
cv2.destroyAllWindows()