"""
Objective
---------
This program allows the user to select a ROI and store two images,
one with the width of the ROI, expressed in pixels, printed on it
and the other with the height of said ROI, again in pixels.
This program was designed to help the user obtain the necessary
measurements for the program Virtual_face_motion.py

Instructions
------------
How the detection works:
    This program detects movement by tracking a ROI defined by the user,
    and checking frame by frame how the position and the angle of the ROI changed.
    The user in fact will have to select a set of points that will be tracked.
    The smallest rectangle containing all the points will define the ROI
    detected by the program.

Preliminary steps:
    Before starting the program, the user has to decide the ROI
    that they will track.
    Although the user can select as many point as they want to define the ROI,
    it is suggested to choose only four points,
    which should act as the vertices of the rectangle.
    A good and easy choice for these four points are the pupils and the corners of the lips.

How to use the program:
    At the start of the program, a window will open
    showing the feed from the camera of the pc.
    The user will have to select four or more points to define a ROI.
    If the user selects by mistake a point, they can
    delete the points selected by pressing the right mouse button.
    Once the user is satisfied with their choice, they have to hold
    the 'w' key to save the image containing the height of the ROI.
    To store the image with the width instead, they have to hold the 'e' key.
    The two images will be stored in the folder called "Models".
    To stop the program, the user has to hold the 'q' key.

Needed libraries and subpackages
--------------------------------
cv2
numpy

Functions
---------
manageTrackedPoints(int, int, int, int, params)
"""
import cv2 as cv
import numpy as np



def manageTrackedPoints(event, x, y, flags, params):
    global point, pointSelected, oldPoints,faceMesh
    if event == cv.EVENT_LBUTTONDOWN:
        point = (x, y)
        pointSelected = True
        oldPoints = np.append(oldPoints, [[np.float32(x), np.float32(y)]])
        oldPoints = np.reshape(oldPoints, (int(len(oldPoints.T)/2), 2))
    if event == cv.EVENT_RBUTTONDOWN:
        oldPoints = np.array([[]], dtype=np.float32)
        pointSelected = False



cap = cv.VideoCapture(0)

lkParams = dict(winSize = (10, 10),
                maxLevel = 4, # It's the pyramid level; each level denotes a window whose size is half of the previous one.
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

pointSelected = False

oldPoints = np.array([[]],dtype=np.float32)

success, oldFrame = cap.read()
if not success:
    raise ValueError("The system failed in reading the first frame.")
oldFrame = cv.flip(oldFrame, 1)
oldFrameGray = cv.cvtColor(oldFrame, cv.COLOR_BGR2GRAY)

cv.namedWindow("Frame")
cv.setMouseCallback("Frame", manageTrackedPoints)

while True:

    success, newFrame = cap.read()
    if not success:
        raise ValueError("The system failed in reading the first frame.")
    newFrame = cv.flip(newFrame,1)
    newFrameGray = cv.cvtColor(newFrame, cv.COLOR_BGR2GRAY)

    if pointSelected == True:
        newPoints, status, error = cv.calcOpticalFlowPyrLK(oldFrameGray, newFrameGray, oldPoints, None, **lkParams)

        for point in newPoints:
            cv.circle(newFrame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

        newRectangle = cv.minAreaRect(newPoints)

        newBox = cv.boxPoints(newRectangle)
        newBox = np.intp(newBox)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv.drawContours(newFrame, [newBox], 0, (0, 255, 0))

        oldPoints = newPoints
        oldFrameGray = newFrameGray.copy()

    cv.imshow("Frame", newFrame)

    if cv.waitKey(1) & 0xff == ord('w'):
        print("Image saved")
        height = np.sqrt((newBox[0, 0] - newBox[1, 0])**2+(newBox[0, 1] - newBox[1, 1])**2)
        cv.putText(newFrame, str(height), (75, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.imwrite("Models/Reference_height_image.png",newFrame)

    if cv.waitKey(1) & 0xff == ord('e'):
        print("Image saved")
        length = np.sqrt((newBox[0, 0] - newBox[-1, 0]) ** 2 + (newBox[0, 1] - newBox[-1, 1]) ** 2)
        cv.putText(newFrame, str(length), (75, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.imwrite("Models/Reference_length_image.png",newFrame)

    if cv.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
