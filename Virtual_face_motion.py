"""
Objective
---------
This program is built to mimic the movements of a human face.
More precisely, the program is able to detect the movement of the user's face
along the x, y and z axis, and the rotation of the user's face along
the axis perpendicular to the screen.
A virtual face then mimics the user's movement.
The movement detection stops if the user's face goes outside the camera's reach
or if the user presses the right mouse button.
The program instead stops when the user presses the 'q' key.

Instructions
------------
How the detection works:
    This program detects movement by tracking a ROI defined by the user,
    and checking frame by frame how the position and the angle of the ROI changed.
    The user in fact will have to select a set of points that will be tracked.
    The smallest rectangle containing all the points will define the ROI
    detected by the program.

Preliminary steps:
    Before starting the program, the user has to decide
    the ROI and measure its dimensions in advance,
    as well as the initial distance between themselves and the camera.
    It's in fact necessary to know the effective length and width of the ROI in cm,
    the width and length as seen by the camera in pixels,
    and the distance of the user's face from the camera in cm
    at the time when these measures were taken.
    Although the user can select as many point as they want to define the ROI,
    it is suggested to choose only four points,
    which should act as the vertices of the rectangle.
    A good and easy choice for these four points are the pupils and the corners of the lips.
    These elements in fact are easy to recognize,
    and it's easy to measure their distance.
    If the user follows this strategy, the width of the ROI
    will be approximately equal to the distance between the pupils,
    and the height of the ROI will be equal to the distance
    between the top of the nose and the mouth.
    The program Capture_reference_image.py can be used to get a reference image
    and obtain immediately a measure of the ROI in pixels.
    In order to improve the accuracy of the program it is suggested
    to attach small stickers to the face in proximity of the suggested locations,
    and to select these locations as corners of the ROI.
    Due to the blinking of the eyes in fact it's very easy
    for the tracked points to slightly move from the correct position.
    The same can be said for the lips in case the user speaks or moves the mouth.

How to use the program:
    At the start of the program, the user will be asked
    to enter the width in cm of the ROI they have decided,
    the height in cm,
    the width in pixels of the ROI as seen by the camera,
    the height in pixels,
    and the distance in cm at which the measurements in pixels were taken.
    Once all five quantities are given,
    two windows will open.
    One will show the feed from the camera of the pc, the other will show
    a virtual reality containing a simulated human head.
    The user then has to select four or more points in the camera window
    by pressing the left mouse button,
    and the program will immediately start tracking them.
    Once at least four points have been selected,
    a rectangular ROI will be formed, and the program will
    move the virtual head according to its movements.
    If the user selects by mistake a point, they can
    delete the points selected by pressing the right mouse button.
    The points will also be deleted if the ROI goes outside the
    reach of the camera.
    To stop the program the user has to hold down the 'q' key.

Needed libraries and subpackages
--------------------------------
vedo
opencv-python
numpy

Needed files with their path
----------------------------
Models/STL_Head.stl

Functions
---------
manageTrackedPoints(int, int, int, int, params)
getFocalLength(float, float, float) -> float
getCmOverPixelsRatio(float, numpy.array) -> float
getDistance(float, float) -> float
moveFace(numpy.array, numpy.array, float, float)
checkIfInsideBoundary(numpy.array, int, int)
getRotationAngle(tuple, tuple) -> float
showFacePosition(vedo.Mesh)
"""
import vedo
from vedo import Plotter, Mesh
import cv2 as cv
import numpy as np



def manageTrackedPoints(event, x, y, flags, params):
    """
    Allows the user to either select and store multiple points or clear
    the stored points

    Parameters
    ----------
    :param event: int
        one of the cv::MouseEventTypes constants; it's used to describe the detected event
    :param x: int
        the x coordinate of the mouse event
    :param y: int
        the y coordinate of the mouse event
    :param flags: int
        one of the cv::MouseEventFlags constants; it's used to describe the detected event
    :param params:
        additional parameters

    :return:
        does not return anything
    """
    global pointSelected, oldPoints, faceMesh
    if event == cv.EVENT_LBUTTONDOWN:
        pointSelected = True
        oldPoints = np.append(oldPoints, [[np.float32(x), np.float32(y)]])
        oldPoints = np.reshape(oldPoints, (int(len(oldPoints.T)/2), 2))
    elif event == cv.EVENT_RBUTTONDOWN:
        pointSelected = False
        oldPoints = np.array([[]], dtype=np.float32)
        faceMesh = Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



acceptedTypes = (int, float, np.float32, np.float64)

def getFocalLength(refDistance, refArea, refDetectedArea):
    """
    Calculates the focal length of the camera used.

    Parameters
    ----------
    :param refDistance: float
        reference distance between the user and the camera in cm
    :param refArea: float
        effective area of the ROI selected in the reference image expressed in cm^2
    :param refDetectedArea: float
        area of the ROI as seen by the camera in the reference image expressed in pixels^2

    :return: float
        the focal length of the camera expressed in pixels

    Exceptions and errors
    ---------------------
    :raises:
        ValueError: if one of arguments is negative, nan or infinite
    :raises:
        ZeroDivisionError: if the argument refArea is equal to zero
    :raises:
        TypeError: if one of the argument isn't int, float, np.float32 or np.float64

    """
    if (refDistance < 0 or refArea < 0 or refDetectedArea < 0):
        raise ValueError("One of the reference values is negative.")
    elif refArea == 0:
        raise ZeroDivisionError("The reference area cannot be equal to zero.")
    elif (np.isnan(refDistance) or np.isnan(refArea) or np.isnan(refDetectedArea)):
        raise ValueError("Nan cannot be a reference value.")
    elif (np.isinf(refDistance) or np.isinf(refArea) or np.isinf(refDetectedArea)):
        raise ValueError("Infinity cannot be a reference value.")
    elif (not type(refDistance) in acceptedTypes or not type(refArea) in acceptedTypes or not type(refDetectedArea) in acceptedTypes):
        raise TypeError("Only numerical values are accepted as reference.")
    return refDistance*np.sqrt(refDetectedArea/refArea)



def getCmOverPixelsRatio(refArea, boxPoints):
    """
    Calculates the ratio between the area of the ROI as seen in the frame in pixels^2 and
    the effective area of the ROI in cm^2.
    This ratio will be used to change unit of measure from pixels to cm.

    Parameters
    ----------
    :param refArea: float
        effective area of the ROI selected in the reference image expressed in cm^2
    :param boxPoints: numpy.array object
        set of four points that defines the corners of the ROI;
        its shape is (4,2) and the type of the elements is numpy.float32

    :return: float
        the square root of the ratio between areas, expressed in cm/pixels

    Exceptions and errors
    ---------------------
    :raises:
        ValueError: if the array boxPoints is empty or contains nan or infinite values.
    """
    if boxPoints.size == 0:
        raise ValueError("An empty array of points was passed to the function.")
    elif (np.isnan(np.sum(boxPoints)) or np.isinf(np.sum(boxPoints))):
        raise ValueError("One of the points passed to the function is nan or infinite.")
    length = np.sqrt((boxPoints[0, 0] - boxPoints[-1, 0]) ** 2 + (boxPoints[0, 1] - boxPoints[-1, 1]) ** 2)
    height = np.sqrt((boxPoints[0, 0] - boxPoints[1, 0]) ** 2 + (boxPoints[0, 1] - boxPoints[1, 1]) ** 2)
    detectedArea = length * height
    return np.sqrt(refArea/detectedArea)



def getDistance(focalLength, cmOverPixelsRatio):
    """
    Calculates the distance between the object tracked through the ROI and the camera.

    Parameters
    ----------
    :param focalLength: float
        the focal length of the camera expressed in pixels
    :param cmOverPixelsRatio: float
        the ratio used to convert pixels to cm

    :return: float
        the distance between object and camera
    """
    return focalLength*cmOverPixelsRatio



def moveFace(oldPoints, newPoints, focalLength, refArea):
    """
    Moves the vedo.Mesh object, called faceMesh, according to the
    movement detected by the camera.

    Parameters
    ----------
    :param oldPoints: numpy.array object
        set of four points that defines the corners of the ROI in the previous frame;
        its shape is (4,2) and the type of the elements is numpy.float32
    :param newPoints: numpy.array object
        set of four points that defines the corners of the ROI in the current frame;
        its shape is (4,2) and the type of the elements is numpy.float32
    :param focalLength: float
        the focal length of the camera expressed in pixels
    :param refArea: float
        effective area of the ROI selected in the reference image expressed in cm^2

    :return:
        does not return anything
    """
    global faceMesh
    oldCentroid = np.mean(oldPoints, axis=0)
    newCentroid = np.mean(newPoints, axis=0)
    newRatio = getCmOverPixelsRatio(refArea, newPoints)

    oldDistance = getDistance(focalLength, getCmOverPixelsRatio(refArea, oldPoints))
    newDistance = getDistance(focalLength, newRatio)

    faceMesh.addPos((newCentroid[0] - oldCentroid[0]) * newRatio,
                    (-newCentroid[1] + oldCentroid[1]) * newRatio,
                    -newDistance + oldDistance)



def checkIfInsideBoundary(boxPoints, windowWidth, windowLength):
    """
    Checks if the tracked ROI is inside the frame window.

    Parameters
    ----------
    :param boxPoints: numpy.array object
        set of four points that defines the corners of the ROI;
        its shape is (4,2) and the type of the elements is numpy.float32
    :param windowWidth: int
        width of the frame window
    :param windowLength: int
        length of the frame window

    :return:
        does not return anything

    Exceptions and errors
    ---------------------
    :raises:
        ValueError: if the array boxPoints contains nan or infinite values.
    """
    if (np.isnan(np.sum(boxPoints)) or np.isinf(np.sum(boxPoints))):
        raise ValueError("The newly tracked points have some coordinates equal to Nan or infinity.")
    maxBoundary = np.full((4, 2), [windowWidth, windowLength])
    minBoundary = np.full((4,2), [0,0])
    if (np.any(boxPoints >= maxBoundary) or np.any(boxPoints <= minBoundary)):
        manageTrackedPoints(event = 2, x = 0, y = 0, flags = 2, params = None)
        print("The tracked ROI has reached the boundary and has been eliminated.\n"
              "Please select a new ROI.")
        pass



def getRotationAngle(oldRectangle, newRectangle):
    """
    Calculates the difference between the angle of rotation of the tracked
    ROI in the previous frame and the angle of the ROI in the current frame.

    Parameters
    ----------
    :param oldRectangle: tuple
        a tuple of three elements describing the ROI in the previous frame as a rectangle.
        The first element is the tuple of floats (x, y) describing the center of mass of the rectangle.
        The second element is the tuple of floats (width, height) describing the width and the height of the rectangle.
        The third element is the float describing the rotation angle in a clockwise direction.
        More information about the angle value can be found in the additional notes.
    :param newRectangle: tuple
        a tuple of three elements describing the ROI in the current frame as a rectangle.
        The first element is the tuple of floats (x, y) describing the center of mass of the rectangle.
        The second element is the tuple of floats (width, height) describing the width and the height of the rectangle.
        The third element is the float describing the rotation angle in a clockwise direction.
        More information about the angle value can be found in the additional notes.

    :return: float
        the detected rotation angle

    Additional notes
    ----------------
    This function assumes that the objects oldRectangle and newRectangle are the results of the
    function cv2.minAreaRect(), therefore it's built to expect and work with angles
    defined in the same way as the angles returned by cv2.minAreaRect.
    More precisely, the function cv2.minAreaRect() takes the four corners of a rectangle and
    orders them starting from the point with the highest y, then proceeding clockwise.
    It considers then the line that connects the first and the last point, and an horizontal line.
    The angle between these two lines is the third element of the tuple returned by cv2.minAreaRect().
    If two points have the same highest y, then the rightmost point is the starting point.
    It follows that the angle value always lies in the range [-90,0) and that
    if the rectangle changes inclination than the angle value could suddenly fall from 0 to -90.
    For this reason, this function rejects difference values higher than a specific threshold.
    A threshold equal to 60 has been chosen since it rejects well all the jumps in value and keeps
    the valid rotation angle differences.
    """
    global faceMesh
    detectedAngleDifference = -newRectangle[-1]+oldRectangle[-1]
    if abs(detectedAngleDifference) > 60:
        detectedAngleDifference = 0
    return detectedAngleDifference



def showFacePosition(faceMesh):
    """
    Prints the position of the vedo.Mesh object called faceMesh on the frame window

    Parameters
    ----------
    :param faceMesh: vedo.Mesh object
        a mesh object with the shape of a human head
    :return:
        does not return anything
    """
    position = faceMesh.pos()
    x = np.around(position[0], 2)
    y = np.around(position[1], 2)
    z = np.around(position[2], 2)
    cv.putText(newFrame, "x: "+str(x), (75, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(newFrame, "y: "+str(y), (75, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(newFrame, "z: "+str(z), (75, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)






pointSelected = False
oldPoints = np.array([[]],dtype=np.float32)
faceMesh = Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)

if __name__ == "__main__":

    print("Welcome.")
    refLength = float(input("Please enter the length, in cm, of the ROI you will define: "))
    refHeight = float(input("Please enter the height, in cm, of the ROI you will define: "))
    refDetectedLength = float(input("Please enter the length, in pixels, of the ROI as seen by the camera: "))
    refDetectedHeight = float(input("Please enter the height, in pixels, of the ROI as seen by the camera: "))
    refDistance = float(input("Please enter the distance, in cm, to which the measurements in pixels refer: "))

    refArea = refLength * refHeight
    refDetectedArea = refDetectedLength * refDetectedHeight

    focalLength = getFocalLength(refDistance, refArea, refDetectedArea)

    cap = cv.VideoCapture(0)

    lkParams = dict(winSize = (10,10),
                    maxLevel = 4, # It's the pyramid level; each level denotes a window whose size is half of the previous one.
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    success, oldFrame = cap.read()

    if not success:
        raise ValueError("The system failed in reading the first frame.")

    oldFrame = cv.flip(oldFrame, 1)
    oldFrameGray = cv.cvtColor(oldFrame, cv.COLOR_BGR2GRAY)

    plotter = Plotter(axes=dict(xtitle='x axis', ytitle='y axis', ztitle='z axis', yzGrid=False),
                        size=(oldFrame.shape[1], oldFrame.shape[0]),interactive=False)
    cv.namedWindow("Frame")
    cv.setMouseCallback("Frame", manageTrackedPoints)
    vedo.show(faceMesh, axes=1)

    while True:

        success, newFrame = cap.read()
        if not success:
            raise ValueError("The system failed in reading the next frame.")

        newFrame = cv.flip(newFrame, 1)
        newFrameGray = cv.cvtColor(newFrame, cv.COLOR_BGR2GRAY)

        if pointSelected == True:
            newPoints, status, error = cv.calcOpticalFlowPyrLK(oldFrameGray, newFrameGray, oldPoints, None, **lkParams)

            for point in newPoints:
                cv.circle(newFrame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

            oldRectangle = cv.minAreaRect(oldPoints)
            newRectangle = cv.minAreaRect(newPoints)

            newBox = cv.boxPoints(newRectangle)
            newBox = np.intp(newBox)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
            cv.drawContours(newFrame, [newBox], 0, (0, 255, 0))

            if len(newPoints)>3:

                checkIfInsideBoundary(newPoints, oldFrame.shape[1], oldFrame.shape[0])

                if pointSelected == False:
                    continue

                moveFace(oldPoints, newPoints, focalLength, refArea)
                faceMesh.rotateY(getRotationAngle(oldRectangle, newRectangle), locally=True)
                showFacePosition(faceMesh)

            oldPoints = newPoints
            oldFrameGray = newFrameGray.copy()

        cv.imshow("Frame", newFrame)
        vedo.show(faceMesh, axes=1)

        if cv.waitKey(1) & 0xff == ord('q'):
            break


    cap.release()
    cv.destroyAllWindows()
    vedo.plotter.closePlotter()
    vedo.closeWindow()






