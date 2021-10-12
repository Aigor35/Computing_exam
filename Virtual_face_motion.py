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
    Sometimes the first selected point get misplaced by the
    tracking algorithm immediately after the selection.
    If this happens, the user has to press the right mouse button
    and reselect the points.
    This problem can happen only for the first point, and it can
    only happen immediately after the selection.
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
manageTrackedPoints(int, int, int, int, params) -> None
getFocalLength(float, float, float) -> float
getCmOverPixelsRatio(float, numpy.array) -> float
getDistance(float, float) -> float
moveFace(numpy.array, numpy.array, float, float) -> None
checkIfInsideBoundary(numpy.array, int, int) -> None
getRotationAngle(tuple, tuple) -> float
showFacePosition(vedo.Mesh) -> None
"""
import vedo
from vedo import Plotter, Mesh
import cv2 as cv
import numpy as np



class dataCluster():
    """
    The dataCluster class allows the user to generate objects representing
    clusters of data, as the name of the class suggests.
    More precisely, the instances of the class, through its arguments,
    can store information about the points selected by the user
    and about the virtual face managed by the program.

    Attributes
    ----------
    :attribute pointSelected: bool
        this attribute's value is False if no tracking point has been selected yet
        by the user, and True otherwise
    :attribute fourPointsSelected: bool
        this attribute's value is False until four points are selected,
        then its value becomes true.
    :attribute oldPoints: numpy.array object
        set of points tracked by the program.
        The type of the points' coordinates is numpy.float32
    : attribute faceMesh: vedo.Mesh object
        a mesh object with the shape of a human head.
        The object is stored inside the "Models" folder

    Methods
    -------
    """
    pointSelected = False
    fourPointsSelected = False
    oldPoints = np.array([[]], dtype=np.float32)
    faceMesh = Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)


    def getPointSelected(self):
        """
        Returns the value of the attribute pointSelected.

        :return: bool
        """
        return self.pointSelected

    def updatePointSelected(self, newValue):
        """
        Updates the value of the attribute pointSelected.

        Parameters
        ----------
        :param newValue: bool
            new value to be assigned to pointSelected

        :return:
            does not return anything.
        """
        self.pointSelected = newValue

    def getFourPointsSelected(self):
        return self.fourPointsSelected

    def updateFourPointsSelected(self, newValue):
        self.fourPointsSelected = newValue

    def getOldPoints(self):
        """
        Returns the set of currently tracked points oldPoints.

        :return: numpy.array object
        """
        return self.oldPoints

    def updateOldPoints(self, newPoint):
        """
        Adds a new point to the set of currently tracked points.

        Parameters
        ----------
        :param newPoint: numpy.array object
            new point to be added to the set.
            Its shape should be (1,2)

        :return:
            does not return anything.
        """
        self.oldPoints = np.append(self.oldPoints, [[np.float32(newPoint[0]), np.float32(newPoint[1])]])
        self.oldPoints = np.reshape(self.oldPoints, (int(len(self.oldPoints.T) / 2), 2))

    def overwriteOldPoints(self, newSetOfPoints):
        """
        Overwrites the set of currently tracked points.

        Parameters
        ----------
        :param newSetOfPoints: numpy.array object
            new set of points that will replace the old set of points.
            The type of the elements should be numpy.float32

        :return:
            does not return anything
        """
        self.oldPoints = np.copy(newSetOfPoints)

    def getFaceMesh(self):
        """
        Returns the vedo.Mesh object described by the attribute faceMesh.

        :return: vedo.Mesh object
        """
        return self.faceMesh

    def updateFaceMesh(self, newFaceMesh):
        """
        Updates the vedo.Mesh object described by the attribute faceMesh.

        Parameters
        ----------
        :param newFaceMesh: vedo.Mesh object
            updated version of the attribute faceMesh

        :return:
            does not return anything
        """
        self.faceMesh = newFaceMesh

    def getRefDetectedArea(self):
        rectangle = cv.minAreaRect(self.oldPoints)
        box = cv.boxPoints(rectangle)
        box = np.intp(box)

        length = np.sqrt((box[0, 0] - box[-1, 0]) ** 2 +
                         (box[0, 1] - box[-1, 1]) ** 2)

        height = np.sqrt((box[0, 0] - box[1, 0]) ** 2 +
                         (box[0, 1] - box[1, 1]) ** 2)

        refDetectedArea = length*height
        refDetectedArea = np.around(refDetectedArea, 1)
        return refDetectedArea

    def resetTrackingData(self):
        self.pointSelected = False
        self.oldPoints = np.array([[]], dtype=np.float32)
        self.faceMesh = Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



def manageTrackedPoints(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        params.updatePointSelected(True)
        params.updateOldPoints([x, y])
    elif event == cv.EVENT_RBUTTONDOWN:
        params.resetTrackingData()



def checkParameter(parameter):
    if parameter <= 0:
        print("The value received is lower or equal to 0, and only values "
              "greater than 0 are accepted.")
        return False
    elif (np.isnan(parameter) or np.isinf(parameter)):
        print("The value received is inf or nan, and only finite values "
              "greater than 0 are accepted.")
        return False
    return True


def getInputParameters():
    print("Welcome.")
    refLength = 0.
    refHeight = 0.
    refDistance = 0.
    isTheParameterValid = False
    while not isTheParameterValid:
        refLength = float(input("Please enter the length, in cm, of the ROI you will define: "))
        isTheParameterValid = checkParameter(refLength)
    isTheParameterValid = False
    while not isTheParameterValid:
        refHeight = float(input("Please enter the height, in cm, of the ROI you will define: "))
        isTheParameterValid = checkParameter(refHeight)
    isTheParameterValid = False
    while not isTheParameterValid:
        refDistance = float(input("Please enter your current distance, in cm, from the camera: "))
        isTheParameterValid = checkParameter(refDistance)
    return refLength, refHeight, refDistance


def getFocalLength(refDistance, refArea, refDetectedArea):
    return refDistance*np.sqrt(refDetectedArea/refArea)



def getCmOverPixelsRatio(refArea, boxPoints):
    length = np.sqrt((boxPoints[0, 0] - boxPoints[-1, 0]) ** 2 + (boxPoints[0, 1] - boxPoints[-1, 1]) ** 2)
    height = np.sqrt((boxPoints[0, 0] - boxPoints[1, 0]) ** 2 + (boxPoints[0, 1] - boxPoints[1, 1]) ** 2)
    detectedArea = length * height
    return np.sqrt(refArea/detectedArea)



def getDistance(focalLength, cmOverPixelsRatio):
    return focalLength*cmOverPixelsRatio



def moveFace(faceMesh, oldPoints, newPoints, focalLength, refArea):
    oldCentroid = np.mean(oldPoints, axis=0)
    newCentroid = np.mean(newPoints, axis=0)
    newRatio = getCmOverPixelsRatio(refArea, newPoints)

    oldDistance = getDistance(focalLength, getCmOverPixelsRatio(refArea, oldPoints))
    newDistance = getDistance(focalLength, newRatio)

    faceMesh.addPos((newCentroid[0] - oldCentroid[0]) * newRatio,
                    (-newCentroid[1] + oldCentroid[1]) * newRatio,
                    -newDistance + oldDistance)
    return faceMesh



def checkIfInsideBoundary(clusterOfData, boxPoints, windowWidth, windowLength):
    if (np.isnan(np.sum(boxPoints)) or np.isinf(np.sum(boxPoints))):
        raise ValueError("The newly tracked points have some coordinates equal to Nan or infinity.")
    maxBoundary = np.full((4, 2), [windowWidth, windowLength])
    minBoundary = np.full((4,2), [0,0])
    if (np.any(boxPoints >= maxBoundary) or np.any(boxPoints <= minBoundary)):
        manageTrackedPoints(event = 2, x = 0, y = 0, flags = 2, params = clusterOfData)
        print("The tracked ROI has reached the boundary and has been eliminated.\n"
              "Please select a new ROI.")
        pass



def getRotationAngle(oldRectangle, newRectangle):
    detectedAngleDifference = -newRectangle[-1]+oldRectangle[-1]
    if abs(detectedAngleDifference) > 60:
        detectedAngleDifference = 0
    return detectedAngleDifference



def showFacePosition(faceMesh):
    position = faceMesh.pos()
    x = np.around(position[0], 2)
    y = np.around(position[1], 2)
    z = np.around(position[2], 2)
    cv.putText(newFrame, "x: "+str(x), (75, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(newFrame, "y: "+str(y), (75, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(newFrame, "z: "+str(z), (75, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)





if __name__ == "__main__":

    refLength, refHeight, refDistance = getInputParameters()

    refArea = refLength * refHeight
    focalLength = 0.

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

    cluster = dataCluster()
    vedo.show(cluster.getFaceMesh(), axes=1)
    cv.setMouseCallback("Frame", manageTrackedPoints, cluster)

    while True:

        success, newFrame = cap.read()
        if not success:
            raise ValueError("The system failed in reading the next frame.")

        newFrame = cv.flip(newFrame, 1)
        newFrameGray = cv.cvtColor(newFrame, cv.COLOR_BGR2GRAY)

        oldPoints = cluster.getOldPoints()
        faceMesh = cluster.getFaceMesh()

        if cluster.getPointSelected():
            newPoints, status, error = cv.calcOpticalFlowPyrLK(oldFrameGray, newFrameGray, oldPoints, None, **lkParams)

            for point in newPoints:
                cv.circle(newFrame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

            oldRectangle = cv.minAreaRect(oldPoints)
            newRectangle = cv.minAreaRect(newPoints)

            newBox = cv.boxPoints(newRectangle)
            newBox = np.intp(newBox)
            cv.drawContours(newFrame, [newBox], 0, (0, 255, 0))

            if len(newPoints)>3:

                if not cluster.getFourPointsSelected():
                    cluster.updateFourPointsSelected(True)
                    print("Changed value")
                    refDetectedArea = cluster.getRefDetectedArea()
                    focalLength = getFocalLength(refDistance, refArea, refDetectedArea)

                checkIfInsideBoundary(cluster, newPoints, oldFrame.shape[1], oldFrame.shape[0])

                if not cluster.getPointSelected():
                    continue

                faceMesh = moveFace(faceMesh, oldPoints, newPoints, focalLength, refArea)
                faceMesh.rotateY(getRotationAngle(oldRectangle, newRectangle), locally=True)
                showFacePosition(faceMesh)

            cluster.overwriteOldPoints(newPoints)
            oldFrameGray = newFrameGray.copy()

        cv.imshow("Frame", newFrame)
        vedo.show(faceMesh, axes=1)

        if cv.waitKey(1) & 0xff == ord('q'):
            break


    cap.release()
    cv.destroyAllWindows()
    plotter.close()
