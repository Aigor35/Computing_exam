"""
Objective
---------
The program Virtual_face_motion.py contains the code required to
mimic the movement of the user's face.
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
    It's in fact necessary to know the effective height and width of the ROI in cm,
    and the distance of the user's face from the camera in cm
    when they first select the points for the ROI.
    The procedure described in the next section suggests
    to use the pupils and the corners of the lips
    as points of reference for the ROI.
    If the user follows this advice,
    the width of the ROI will be approximately equal to the distance between the pupils,
    and the height of the ROI will be equal to the distance
    between the top of the nose and the mouth.
    In alternative, in order to improve the accuracy,
    the user can attach small stickers to the face in proximity of the suggested locations,
    and to select these locations as points of reference.


How to use the program:
    At the start of the program, the user will be asked
    to enter the width of the ROI they have decided, its height,
    and the initial distance between the user's face and the camera.
    All these quantities must be expressed in cm.
    Once all three quantities are given, two windows will open.
    One will show the feed from the camera of the pc, the other will show
    a virtual reality containing a simulated human head.
    The user then has to select four or more points in the camera window
    by pressing the left mouse button,
    and the program will immediately start tracking them.
    Although the user can select more than four points,
    it's important to notice that the first four points selected will
    define the reference area in pixels^2 of the ROI,
    and therefore the first four points should be the ones
    that define the corners of the ROI.
    A good and easy choice for these four points are the pupils and the corners of the lips.
    These elements in fact are easy to recognize,
    and it's easy to measure their distance.
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



Known problems
--------------
The program is rather sensible to the blinking of the eyes
and the movements of the lips.
For this reason, in case the user selects the pupils
and the corners of the lips as points of reference,
they should avoid moving the lips as much as possible.
This problem can be solved by attaching small stickers
to the face in proximity of the suggested locations,
and to select these locations as points of reference.
Another small problems comes from the tracking algorithm itself.
Sometimes the first point selected is immediately mismatched
by the tracking algorithm.
In this case, the user can simply press the right mouse button
and select the point anew.
This problem occur only for the first point,
and only when a new ROI is selected.


Needed libraries and subpackages
--------------------------------
vedo
opencv-python
numpy

Needed files with their path
----------------------------
Models/STL_Head.stl

Classes
-------
dataCluster

Functions
---------
manageDataCluster(int, int, int, int, dataCluster) -> None
checkString(str) -> (bool, float)
getInputParameters() -> (float, float, float)
getFocalLength(float, float, float) -> float
getCmOverPixelsRatio(float, numpy.array) -> float
getDistance(float, float) -> float
moveFace(vedo.Mesh, numpy.array, numpy.array, float, float) -> vedo.Mesh
checkIfInsideBoundary(dataCluster, numpy.array, int, int) -> None
getRotationAngle(tuple, tuple) -> float
showFacePosition(vedo.Mesh) -> None
"""
import vedo
from vedo import Plotter, Mesh
import cv2 as cv
import numpy as np



class dataCluster():
    """
    The dataCluster class, as the name of the class suggests,
    allows the user to generate objects representing clusters of data.
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
    __init__(self) -> None
    getPointSelected(self) -> bool
    updatePointSelected(self, bool) -> None
    getFourPointsSelected(self) -> bool
    updateFourPointsSelected(self, bool) -> None
    getOldPoints(self) -> numpy.array
    trackOneNewPoint(self, numpy.array) -> None
    updateOldPoints(self, numpy.array) -> None
    getFaceMesh(self) -> vedo.Mesh
    updateFaceMesh(self, vedo.Mesh) -> None
    getRefDetectedArea(self) -> float
    resetTrackingData(self) -> None
    """

    def __init__(self):
        """
        Creates an instance of the dataCluster class and initializes its attributes.

        :return: None
        """
        self.pointSelected = False
        self.fourPointsSelected = False
        self.oldPoints = np.array([[]], dtype=np.float32)
        self.faceMesh = Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



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

        :return: None
        """
        self.pointSelected = newValue

    def getFourPointsSelected(self):
        """
        Returns the value of the attribute fourPointsSelected.

        :return: bool
        """
        return self.fourPointsSelected

    def updateFourPointsSelected(self, newValue):
        """
        Updates the value of the parameter fourPointsSelected.

        Parameters
        ----------
        :param newValue: bool
            new value to be assigned to fourPointsSelected

        :return: None
        """
        self.fourPointsSelected = newValue

    def getOldPoints(self):
        """
        Returns the set of currently tracked points oldPoints.

        :return: numpy.array object
        """
        return self.oldPoints

    def trackOneNewPoint(self, newPoint):
        """
        Adds a new point to the set of currently tracked points.

        Parameters
        ----------
        :param newPoint: numpy.array object
            new point to be added to the set.
            Its shape should be (2, )

        :return: None
        """
        self.oldPoints = np.append(self.oldPoints, [[np.float32(newPoint[0]), np.float32(newPoint[1])]])
        self.oldPoints = np.reshape(self.oldPoints, (int(len(self.oldPoints.T) / 2), 2))

    def updateOldPoints(self, newSetOfPoints):
        """
        Updates the set of currently tracked points.

        Parameters
        ----------
        :param newSetOfPoints: numpy.array object
            new set of points that will replace the old set of points.
            The type of the elements should be numpy.float32

        :return: None
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

        :return: None
        """
        self.faceMesh = newFaceMesh

    def getRefDetectedArea(self):
        """
        Returns the reference area that will be used through the program
        to calculate the movement along the z axis.
        This area is defined as the smallest rectangle containing the first
        four points selected by the user.

        :return: float
        """
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
        """
        Reset the values of the attributes pointSelected, oldPoints and faceMesh.

        :return: None
        """
        self.pointSelected = False
        self.oldPoints = np.array([[]], dtype=np.float32)
        self.faceMesh = Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



def manageDataCluster(event, x, y, flags, params):
    """
    Manages the information stored inside an instance of the class dataCluster
    according to the mouse inputs of the user.

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
    :param params: dataCluster
        the last argument of this function represents all parameters that the programmer
        needs to pass to the function.
        In this specific program, the passed argument is an instance of the dataCluster class.

    :return: None
    """
    if event == cv.EVENT_LBUTTONDOWN:
        params.updatePointSelected(True)
        params.trackOneNewPoint([x, y])
    elif event == cv.EVENT_RBUTTONDOWN:
        params.resetTrackingData()



def checkString(string):
    """
    Check if the string received as argument can be interpreted as a floating number,
    and if said floating number is equal to 0, negative, infinite or nan.

    Parameters
    ----------
    :param string: str
        string object representing the user's keyboard input.

    :return: bool,float
        if the received string is valid, the function returns True,float(string)
        otherwise it returns False,0
    """
    try:
        value = float(string)
        if value <= 0:
            print("The value received is lower or equal to 0, and only values "
                  "greater than 0 are accepted.")
            return False, 0
        elif (np.isnan(value) or np.isinf(value)):
            print("The value received is inf or nan, and only finite values "
                  "greater than 0 are accepted.")
            return False, 0
    except ValueError:
        print("The received input cannot be interpreted as a floating number.")
        return False, 0
    return True, value


def getInputParameters():
    """
    Asks the user for three inputs, one at a time, and sends them to the function
    checkString to make sure that they are valid.
    If they're not, the function will ask for new valid inputs.

    :return: float, float, float
        the three valid inputs received by the user
    """
    print("Welcome.")
    refLength = 0.
    refHeight = 0.
    refDistance = 0.
    isTheParameterValid = False
    while not isTheParameterValid:
        receivedInput = input("Please enter the length, in cm, of the ROI you will define: ")
        isTheParameterValid, refLength = checkString(receivedInput)
    isTheParameterValid = False
    while not isTheParameterValid:
        receivedInput = input("Please enter the height, in cm, of the ROI you will define: ")
        isTheParameterValid, refHeight = checkString(receivedInput)
    isTheParameterValid = False
    while not isTheParameterValid:
        receivedInput = input("Please enter your current distance, in cm, from the camera: ")
        isTheParameterValid, refDistance = checkString(receivedInput)
    return refLength, refHeight, refDistance


def getFocalLength(refDistance, refArea, refDetectedArea):
    """
    Calculates the focal length of the camera.

    Parameters
    ----------
    :param refDistance: float
        reference distance, in cm, between the user and the camera.
        It's the distance at which the user selects the points
        defining the ROI for the first time.
    :param refArea: float
        reference area, in cm^2, of the ROI selected by the user.
    :param refDetectedArea:
        reference area, in pixels^2, of the ROI detected by the camera
        when the user selects the points defining said ROI for
        the first time.

    :return: float
        focal length of the camera expressed in pixels
    """
    return refDistance*np.sqrt(refDetectedArea/refArea)



def getCmOverPixelsRatio(refArea, boxPoints):
    """
    Calculates the ratio between the area of the ROI detected by the camera in pixels^2
    and the effective area of the ROI in cm^2.
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
    """
    length = np.sqrt((boxPoints[0, 0] - boxPoints[-1, 0]) ** 2 + (boxPoints[0, 1] - boxPoints[-1, 1]) ** 2)
    height = np.sqrt((boxPoints[0, 0] - boxPoints[1, 0]) ** 2 + (boxPoints[0, 1] - boxPoints[1, 1]) ** 2)
    detectedArea = length * height
    return np.sqrt(refArea/detectedArea)



def getDistance(focalLength, cmOverPixelsRatio):
    """
    Calculates the distance between the user's face tracked through the ROI and the camera.

    Parameters
    ----------
    :param focalLength: float
        the focal length of the camera expressed in pixels
    :param cmOverPixelsRatio: float
        the ratio used to convert pixels to cm

    :return: float
        the distance between face and camera
    """
    return focalLength*cmOverPixelsRatio



def moveFace(faceMesh, oldPoints, newPoints, focalLength, refArea):
    """
    Moves the vedo.Mesh object, called faceMesh, according to the
    movement detected by the camera.

    Parameters
    ----------
    :param faceMesh
        the vedo.Mesh object that mimics the movements of the user's face
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

    :return: vedo.Mesh
        the object faceMesh with its position updated
    """
    oldCentroid = np.mean(oldPoints, axis=0)
    newCentroid = np.mean(newPoints, axis=0)
    newRatio = getCmOverPixelsRatio(refArea, newPoints)

    oldDistance = getDistance(focalLength, getCmOverPixelsRatio(refArea, oldPoints))
    newDistance = getDistance(focalLength, newRatio)

    faceMesh.addPos((newCentroid[0] - oldCentroid[0]) * newRatio,
                    (-newCentroid[1] + oldCentroid[1]) * newRatio,
                    -newDistance + oldDistance)
    return faceMesh



def checkIfInsideBoundary(clusterOfData, boxPoints, windowWidth, windowHeight):
    """
    Makes sure that the ROI detected by the camera is still inside the frame window.
    If the ROI touches the boundaries of the window, all information regarding
    the tracking are reset and the user has to select a new ROI.

    Parameters
    ----------
    :param clusterOfData: dataCluster
        instance of the dataCluster class containing the information regarding
        the virtual face
    :param boxPoints: numpy.array object
        set of four points that defines the corners of the ROI;
        its shape is (4,2) and the type of the elements is numpy.float32
    :param windowWidth: int
        width of the frame window
    :param windowLength: int
        length of the frame window

    :return: None
    """
    if (np.isnan(np.sum(boxPoints)) or np.isinf(np.sum(boxPoints))):
        raise ValueError("The newly tracked points have some coordinates equal to Nan or infinity.")
    maxBoundary = np.full((4, 2), [windowWidth, windowHeight])
    minBoundary = np.full((4,2), [0,0])
    if (np.any(boxPoints >= maxBoundary) or np.any(boxPoints <= minBoundary)):
        manageDataCluster(event = 2, x = 0, y = 0, flags = 2, params = clusterOfData)
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
        the vedo.Mesh object that mimics the movements of the user's face
    :return: None
    """
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

    lkParams = dict(winSize = (10, 10),
                    maxLevel = 4,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    success, oldFrame = cap.read()

    if not success:
        raise ValueError("The system failed in reading the first frame.")

    oldFrame = cv.flip(oldFrame, 1)
    oldFrameGray = cv.cvtColor(oldFrame, cv.COLOR_BGR2GRAY)

    plotter = Plotter(axes=dict(xtitle='x axis', ytitle='y axis', ztitle='z axis', yzGrid=False),
                      size=(oldFrame.shape[1], oldFrame.shape[0]), interactive=False)
    cv.namedWindow("Frame")

    cluster = dataCluster()
    vedo.show(cluster.getFaceMesh(), axes=1)
    cv.setMouseCallback("Frame", manageDataCluster, cluster)

    while True:

        success, newFrame = cap.read()
        if not success:
            raise ValueError("The system failed in reading the next frame.")

        newFrame = cv.flip(newFrame, 1)
        newFrameGray = cv.cvtColor(newFrame, cv.COLOR_BGR2GRAY)

        oldPoints = cluster.getOldPoints()
        faceMesh = cluster.getFaceMesh()

        if cluster.getPointSelected():
            newPoints, status, error = cv.calcOpticalFlowPyrLK(oldFrameGray, newFrameGray,
                                                               oldPoints, None, **lkParams)

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

                checkIfInsideBoundary(cluster, newBox, oldFrame.shape[1], oldFrame.shape[0])

                if not cluster.getPointSelected():
                    continue

                faceMesh = moveFace(faceMesh, oldPoints, newPoints, focalLength, refArea)
                faceMesh.rotateY(getRotationAngle(oldRectangle, newRectangle), locally=True)
                cluster.updateFaceMesh(faceMesh)
                showFacePosition(faceMesh)

            cluster.updateOldPoints(newPoints)
            oldFrameGray = newFrameGray.copy()

        cv.imshow("Frame", newFrame)
        vedo.show(faceMesh, axes=1)

        if cv.waitKey(1) & 0xff == ord('q'):
            break


    cap.release()
    cv.destroyAllWindows()
    plotter.close()
