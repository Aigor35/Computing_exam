import vedo
from vedo import Plotter, Mesh
import cv2 as cv
import numpy as np



# TO DO: Aggiungere gestione bordi
# TO DO: Migliorare l'acquisizione del primo punto
# TO DO: Eliminare il print delle coordinate in moveFace



def manageTrackedPoints(event, x, y, flags, params):
    global pointSelected, oldPoints, faceMesh
    if event == cv.EVENT_LBUTTONDOWN:
        pointSelected = True
        oldPoints = np.append(oldPoints, [[np.float32(x), np.float32(y)]])
        oldPoints = np.reshape(oldPoints, (int(len(oldPoints.T)/2), 2))
    elif event == cv.EVENT_RBUTTONDOWN:
        pointSelected = False
        oldPoints = np.array([[]], dtype=np.float32)
        faceMesh = Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)


"""
# refDistance and refLength are in cm, refDetectedLength is in pixels
def getFocalLength(refDistance,refLength,refDetectedLength):
    focalLength = refDistance*(refDetectedLength/refLength)
    return focalLength

# refLength is in cm, focalLength and detectedLength are in pixels
def getDistance(focalLength,refLength,detectedLength):
    distance = refLength*(focalLength/detectedLength)
    return distance
"""

acceptedTypes = (int, float, np.float32, np.float64)


def getFocalLength(refDistance, refArea, refDetectedArea):
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
    if boxPoints.size == 0:
        raise ValueError("An empty array of points was passed to the function.")
    elif (np.isnan(np.sum(boxPoints)) or np.isinf(np.sum(boxPoints))):
        raise ValueError("One of the points passed to the function is nan or infinite.")
    length = np.sqrt((boxPoints[0, 0] - boxPoints[-1, 0]) ** 2 + (boxPoints[0, 1] - boxPoints[-1, 1]) ** 2)
    height = np.sqrt((boxPoints[0, 0] - boxPoints[1, 0]) ** 2 + (boxPoints[0, 1] - boxPoints[1, 1]) ** 2)
    detectedArea = length * height
    return np.sqrt(refArea/detectedArea)



# This returns np.float64
def getDistance(focalLength, cmOverPixelsRatio):
    return focalLength*cmOverPixelsRatio



def moveFace(oldPoints, newPoints, focalLength, refArea):
    global faceMesh
    if (np.isnan(np.sum(newPoints)) or np.isinf(np.sum(newPoints))):
        raise ValueError("The newly tracked points have Nan or infinite value.")
    else:
        oldCentroid = np.mean(oldPoints, axis=0)
        newCentroid = np.mean(newPoints, axis=0)
        newRatio = getCmOverPixelsRatio(refArea, newPoints)

        oldDistance = getDistance(focalLength, getCmOverPixelsRatio(refArea, oldPoints))
        newDistance = getDistance(focalLength, newRatio)

        faceMesh.addPos(int(newCentroid[0] - oldCentroid[0]) * newRatio,
                        int(-newCentroid[1] + oldCentroid[1]) * newRatio,
                        -newDistance + oldDistance)
        # print(faceMesh.pos())


def checkIfInsideBoundary(boxPoints, windowWidth, windowLength):
    maxBoundary = np.full((4, 2), [windowWidth, windowLength])
    minBoundary = np.full((4,2), [0,0])
    if (np.any(boxPoints >= maxBoundary) or np.any(boxPoints <= minBoundary)):
        manageTrackedPoints(event = 2, x = 0, y = 0, flags = 2, params = None)
        print("The tracked ROI has reached the boundary and has been eliminated.\n"
              "Please select a new ROI.")
        pass


# I think I can use faceMesh.rotateX().rotateY().rotateZ()
# I'm not using radiants, so rad = False (it's the default value)
def rotateFace(oldRectangle, newRectangle):
    global faceMesh
    detectedAngleDifference = -newRectangle[-1]+oldRectangle[-1]
    if abs(detectedAngleDifference) > 60:
        detectedAngleDifference = 0
    faceMesh.rotateY(detectedAngleDifference,locally=True)


def printFacePosition(faceMesh):
    position = faceMesh.pos()
    x = np.around(position[0],2)
    y = np.around(position[1],2)
    z = np.around(position[2],2)
    cv.putText(newFrame, "x: "+str(x), (75, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(newFrame, "y: "+str(y), (75, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(newFrame, "z: "+str(z), (75, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)




pointSelected = False
oldPoints = np.array([[]],dtype=np.float32)
faceMesh = Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)

if __name__ == "__main__":

    cap = cv.VideoCapture(0)



    lkParams = dict(winSize = (10,10),
                    maxLevel = 4, # It's the pyramid level; each level denotes a window whose size is half of the previous one.
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))



    success, oldFrame = cap.read()

    if not success:
        raise ValueError("The system failed in reading the first frame.")
    oldFrame = cv.flip(oldFrame,1)
    oldFrameGray = cv.cvtColor(oldFrame,cv.COLOR_BGR2GRAY)

    plotter = Plotter(axes=dict(xtitle='x axis', ytitle='y axis', ztitle='z axis', yzGrid=False),
                        size=(oldFrame.shape[1], oldFrame.shape[0]),interactive=False)
    cv.namedWindow("Frame")
    cv.setMouseCallback("Frame", manageTrackedPoints)
    vedo.show(faceMesh,axes=1)


    """
    refDistance = 28.5 #cm
    refLength = 6. #cm
    refHeight = 7. #cm
    refDetectedLength = 145.4 #pixels
    refDetectedHeight = 161.3 #pixels
    """

    refDistance = 58 #cm
    refLength = 5. #cm
    refHeight = 9. #cm
    refDetectedLength = 59.0 #pixels
    refDetectedHeight = 107.0 #pixels


    refArea = refLength*refHeight
    refDetectedArea = refDetectedLength*refDetectedHeight

    focalLength = getFocalLength(refDistance, refArea, refDetectedArea)


    while True:

        success, newFrame = cap.read()
        if not success:
            raise ValueError("The system failed in reading the next frame.")

        newFrame = cv.flip(newFrame,1)
        newFrameGray = cv.cvtColor(newFrame,cv.COLOR_BGR2GRAY)

        if pointSelected == True:
            newPoints, status, error = cv.calcOpticalFlowPyrLK(oldFrameGray,newFrameGray,oldPoints,None,**lkParams)

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
                rotateFace(oldRectangle,newRectangle)
                printFacePosition(faceMesh)

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






