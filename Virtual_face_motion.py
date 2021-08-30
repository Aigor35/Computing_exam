import vedo
from vedo import Plotter, Mesh
import cv2 as cv
import numpy as np



# TO DO: Aggiungere gestione bordi
# TO DO: Migliorare l'acquisizione del primo punto


def manageTrackedPoints(event, x, y, flags, params):
    global pointSelected, oldPoints, faceMesh
    if event == cv.EVENT_LBUTTONDOWN:
        pointSelected = True
        print("x: ", x, "y: ", y)
        print("x float: ", np.float32(x), "y float: ", np.float32(y))
        oldPoints = np.append(oldPoints, [[np.float32(x), np.float32(y)]])
        oldPoints = np.reshape(oldPoints, (int(len(oldPoints.T)/2), 2))
    if event == cv.EVENT_RBUTTONDOWN:
        oldPoints = np.array([[]], dtype=np.float32)
        pointSelected = False
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



def getFocalLength(refDistance,refArea,refDetectedArea):
    if (refDistance < 0 or refArea < 0 or refDetectedArea < 0):
        raise ValueError("One of the reference values is negative.")
    if refArea == 0:
        raise ZeroDivisionError("The reference area cannot be equal to zero.")
    if (np.isnan(refDistance) or np.isnan(refArea) or np.isnan(refDetectedArea)):
        raise ValueError("Nan cannot be a reference value.")
    if (np.isinf(refDistance) or np.isinf(refArea) or np.isinf(refDetectedArea)):
        raise ValueError("Infinity cannot be a reference value.")
    if (not type(refDistance) in (int, float) or not type(refArea) in (int, float) or not type(refDetectedArea) in (int, float)):
        raise TypeError("Only numerical values are accepted as reference.")
    return refDistance*np.sqrt(refDetectedArea/refArea)


def getDistance(focalLength,refArea,boxPoints):
    length = np.sqrt((boxPoints[0,0]-boxPoints[-1,0])**2+(boxPoints[0,1]-boxPoints[-1,1])**2)
    height = np.sqrt((boxPoints[0,0]-boxPoints[1,0])**2+(boxPoints[0,1]-boxPoints[1,1])**2)
    detectedArea = length*height
    return focalLength*np.sqrt(refArea/detectedArea)

# TO DO: MOVE THE CAME INSTEAD OF THE FACE FOR THE Z AXIS MOVEMENT
def moveFace(oldPoints,newPoints):
    global faceMesh,focalLength,refArea
    oldCentroid = np.mean(oldPoints,axis=0)
    newCentroid = np.mean(newPoints,axis=0)
    oldDistance = getDistance(focalLength,refArea,oldPoints)
    newDistance = getDistance(focalLength,refArea,newPoints)
    faceMesh.addPos(int(newCentroid[0]-oldCentroid[0]),int(-newCentroid[1]+oldCentroid[1]),-newDistance+oldDistance)
    print(faceMesh.pos())

# I think I can use faceMesh.rotateX().rotateY().rotateZ()
# I'm not using radiants, so rad = False (it's the default value)
def rotateFace(oldRectangle,newRectangle):
    global faceMesh
    detectedAngleDifference = -newRectangle[-1]+oldRectangle[-1]
    if abs(detectedAngleDifference) > 60:
        detectedAngleDifference = 0
    faceMesh.rotateY(detectedAngleDifference,locally=True)



if __name__ == "__main__":

    cap = cv.VideoCapture(0)

    faceMesh = Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)

    lkParams = dict(winSize = (10,10),
                    maxLevel = 4, # It's the pyramid level; each level denotes a window whose size is half of the previous one.
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    pointSelected = False

    oldPoints = np.array([[]],dtype=np.float32)


    _, oldFrame = cap.read()
    oldFrame = cv.flip(oldFrame,1)
    oldFrameGray = cv.cvtColor(oldFrame,cv.COLOR_BGR2GRAY)

    plotter = Plotter(axes=dict(xtitle='x axis', ytitle='y axis', ztitle='z axis', yzGrid=False),
                        size=(oldFrame.shape[1], oldFrame.shape[0]),interactive=False)

    cv.namedWindow("Frame")
    cv.setMouseCallback("Frame",manageTrackedPoints)
    vedo.show(faceMesh,axes=1)


    # TO DO: CHANGE THESE VALUES
    refDistance = 28.5 #cm
    refLength = 6. #cm
    refHeight = 7. #cm
    refDetectedLength = 145.4 #pixels
    refDetectedHeight = 161.3 #pixels

    refArea = refLength*refHeight
    refDetectedArea = refDetectedLength*refDetectedHeight

    focalLength = getFocalLength(refDistance,refArea,refDetectedArea)


    while True:

        _, newFrame = cap.read()
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
                moveFace(oldPoints, newPoints)
                rotateFace(oldRectangle,newRectangle)

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






