import vedo
from vedo import Plotter, Mesh
import cv2 as cv
from copy import deepcopy
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
        faceMesh = Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)


def findCentroid(arrayOfPoints):
    mean = np.mean(arrayOfPoints,axis=0)
    return mean


def moveFace(oldPoints,newPoints):
    global faceMesh
    oldCentroid = findCentroid(oldPoints)
    newCentroid = findCentroid(newPoints)
    faceMesh.addPos(int(newCentroid[0]-oldCentroid[0]),int(-newCentroid[1]+oldCentroid[1]))
    print(faceMesh.pos())

# I think I can use faceMesh.rotateX().rotateY().rotateZ()
# I'm not using radiants, so rad = False (it's the default value)
def rotateFace(oldRectangle,newRectangle):
    global faceMesh
    detectedAngleDifference = -newRectangle[-1]+oldRectangle[-1]
    if abs(detectedAngleDifference) > 60:
        detectedAngleDifference = 0
    faceMesh.rotateY(detectedAngleDifference,locally=True)


def getFocalLength(referenceDistance,referenceFaceWidth,faceWidthInFrame):
    focalLength = (faceWidthInFrame*referenceDistance)/referenceFaceWidth
    return focalLength


def getDistance(focalLength,referenceFaceWidth,faceWidthInFrame):
    distance = (referenceFaceWidth*focalLength)/faceWidthInFrame
    return distance





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

        moveFace(oldPoints, newPoints)
        if len(newPoints)>3:
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





