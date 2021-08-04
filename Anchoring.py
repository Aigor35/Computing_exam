import numpy as np
import cv2 as cv


def selectPoint(event,x,y,flags,params):
    global point, pointSelected, oldPoints
    if event == cv.EVENT_LBUTTONDOWN:
        point = (x,y)
        pointSelected = True
        oldPoints = np.array([[x, y]],dtype=np.float32)


cap = cv.VideoCapture(0)
anchors = []



lkParams = dict(winSize = (10,10),
                maxLevel = 4, # It's the pyramid level; each level denotes a window whose size is half of the previous one.
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

pointSelected = False
point = ()

oldPoints = np.array([[]])

_, oldFrame = cap.read()
oldFrameGray = cv.cvtColor(oldFrame,cv.COLOR_BGR2GRAY)

cv.namedWindow("Frame")
cv.setMouseCallback("Frame",selectPoint)


while True:

    _, newFrame = cap.read()
    newFrameGray = cv.cvtColor(newFrame,cv.COLOR_BGR2GRAY)

    if pointSelected == True:
        cv.circle(newFrame,point,5,(0,0,255),-1)
        newPoints, status, error = cv.calcOpticalFlowPyrLK(oldFrameGray,newFrameGray,oldPoints,None,**lkParams)
        oldPoints = newPoints
        oldFrameGray = newFrameGray.copy()


        x, y = newPoints.ravel()
        cv.circle(newFrame, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv.imshow("Frame", newFrame)

    if cv.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
