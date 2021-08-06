import numpy as np
import cv2 as cv


def selectOrResetPoints(event, x, y, flags, params):
    global point, pointSelected, oldPoints
    if event == cv.EVENT_LBUTTONDOWN:
        point = (x, y)
        pointSelected = True
        oldPoints = np.append(oldPoints, [[np.float32(x), np.float32(y)]])
        oldPoints = np.reshape(oldPoints, (int(len(oldPoints.T)/2), 2))
    if event == cv.EVENT_RBUTTONDOWN:
        oldPoints = np.array([[]], dtype=np.float32)
        pointSelected = False


cap = cv.VideoCapture(0)


lkParams = dict(winSize = (10,10),
                maxLevel = 4, # It's the pyramid level; each level denotes a window whose size is half of the previous one.
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

pointSelected = False
point = ()

oldPoints = np.array([[]],dtype=np.float32)

_, oldFrame = cap.read()
oldFrameGray = cv.cvtColor(oldFrame,cv.COLOR_BGR2GRAY)

cv.namedWindow("Frame")
cv.setMouseCallback("Frame",selectOrResetPoints)


while True:

    _, newFrame = cap.read()
    newFrameGray = cv.cvtColor(newFrame,cv.COLOR_BGR2GRAY)

    if pointSelected == True:
        newPoints, status, error = cv.calcOpticalFlowPyrLK(oldFrameGray,newFrameGray,oldPoints,None,**lkParams)
        oldPoints = newPoints
        oldFrameGray = newFrameGray.copy()

        for point in oldPoints:
            cv.circle(newFrame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

        rectangle = cv.minAreaRect(oldPoints)
        box = cv.boxPoints(rectangle)
        box = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv.drawContours(newFrame, [box], 0, (0,255,0))

    cv.imshow("Frame", newFrame)

    if cv.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
