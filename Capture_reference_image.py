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


lkParams = dict(winSize = (10,10),
                maxLevel = 4, # It's the pyramid level; each level denotes a window whose size is half of the previous one.
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

pointSelected = False

oldPoints = np.array([[]],dtype=np.float32)


_, oldFrame = cap.read()
oldFrame = cv.flip(oldFrame,1)
oldFrameGray = cv.cvtColor(oldFrame,cv.COLOR_BGR2GRAY)


cv.namedWindow("Frame")
cv.setMouseCallback("Frame",manageTrackedPoints)


while True:

    _, newFrame = cap.read()
    newFrame = cv.flip(newFrame,1)
    newFrameGray = cv.cvtColor(newFrame,cv.COLOR_BGR2GRAY)

    if pointSelected == True:
        newPoints, status, error = cv.calcOpticalFlowPyrLK(oldFrameGray,newFrameGray,oldPoints,None,**lkParams)

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
        height = np.sqrt((newBox[0,0]-newBox[1,0])**2+(newBox[0,1]-newBox[1,1])**2)
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


