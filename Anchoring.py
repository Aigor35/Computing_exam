import numpy as np
import cv2 as cv


def setAnchors(image):

    cv.imshow("Anchor selection",image)
    cv.setMouseCallback("Anchor selection",mouseEvent)
    print("Select the anchors, then press 'q'")
    while True:
        cv.imshow("Anchor selection",image)
        if len(anchors):
            cv.circle(image, (anchors[-1][0],anchors[-1][1]), 3, (255, 0, 0), -1)
        if cv.waitKey(1) & 0xff == ord('q'):
            break
    cv.destroyWindow("Anchor selection")
    return anchors

def mouseEvent(event,x,y,flags,param):
    if event == cv.EVENT_RBUTTONDOWN:
        anchors.append([x,y])



cap = cv.VideoCapture(0)
anchors = []

#feature_params = dict( maxCorners = 100,
#                       qualityLevel = 0.3,
#                       minDistance = 7,
#                       blockSize = 7 )

feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )

#lk_params = dict( winSize  = (15,15),
#                  maxLevel = 2,
#                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


lk_params = dict( winSize  = (10,10),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 0.03))


success, oldFrame = cap.read()
oldFrame = cv.flip(oldFrame,1)
#anchors = setAnchors(oldFrame.copy())

oldGray = cv.cvtColor(oldFrame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(oldGray, mask = None, **feature_params)
mask = np.zeros_like(oldFrame)



while True:
    success, frame = cap.read()
    frame = cv.flip(frame,1)
    if not success:
        print("Frame capture failed")
        break


    frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1, status, err = cv.calcOpticalFlowPyrLK(oldGray, frameGray, p0, None, **lk_params)

    if p1 is not None:
        goodNew = p1[status==1]
        goodOld = p0[status==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(goodNew, goodOld)):
        a,b = new.ravel()
        c,d = old.ravel()
        #mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), (0,255,0), 2)
        frame = cv.circle(frame,(int(a),int(b)),5,(0,0,255),-1)
    img = cv.add(frame,mask)

#    for anchor in anchors:
#        cv.circle(img, (anchor[0],anchor[1]), 3, (255, 0, 0), -1)

    cv.imshow("Movement tracking",img)
    if cv.waitKey(1) & 0xff == ord('q'):
        break
    # Now update the previous frame and previous points
    oldGray = frameGray.copy()
    p0 = goodNew.reshape(-1,1,2)