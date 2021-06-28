import vedo
from vedo import Plotter, Mesh
import cv2 as cv
from copy import deepcopy
import numpy





faceMesh = Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)
# Width of my face in real world (measured)
referenceFaceWidth = 14 #cm
# Distance from the camera in the reference image (measured)
referenceDistance = 53 #cm
faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")







def drawBox(img, boundingBox):
    x, y, w, h = int(boundingBox[0]), int(boundingBox[1]), int(boundingBox[2]), int(boundingBox[3])
    cv.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    cv.putText(img, "Tracking", (75, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# I need to add a way to select only the biggest face detected.
def getFaceData(img):
    faceWidth = 0
    imgGrey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    facesDetected = faceCascade.detectMultiScale(imgGrey,1.3,5)
    for (x,y,w,h) in facesDetected:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        faceWidth = w
    return faceWidth


def getFocalLength(referenceDistance,referenceFaceWidth,faceWidthInFrame):
    focalLength = (faceWidthInFrame*referenceDistance)/referenceFaceWidth
    return focalLength


def getDistance(focalLength,referenceFaceWidth,faceWidthInFrame):
    distance = (referenceFaceWidth*focalLength)/faceWidthInFrame
    return distance


def moveFace(img,boundingBoxOld,boundingBoxNew,face):
    x_old, y_old, w_old, h_old = int(boundingBoxOld[0]), int(boundingBoxOld[1]), int(boundingBoxOld[2]), int(boundingBoxOld[3])
    x_new, y_new, w_new, h_new = int(boundingBoxNew[0]), int(boundingBoxNew[1]), int(boundingBoxNew[2]), int(boundingBoxNew[3])
    face.addPos(x_new-x_old,-y_new+y_old,0)





cap = cv.VideoCapture(0)
referenceImage = cv.imread('Models/Reference_image.png')
tracker = cv.legacy.TrackerCSRT_create()


success, img = cap.read()
referenceImage = cv.resize(referenceImage,(img.shape[1],img.shape[0]))

referenceImageFaceWidth = getFaceData(referenceImage)
focalLength = getFocalLength(referenceDistance,referenceFaceWidth,referenceImageFaceWidth)

boundingBox = cv.selectROI("Tracking", img, False)
tracker.init(img, boundingBox)

plotter_1 = Plotter(axes=dict(xtitle='x axis', ytitle='y axis', ztitle='z axis', yzGrid=False),
                    size=(img.shape[1], img.shape[0]),)
vedo.show(faceMesh, axes=1)








while True:
    timer = cv.getTickCount()
    success, img = cap.read()

    boundingBoxOld = deepcopy(boundingBox)
    success, boundingBox = tracker.update(img)
    if success:
        boundingBoxNew = deepcopy(boundingBox)
        drawBox(img, boundingBox)
        moveFace(img,boundingBoxOld,boundingBoxNew,faceMesh)
    else:
        cv.putText(img, "Object lost", (75, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

    faceWidth = getFaceData(img)
    distance = getDistance(focalLength,referenceFaceWidth,faceWidth)
    cv.putText(img, str(int(fps)), (75, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.imshow("Tracking", img)
    vedo.show(faceMesh, axes=1)

    if cv.waitKey(1) & 0xff == ord('q'):
        break


"""
while True:
    success,img = cap.read()
    faceWidth = getFaceData(img)
    cv.imshow("frame",img)
    if cv.waitKey(1) & 0xff == ord('q'):
        break
"""

vedo.plotter.closePlotter()
vedo.closeWindow()
cap.release()
cv.destroyAllWindows()

"""
ret,frame1 = cv.VideoCapture(0)
ret,frame2 = cv.VideoCapture(0)
diff = cv.absdiff(frame1,frame2)
gray = cv.cvtColor(diff,cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(5,5),0)
_ , thresh = cv.threshold(blur,20,255,cv.THRESH_BINARY)
dilated = cv.dilate(thresh,None,iterations=3)
contours, _ = cv.findContours(dilated,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(frame1,contours,-1,(0,255,0),2)
frame1 = frame2
ret,frame2 = cap.read()
# Up until now the program draws the contours of every moving object, not bounding boxes.
for contour in contours:
    (x,y,w,h) = cv.boundingRect(contour)
    if cv.contourArea(contour) < 700:
        continue
    cv.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
    
    

"""


