import vedo
from vedo import Plotter, Mesh
import cv2 as cv
from copy import deepcopy
import numpy





faceMesh = Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)

# Width of my face in real world (measured)
referenceFaceWidth = 14 #cm
# Distance from the camera in the reference image (measured)
referenceDistance = 58 #cm
faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")





# Add a way to select only the biggest face
def getFaceData(img):
    faceWidth = 0
    imgGrey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    facesDetected = faceCascade.detectMultiScale(imgGrey,1.3,5)
    for (x,y,w,h) in facesDetected:
        #cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        faceWidth = w
    return faceWidth


def getFocalLength(referenceDistance,referenceFaceWidth,faceWidthInFrame):
    focalLength = (faceWidthInFrame*referenceDistance)/referenceFaceWidth
    return focalLength


def getDistance(focalLength,referenceFaceWidth,faceWidthInFrame):
    distance = (referenceFaceWidth*focalLength)/faceWidthInFrame
    return distance




def drawBox(img, boundingBox):
    x, y, w, h = int(boundingBox[0]), int(boundingBox[1]), int(boundingBox[2]), int(boundingBox[3])
    cv.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    cv.putText(img, "Tracking", (75, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def moveFace(img,boundingBoxOld,boundingBoxNew,distanceOld,distanceNew,pixelsToCmRatio,face):
    x_old, y_old, w_old, h_old = int(boundingBoxOld[0]), int(boundingBoxOld[1]), int(boundingBoxOld[2]), int(boundingBoxOld[3])
    x_new, y_new, w_new, h_new = int(boundingBoxNew[0]), int(boundingBoxNew[1]), int(boundingBoxNew[2]), int(boundingBoxNew[3])
    z_old = int(distanceOld)
    z_new = int(distanceNew)
    face.addPos((x_new-x_old)*pixelsToCmRatio,(-y_new+y_old)*pixelsToCmRatio,-(z_new-z_old))


def setAnchors(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),3,(255,0,0),-1)
        anchors.append([x,y])




cap = cv.VideoCapture(0)
#referenceImage = cv.imread('Models/Reference_image.png')
#tracker = cv.legacy.TrackerCSRT_create()
anchors = []

success, img = cap.read()
#referenceImage = cv.resize(referenceImage,(img.shape[1],img.shape[0]))

# Maybe it would be better to just measure the pixel width using paint
#referenceImageFaceWidth = getFaceData(referenceImage)
#boundingBoxReference = cv.selectROI("Face width",referenceImage,False)
#referenceImageFaceWidth = int(boundingBoxReference[2])-int(boundingBoxReference[0])
#focalLength = getFocalLength(referenceDistance,referenceFaceWidth,referenceImageFaceWidth)

"""
boundingBox = cv.selectROI("Tracking", img, False)
tracker.init(img, boundingBox)



plotter_1 = Plotter(axes=dict(xtitle='x axis', ytitle='y axis', ztitle='z axis', yzGrid=False),
                    size=(img.shape[1], img.shape[0]),interactive=False)
vedo.show(faceMesh, axes=1)

pixelsToCmRatio = referenceFaceWidth/referenceImageFaceWidth

distanceOld = referenceDistance
distanceNew = referenceDistance
count = 0
"""

while True:
    timer = cv.getTickCount()
    success, img = cap.read()

    """
    boundingBoxOld = deepcopy(boundingBox)
    success, boundingBox = tracker.update(img)

    faceWidth = getFaceData(img)
    if faceWidth != 0:
        if count == 0:
            distanceOld = getDistance(focalLength, referenceFaceWidth, faceWidth)
            distanceNew = distanceOld
            count += 1
        else:
            distanceNew = getDistance(focalLength, referenceFaceWidth, faceWidth)

    if success:
        boundingBoxNew = deepcopy(boundingBox)
        drawBox(img, boundingBox)
        moveFace(img, boundingBoxOld, boundingBoxNew, distanceOld, distanceNew, pixelsToCmRatio, faceMesh)
    else:
        cv.putText(img, "Object lost", (75, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

    cv.putText(img, str(int(fps)), (75, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.imshow("Tracking", img)
    vedo.show(faceMesh, axes=1,interactive=False)

    #print(faceMesh.pos())
    distanceOld = distanceNew
    """
    cv.imshow("Anchors",img)
    cv.setMouseCallback("Anchors",setAnchors)
    print(anchors)
    if cv.waitKey(1) & 0xff == ord('q'):
        break


vedo.plotter.closePlotter()
vedo.closeWindow()
cap.release()
cv.destroyAllWindows()




