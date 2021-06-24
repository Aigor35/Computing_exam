import vedo
from vedo import Plotter, Mesh
import cv2 as cv
from copy import deepcopy
import numpy





face = Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



cap = cv.VideoCapture(0)

tracker = cv.legacy.TrackerCSRT_create()

success, img = cap.read()
boundingBox = cv.selectROI("Tracking", img, False)
tracker.init(img, boundingBox)

plotter_1 = Plotter(axes=dict(xtitle='x axis', ytitle='y axis', ztitle='z axis', yzGrid=False),
                    size=(img.shape[1], img.shape[0]),)

vedo.show(face, axes=1)


def drawBox(img, boundingBox):
    x, y, w, h = int(boundingBox[0]), int(boundingBox[1]), int(boundingBox[2]), int(boundingBox[3])
    cv.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    cv.putText(img, "Tracking", (75, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)



def moveFace(img,boundingBoxOld,boundingBoxNew,face):
    x_old, y_old, w_old, h_old = int(boundingBoxOld[0]), int(boundingBoxOld[1]), int(boundingBoxOld[2]), int(boundingBoxOld[3])
    x_new, y_new, w_new, h_new = int(boundingBoxNew[0]), int(boundingBoxNew[1]), int(boundingBoxNew[2]), int(boundingBoxNew[3])
    face.addPos(x_new-x_old,-y_new+y_old,0)


while True:
    timer = cv.getTickCount()
    success, img = cap.read()

    boundingBoxOld = deepcopy(boundingBox)
    success, boundingBox = tracker.update(img)
    if success:
        boundingBoxNew = deepcopy(boundingBox)
        drawBox(img, boundingBox)
        moveFace(img,boundingBoxOld,boundingBoxNew,face)
    else:
        cv.putText(img, "Object lost", (75, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

    cv.putText(img, str(int(fps)), (75, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.imshow("Tracking", img)
    vedo.show(face, axes=1)
    print(face.pos())

    if cv.waitKey(1) & 0xff == ord('q'):
        break

vedo.plotter.closePlotter()
vedo.closeWindow()
cv.destroyAllWindows()