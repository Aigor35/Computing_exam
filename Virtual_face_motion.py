import vedo
from vedo import Plotter, Mesh
import cv2 as cv
import numpy





face = Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



cap = cv.VideoCapture(0)

tracker = cv.legacy.TrackerCSRT_create()

success, img = cap.read()
bounding_box = cv.selectROI("Tracking", img, False)
tracker.init(img, bounding_box)

plotter_1 = Plotter(axes=dict(xtitle='x axis', ytitle='y axis', ztitle='z axis', yzGrid=False),
                    size=(img.shape[1], img.shape[0]),)

vedo.show(face, axes=1)



def drawBox(img, bounding_box):
    x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
    cv.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    cv.putText(img, "Tracking", (75, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


while True:
    timer = cv.getTickCount()
    success, img = cap.read()
    print(img.shape)


    success, bounding_box = tracker.update(img)
    if success:
        drawBox(img, bounding_box)
    else:
        cv.putText(img, "Object lost", (75, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

    cv.putText(img, str(int(fps)), (75, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.imshow("Tracking", img)

    if cv.waitKey(1) & 0xff == ord('q'):
        break

vedo.plotter.closePlotter()
cv.destroyAllWindows()