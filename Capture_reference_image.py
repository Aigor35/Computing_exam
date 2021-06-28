import cv2 as cv

camera = cv.VideoCapture(0)
captureImage = False

while True:
    success,image = camera.read()

    if captureImage == True:
        cv.imwrite("Models/Reference_image.png",image)

    cv.imshow("Image to be saved",image)

    if cv.waitKey(1) == ord('s'):
        captureImage = True
    if cv.waitKey(1) == ord('q'):
        break

camera.release()
cv.destroyAllWindows()