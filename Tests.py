"""
Objective
---------
This program contains all the tests relative
to the file Virtual_face_motion.py
The framework used to run the tests is pytest.
In order to increase the readability of the code,
the name of each unit test follows a specific format, that is
"test"_"function to be tested"_"given parameters or specific condition to be tested"_"expected result".
As for the property based tests, they follow another format, namely
"test"_"function to be tested"_"property to be tested"

Needed libraries and subpackages
--------------------------------
hypothesis.extra.numpy
pytest
vedo
hypothesis.given
hypothesis.strategies
hypothesis.settings
Virtual_face_motion
numpy
cv2

Needed files with their path
----------------------------
Models/STL_Head.stl

Tests
-----
test_getFocalLength_giveParametersEqualToOne_returnOne()
test_getFocalLength_giveSpecificValues_returnSpecificResult()
test_getFocalLength_giveRefAreaEqualToZero_raiseZeroDivisionError()
test_getFocalLength_giveRefDistanceLowerThanZero_raiseValueError()
test_getFocalLength_giveRefAreaLowerThanZero_raiseValueError()
test_getFocalLength_giveRefDetectedAreaLowerThanZero_raiseValueError()
test_getFocalLength_giveRefDistanceEqualToNan_raiseValueError()
test_getFocalLength_giveRefAreaEqualToNan_raiseValueError()
test_getFocalLength_giveRefDetectedAreaEqualToNan_raiseValueError()
test_getFocalLength_giveRefDistanceEqualToInf_raiseValueError()
test_getFocalLength_giveRefAreaEqualToInf_raiseValueError()
test_getFocalLength_giveRefDetectedAreaEqualToInf_raiseValueError()
test_getFocalLength_giveRefDistanceNotNumerical_raiseTypeError()
test_getFocalLength_giveRefAreaNotNumerical_raiseTypeError()
test_getFocalLength_giveRefDetectedAreaNotNumerical_raiseTypeError()
test_getFocalLength_returnsOnlyNumericalValues(float, float, float)
test_getDistance_isInverseOf_getFocalLength(float, float, int, int, int, int)
test_getFocalLength_isInverseOf_getDistance(float, float, int, int, int, int)
test_getCmOverPixelsRatio_giveParametersEqualToOne_returnOne()
test_getCmOverPixelsRatio_giveSpecificValues_returnSpecificResult()
test_getDistance_giveEmptyBoxPoints_raiseValueError()
test_getCmOverPixelsRatio_giveBoxPointsContainingNan_raiseValueError()
test_getCmOverPixelsRatio_giveBoxPointsContainingInf_raiseValueError()
test_manageTrackedPoints_dataStoredCorrectly(int, int)
test_manageTrackedPoints_dataResetCorrectly(int, int, float, float, float)
test_checkIfInsideBoundary_ifInsideDoNothing()
test_checkIfInsideBoundary_ifPointIsFurtherThanMaxXBoundary_ResetData()
test_checkIfInsideBoundary_ifPointIsFurtherThanMaxYBoundary_ResetData()
test_checkIfInsideBoundary_ifPointTouchesMaxXBoundary_ResetData()
test_checkIfInsideBoundary_ifPointTouchesMaxYBoundary_ResetData()
test_checkIfInsideBoundary_ifPointIsFurtherThanMinXBoundary_ResetData()
test_checkIfInsideBoundary_ifPointIsFurtherThanMinYBoundary_ResetData()
test_checkIfInsideBoundary_ifPointTouchesMinXBoundary_ResetData()
test_checkIfInsideBoundary_ifPointTouchesMinYBoundary_ResetData()
test_checkIfInsideBoundary_giveNanValue_raiseValueError()
test_checkIfInsideBoundary_giveInfiniteValue_raiseValueError()
test_checkIfInsideBoundary_generateBoxPointInsideBoundary_doNothing(numpy.array, numpy.array)
test_moveFace_giveSpecificValues_returnSpecificResult()
test_getRotationAngle_giveSpecificValues_rotateFaceWithSpecificAngle()
test_getRotationAngle_giveEqualRectangles_returnZero()
test_getRotationAngle_giveRotationAngleGraterThan60Degrees_returnZero()
test_getRotationAngle_rotateRectangleCounterclockwiseAndClockwiseBySameAngle_obtainAgainInitialRectangle(int, int, float, float, int)
test_getRotationAngle_rotateRectangleClockwiseAndCounterclockwiseBySameAngle_obtainAgainInitialRectangle(int, int, float, float, int)
"""
import hypothesis.extra.numpy
import pytest
import vedo
from hypothesis import given
from hypothesis import strategies as st
from hypothesis import settings
import Virtual_face_motion as vfm
import numpy as np
import cv2 as cv



def test_getFocalLength_giveParametersEqualToOne_returnOne():
    refDistance = 1
    refArea = 1
    refDetectedArea = 1
    focalLgenth = vfm.getFocalLength(refDistance, refArea, refDetectedArea)
    assert focalLgenth == 1



def test_getFocalLength_giveSpecificValues_returnSpecificResult():
    refDistance = 1
    refArea = 4
    refDetectedArea = 16
    focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)
    assert focalLength == 2



def test_getFocalLength_giveRefAreaEqualToZero_raiseZeroDivisionError():
    refDistance = 1
    refArea = 0
    refDetectedArea = 1
    with pytest.raises(ZeroDivisionError):
        vfm.getFocalLength(refDistance, refArea, refDetectedArea)



def test_getFocalLength_giveRefDistanceLowerThanZero_raiseValueError():
    refDistance = -1
    refArea = 1
    refDetectedArea = 1
    with pytest.raises(ValueError):
        vfm.getFocalLength(refDistance, refArea, refDetectedArea)



def test_getFocalLength_giveRefAreaLowerThanZero_raiseValueError():
    refDistance = 1
    refArea = -1
    refDetectedArea = 1
    with pytest.raises(ValueError):
        vfm.getFocalLength(refDistance, refArea, refDetectedArea)



def test_getFocalLength_giveRefDetectedAreaLowerThanZero_raiseValueError():
    refDistance = 1
    refArea = 1
    refDetectedArea = -1
    with pytest.raises(ValueError):
        vfm.getFocalLength(refDistance, refArea, refDetectedArea)



def test_getFocalLength_giveRefDistanceEqualToNan_raiseValueError():
    refDistance = float("nan")
    refArea = 1
    refDetectedArea = 1
    with pytest.raises(ValueError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)



def test_getFocalLength_giveRefAreaEqualToNan_raiseValueError():
    refDistance = 1
    refArea = float("nan")
    refDetectedArea = 1
    with pytest.raises(ValueError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)



def test_getFocalLength_giveRefDetectedAreaEqualToNan_raiseValueError():
    refDistance = 1
    refArea = 1
    refDetectedArea = float("nan")
    with pytest.raises(ValueError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)



def test_getFocalLength_giveRefDistanceEqualToInf_raiseValueError():
    refDistance = float("inf")
    refArea = 1
    refDetectedArea = 1
    with pytest.raises(ValueError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)



def test_getFocalLength_giveRefAreaEqualToInf_raiseValueError():
    refDistance = 1
    refArea = float("inf")
    refDetectedArea = 1
    with pytest.raises(ValueError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)



def test_getFocalLength_giveRefDetectedAreaEqualToInf_raiseValueError():
    refDistance = 1
    refArea = 1
    refDetectedArea = float("inf")
    with pytest.raises(ValueError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)



def test_getFocalLength_giveRefDistanceNotNumerical_raiseTypeError():
    refDistance = True
    refArea = 1
    refDetectedArea = 1
    with pytest.raises(TypeError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)



def test_getFocalLength_giveRefAreaNotNumerical_raiseTypeError():
    refDistance = 1
    refArea = True
    refDetectedArea = 1
    with pytest.raises(TypeError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)



def test_getFocalLength_giveRefDetectedAreaNotNumerical_raiseTypeError():
    refDistance = 1
    refArea = 1
    refDetectedArea = True
    with pytest.raises(TypeError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)


@given(refDistance = st.floats(min_value = 10**(-10), max_value = 10**5, allow_nan = False,
                               allow_infinity = False,  exclude_min = True, exclude_max = True),
       refArea = st.floats(min_value = 10**(-10), max_value = 10**5, allow_nan = False,
                               allow_infinity = False,  exclude_min = True, exclude_max = True),
       refDetectedArea = st.floats(min_value = 10**(-10), max_value = 10**5, allow_nan = False,
                               allow_infinity = False,  exclude_min = True, exclude_max = True))
def test_getFocalLength_returnsOnlyNumericalValues(refDistance,refArea,refDetectedArea):
    focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)
    assert type(focalLength) in vfm.acceptedTypes



@given(refDistance = st.floats(min_value = 10**(-5), max_value = 10**5, allow_nan = False,
                               allow_infinity = False,  exclude_min = True, exclude_max = True),
       refArea = st.floats(min_value = 10**(-5), max_value = 10**5, allow_nan = False,
                           allow_infinity = False,  exclude_min = True, exclude_max = True),
       width = st.integers(min_value = 1, max_value = 640),
       height = st.integers(min_value = 1, max_value = 640),
       x = st.integers(min_value = 0, max_value = 640),
       y = st.integers(min_value = 0, max_value = 480)
       )
def test_getDistance_isInverseOf_getFocalLength(refDistance, refArea, x, y, width, height):
    boxPoints = np.array([[x + width, y], [x, y], [x, y + height], [x + width, y + height]], dtype = np.float32)
    refDetectedArea = width*height
    ratio = vfm.getCmOverPixelsRatio(refArea,boxPoints)
    assert np.round(refDistance, 3) == np.round(vfm.getDistance(vfm.getFocalLength(refDistance,refArea, refDetectedArea),
                                                                ratio), 3)



@given(focalLength = st.floats(min_value = 10**(-5), max_value = 10**5, allow_nan = False,
                               allow_infinity = False,  exclude_min = True, exclude_max = True),
       refArea = st.floats(min_value = 10**(-5), max_value = 10**5, allow_nan = False,
                           allow_infinity = False,  exclude_min = True, exclude_max = True),
       width = st.integers(min_value = 1, max_value = 640),
       height = st.integers(min_value = 1, max_value = 640),
       x = st.integers(min_value = 0, max_value = 640),
       y = st.integers(min_value = 0, max_value = 480)
       )
def test_getFocalLength_isInverseOf_getDistance(focalLength, refArea, x, y, width, height):
    boxPoints = np.array([[x + width, y],
                          [x, y],
                          [x, y + height],
                          [x + width, y + height]], dtype=np.float32)
    refDetectedArea = width * height
    ratio = vfm.getCmOverPixelsRatio(refArea,boxPoints)
    assert np.round(focalLength, 3) == np.round(vfm.getFocalLength(vfm.getDistance(focalLength, ratio),
                                                                   refArea, refDetectedArea), 3)


def test_getCmOverPixelsRatio_giveParametersEqualToOne_returnOne():
    refArea = 1
    boxPoints = np.array([[1, 0], [0, 0], [0, 1], [1, 1]], dtype = np.float32) # square of area 1
    ratio = vfm.getCmOverPixelsRatio(refArea, boxPoints)
    assert ratio == 1



def test_getCmOverPixelsRatio_giveSpecificValues_returnSpecificResult():
    refArea = 16
    boxPoints = np.array([[2, 0], [0, 0], [0, 2], [2, 2]], dtype = np.float32)
    ratio = vfm.getCmOverPixelsRatio(refArea, boxPoints)
    assert ratio == 2



def test_getDistance_giveEmptyBoxPoints_raiseValueError():
    refArea = 1
    boxPoints = np.array([])
    with pytest.raises(ValueError):
        vfm.getCmOverPixelsRatio(refArea, boxPoints)



def test_getCmOverPixelsRatio_giveBoxPointsContainingNan_raiseValueError():
    refArea = 1
    boxPoints = np.array([[1, 0], [0, float("nan")], [0, 1], [1, 1]], dtype = np.float32)
    with pytest.raises(ValueError):
        vfm.getCmOverPixelsRatio(refArea, boxPoints)



def test_getCmOverPixelsRatio_giveBoxPointsContainingInf_raiseValueError():
    refArea = 1
    boxPoints = np.array([[1, 0], [0, 0], [0, 1], [float("inf"), 1]], dtype = np.float32)
    with pytest.raises(ValueError):
        vfm.getCmOverPixelsRatio(refArea, boxPoints)



@given(x = st.integers(min_value = 0, max_value = 640),
       y = st.integers(min_value = 0, max_value = 480))
def test_manageTrackedPoints_dataStoredCorrectly(x, y):
    vfm.oldPoints = np.array([[]], dtype=np.float32)
    event = 1
    flags = 1
    params = None
    vfm.manageTrackedPoints(event, x, y, flags, params)
    assert np.all(vfm.oldPoints == np.array([[x, y]], dtype = np.float32))



@settings(deadline = 400, max_examples = 50)
@given(x = st.integers(min_value = 0, max_value = 640),
       y = st.integers(min_value = 0, max_value = 480),
       xPos=st.floats(allow_nan=False, allow_infinity=False),
       yPos=st.floats(allow_nan=False, allow_infinity=False),
       zPos=st.floats(allow_nan=False, allow_infinity=False))
def test_manageTrackedPoints_dataResetCorrectly(x, y, xPos, yPos, zPos):
    vfm.oldPoints = np.array([[x, y]], dtype = np.float32)
    vfm.faceMesh.addPos(xPos, yPos, zPos)
    event = 2
    flags = 2
    params = None
    vfm.manageTrackedPoints(event, x, y, flags, params)
    assert vfm.oldPoints.size == 0
    assert np.all(vfm.faceMesh.pos() == np.array([[0., 0., 0.]]))



def test_checkIfInsideBoundary_ifInsideDoNothing():
    vfm.pointSelected = True
    vfm.oldPoints = np.full((4, 2), 1., dtype=np.float32)
    newPoints = np.array([[1., 1.],
                         [1., 1.],
                         [1., 1.],
                         [1., 1.]], dtype = np.float32)
    vfm.faceMesh.addPos(5., 5., 5.)
    vfm.checkIfInsideBoundary(newPoints, 640, 480)
    assert vfm.pointSelected == True
    assert np.all(vfm.oldPoints == np.full((4, 2), 1., dtype=np.float32))
    assert np.all(vfm.faceMesh.pos() == np.array([5., 5., 5.]))



def test_checkIfInsideBoundary_ifPointIsFurtherThanMaxXBoundary_ResetData():
    vfm.pointSelected = True
    vfm.oldPoints = np.full((4, 2), 1., dtype=np.float32)
    newPoints = np.array([[700., 1.],
                          [1., 1.],
                          [1., 1.],
                          [1., 1.]], dtype=np.float32)
    vfm.faceMesh.addPos(5., 5., 5.)
    vfm.checkIfInsideBoundary(newPoints, 640, 480)
    assert vfm.pointSelected == False
    assert vfm.oldPoints.size == 0
    assert np.all(vfm.faceMesh.pos() == np.array([0., 0., 0.]))
    vfm.faceMesh = vedo.Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



def test_checkIfInsideBoundary_ifPointIsFurtherThanMaxYBoundary_ResetData():
    vfm.pointSelected = True
    vfm.oldPoints = np.full((4, 2), 1., dtype=np.float32)
    newPoints = np.array([[1., 1.],
                          [1., 1.],
                          [1., 480.],
                          [1., 1.]], dtype=np.float32)
    vfm.faceMesh.addPos(5., 5., 5.)
    vfm.checkIfInsideBoundary(newPoints, 640, 480)
    assert vfm.pointSelected == False
    assert vfm.oldPoints.size == 0
    assert np.all(vfm.faceMesh.pos() == np.array([0., 0., 0.]))
    vfm.faceMesh = vedo.Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



def test_checkIfInsideBoundary_ifPointTouchesMaxXBoundary_ResetData():
    vfm.pointSelected = True
    vfm.oldPoints = np.full((4, 2), 1., dtype=np.float32)
    newPoints = np.array([[1., 1.],
                          [1., 1.],
                          [1., 1.],
                          [640., 1.]], dtype=np.float32)
    vfm.faceMesh.addPos(5., 5., 5.)
    vfm.checkIfInsideBoundary(newPoints, 640, 480)
    assert vfm.pointSelected == False
    assert vfm.oldPoints.size == 0
    assert np.all(vfm.faceMesh.pos() == np.array([0., 0., 0.]))
    vfm.faceMesh = vedo.Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



def test_checkIfInsideBoundary_ifPointTouchesMaxYBoundary_ResetData():
    vfm.pointSelected = True
    vfm.oldPoints = np.full((4, 2), 1., dtype=np.float32)
    newPoints = np.array([[1., 480.],
                          [1., 1.],
                          [1., 1.],
                          [1., 1.]], dtype=np.float32)
    vfm.faceMesh.addPos(5., 5., 5.)
    vfm.checkIfInsideBoundary(newPoints, 640, 480)
    assert vfm.pointSelected == False
    assert vfm.oldPoints.size == 0
    assert np.all(vfm.faceMesh.pos() == np.array([0., 0., 0.]))
    vfm.faceMesh = vedo.Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



def test_checkIfInsideBoundary_ifPointIsFurtherThanMinXBoundary_ResetData():
    vfm.pointSelected = True
    vfm.oldPoints = np.full((4, 2), 1., dtype=np.float32)
    newPoints = np.array([[1., 1.],
                          [-30., 1.],
                          [1., 1.],
                          [1., 1.]], dtype=np.float32)
    vfm.faceMesh.addPos(5., 5., 5.)
    vfm.checkIfInsideBoundary(newPoints, 640, 480)
    assert vfm.pointSelected == False
    assert vfm.oldPoints.size == 0
    assert np.all(vfm.faceMesh.pos() == np.array([0., 0., 0.]))
    vfm.faceMesh = vedo.Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



def test_checkIfInsideBoundary_ifPointIsFurtherThanMinYBoundary_ResetData():
    vfm.pointSelected = True
    vfm.oldPoints = np.full((4, 2), 1., dtype=np.float32)
    newPoints = np.array([[1., 1.],
                          [1., 1.],
                          [1., -30.],
                          [1., 1.]], dtype=np.float32)
    vfm.faceMesh.addPos(5., 5., 5.)
    vfm.checkIfInsideBoundary(newPoints, 640, 480)
    assert vfm.pointSelected == False
    assert vfm.oldPoints.size == 0
    assert np.all(vfm.faceMesh.pos() == np.array([0., 0., 0.]))
    vfm.faceMesh = vedo.Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



def test_checkIfInsideBoundary_ifPointTouchesMinXBoundary_ResetData():
    vfm.pointSelected = True
    vfm.oldPoints = np.full((4, 2), 1., dtype=np.float32)
    newPoints = np.array([[1., 1.],
                          [1., 1.],
                          [0., 1.],
                          [1., 1.]], dtype=np.float32)
    vfm.faceMesh.addPos(5., 5., 5.)
    vfm.checkIfInsideBoundary(newPoints, 640, 480)
    assert vfm.pointSelected == False
    assert vfm.oldPoints.size == 0
    assert np.all(vfm.faceMesh.pos() == np.array([0., 0., 0.]))
    vfm.faceMesh = vedo.Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



def test_checkIfInsideBoundary_ifPointTouchesMinYBoundary_ResetData():
    vfm.pointSelected = True
    vfm.oldPoints = np.full((4, 2), 1., dtype=np.float32)
    newPoints = np.array([[1., 1.],
                          [1., 1.],
                          [1., 1.],
                          [1., 0.]], dtype=np.float32)
    vfm.faceMesh.addPos(5., 5., 5.)
    vfm.checkIfInsideBoundary(newPoints, 640, 480)
    assert vfm.pointSelected == False
    assert vfm.oldPoints.size == 0
    assert np.all(vfm.faceMesh.pos() == np.array([0., 0., 0.]))
    vfm.faceMesh = vedo.Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



def test_checkIfInsideBoundary_giveNanValue_raiseValueError():
    newPoints = np.array([[np.float("nan"), 1.],
                          [1., 1.],
                          [1., 1.],
                          [1., 1.]], dtype=np.float32)
    with pytest.raises(ValueError):
        vfm.checkIfInsideBoundary(vfm.oldPoints, 640, 480)



def test_checkIfInsideBoundary_giveInfiniteValue_raiseValueError():
    newPoints = np.array([[1., 1.],
                          [1., np.float("inf")],
                          [1., 1.],
                          [1., 1.]], dtype=np.float32)
    with pytest.raises(ValueError):
        vfm.checkIfInsideBoundary(vfm.oldPoints, 640, 480)



@settings(deadline = 400, max_examples = 50)
@given(xCoordinates = hypothesis.extra.numpy.arrays(int, 4,
                                               elements = st.integers(min_value = 1, max_value= 639)),
       yCoordinates = hypothesis.extra.numpy.arrays(int, 4,
                                               elements = st.integers(min_value = 1, max_value= 479)))
def test_checkIfInsideBoundary_generateBoxPointInsideBoundary_doNothing(xCoordinates, yCoordinates):
    vfm.pointSelected = True
    newPoints = np.array([], dtype = np.float32)
    vfm.oldPoints = np.full((4, 2), 1., dtype = np.float32)
    for i in range(0, 4):
        newPoints = np.append(newPoints, xCoordinates[i])
        newPoints = np.append(newPoints, yCoordinates[i])
    newPoints = np.reshape(newPoints, (4, 2))
    vfm.faceMesh.addPos(5., 5., 5.)
    vfm.checkIfInsideBoundary(newPoints, 640, 480)
    assert vfm.pointSelected == True
    assert np.all(vfm.oldPoints == np.full((4, 2), 1., dtype = np.float32))
    assert np.all(vfm.faceMesh.pos() == np.array([5., 5., 5.]))
    vfm.faceMesh = vedo.Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



def test_moveFace_giveSpecificValues_returnSpecificResult():
    oldPoints = np.array([[3.5, 2.5], [2.5, 2.5], [2.5, 3.5], [3.5, 3.5]], dtype = np.float32)
    newPoints = np.array([[5., 3.], [3., 3.], [3., 5.], [5., 5.]], dtype = np.float32)
    focalLength = 0.5
    refArea = 16.
    vfm.moveFace(oldPoints, newPoints, focalLength, refArea)
    assert np.all(vfm.faceMesh.pos() == np.array([[2., -2., 1.]]))
    vfm.faceMesh = vedo.Mesh('Models/STL_Head.stl').rotateX(-90).rotateY(180)



def test_getRotationAngle_giveSpecificValues_rotateFaceWithSpecificAngle():
    alpha = np.radians(20)
    oldPoints = np.array([[100., 100.],
                             [100., 200.],
                             [150., 200.],
                             [150., 100.]], dtype = np.float32)
    rotationMatrix = np.array([[np.cos(alpha), -np.sin(alpha)],
                               [np.sin(alpha), np.cos(alpha)]], dtype = np.float32)
    oldPoints -= 100
    oldPoints = np.matmul(rotationMatrix, oldPoints.T)
    newPoints = np.matmul(rotationMatrix, oldPoints)

    oldPoints = oldPoints.T
    oldPoints += 100
    newPoints = newPoints.T
    newPoints += 100

    oldRectangle = cv.minAreaRect(oldPoints)
    newRectangle = cv.minAreaRect(newPoints)
    assert np.round(vfm.getRotationAngle(oldRectangle, newRectangle), 3) == -20



def test_getRotationAngle_giveEqualRectangles_returnZero():
    alpha = np.radians(20)
    oldPoints = np.array([[100., 100.],
                          [100., 200.],
                          [150., 200.],
                          [150., 100.]], dtype=np.float32)
    oldRectangle = cv.minAreaRect(oldPoints)
    newRectangle = cv.minAreaRect(oldPoints)
    assert vfm.getRotationAngle(oldRectangle, newRectangle) == 0



def test_getRotationAngle_giveRotationAngleGraterThan60Degrees_returnZero():
    alpha = np.radians(10)
    oldPoints = np.array([[100., 100.],
                          [100., 200.],
                          [150., 200.],
                          [150., 100.]], dtype=np.float32)
    rotationMatrix = np.array([[np.cos(alpha), -np.sin(alpha)],
                               [np.sin(alpha), np.cos(alpha)]], dtype=np.float32)
    oldPoints -= 100
    oldPoints = np.matmul(rotationMatrix, oldPoints.T)

    alpha = np.radians(70)
    rotationMatrix = np.array([[np.cos(alpha), -np.sin(alpha)],
                               [np.sin(alpha), np.cos(alpha)]], dtype=np.float32)

    newPoints = np.matmul(rotationMatrix, oldPoints)

    oldPoints = oldPoints.T
    oldPoints += 100
    newPoints = newPoints.T
    newPoints += 100

    oldRectangle = cv.minAreaRect(oldPoints)
    newRectangle = cv.minAreaRect(newPoints)
    assert vfm.getRotationAngle(oldRectangle, newRectangle) == 0


@given(x = st.integers(min_value = 0, max_value = 640),
       y = st.integers(min_value = 0, max_value = 480),
       width = st.integers(min_value = 1, max_value = 640),
       height = st.integers(min_value = 1, max_value = 640),
       beta = st.integers(min_value = 1, max_value = 80))
def test_getRotationAngle_rotateRectangleCounterclockwiseAndClockwiseBySameAngle_obtainAgainInitialRectangle(x, y, width,
                                                                                                             height, beta):
    setOfPoints = np.array([[x, y],
                               [x, y + height],
                               [x + width, y + height],
                               [x + width, y]], dtype = np.float32)
    alpha = np.radians(10)
    rotationMatrix = np.array([[np.cos(alpha), -np.sin(alpha)],
                               [np.sin(alpha), np.cos(alpha)]], dtype=np.float32)
    setOfPoints -= np.array([x, y], dtype = np.float32)
    startingPoints = np.matmul(rotationMatrix, setOfPoints.T)

    beta = np.radians(beta)
    counterclockwiseRotationMatrix = np.array([[np.cos(beta), -np.sin(beta)],
                                               [np.sin(beta), np.cos(beta)]], dtype=np.float32)
    counterclockwiseRotatedPoints = np.matmul(counterclockwiseRotationMatrix, startingPoints)

    clockwiseRotationMatrix = counterclockwiseRotationMatrix.T
    clockwiseRotatedPoints = np.matmul(clockwiseRotationMatrix, counterclockwiseRotatedPoints)

    startingPoints = startingPoints.T
    startingPoints += np.array([x, y], dtype = np.float32)
    startingPoints = np.around(startingPoints, 0)

    clockwiseRotatedPoints = clockwiseRotatedPoints.T
    clockwiseRotatedPoints += np.array([x, y], dtype = np.float32)
    clockwiseRotatedPoints = np.around(clockwiseRotatedPoints, 0)

    assert np.all(startingPoints == clockwiseRotatedPoints)



@given(x = st.integers(min_value = 0, max_value = 640),
       y = st.integers(min_value = 0, max_value = 480),
       width = st.integers(min_value = 1, max_value = 640),
       height = st.integers(min_value = 1, max_value = 640),
       beta = st.integers(min_value = 1, max_value = 80))
def test_getRotationAngle_rotateRectangleClockwiseAndCounterclockwiseBySameAngle_obtainAgainInitialRectangle(x, y, width,
                                                                                                             height, beta):
    setOfPoints = np.array([[x, y],
                               [x, y + height],
                               [x + width, y + height],
                               [x + width, y]], dtype = np.float32)
    alpha = np.radians(-10)
    rotationMatrix = np.array([[np.cos(alpha), -np.sin(alpha)],
                               [np.sin(alpha), np.cos(alpha)]], dtype=np.float32)
    setOfPoints -= np.array([x, y], dtype = np.float32)
    startingPoints = np.matmul(rotationMatrix, setOfPoints.T)

    beta = np.radians(beta)
    clockwiseRotationMatrix = np.array([[np.cos(beta), np.sin(beta)],
                                        [-np.sin(beta), np.cos(beta)]], dtype=np.float32)
    clockwiseRotatedPoints = np.matmul(clockwiseRotationMatrix, startingPoints)

    counterclockwiseRotationMatrix = clockwiseRotationMatrix.T
    counterclockwiseRotatedPoints = np.matmul(counterclockwiseRotationMatrix, clockwiseRotatedPoints)

    startingPoints = startingPoints.T
    startingPoints += np.array([x, y], dtype = np.float32)
    startingPoints = np.around(startingPoints, 0)

    counterclockwiseRotatedPoints = counterclockwiseRotatedPoints.T
    counterclockwiseRotatedPoints += np.array([x, y], dtype = np.float32)
    counterclockwiseRotatedPoints = np.around(counterclockwiseRotatedPoints, 0)

    assert np.all(startingPoints == counterclockwiseRotatedPoints)
