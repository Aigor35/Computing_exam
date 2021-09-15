import hypothesis.extra.numpy
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis import example
from hypothesis import assume
from hypothesis import settings
import Virtual_face_motion as vfm
import numpy as np






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
    boxPoints = np.array([[x + width, y], [x, y], [x, y + height], [x + width, y + height]], dtype=np.float32)
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



def test_moveFace_giveSpecificValues_returnSpecificResult():
    oldPoints = np.array([[3.5, 2.5], [2.5, 2.5], [2.5, 3.5], [3.5, 3.5]], dtype = np.float32)
    newPoints = np.array([[5., 3.], [3., 3.], [3., 5.], [5., 5.]], dtype = np.float32)
    focalLength = 0.5
    refArea = 16.
    vfm.moveFace(oldPoints,newPoints,focalLength,refArea)
    assert np.all(vfm.faceMesh.pos() == np.array([[2., -2., 1.]]))


def test_moveFace_giveNanValueToNewPoints_raiseValueError():
    oldPoints = np.array([[2., 1.], [1., 1.], [1., 2.], [2., 2.]], dtype=np.float32)
    newPoints = np.array([[np.float("nan"), 3.], [3., 3.], [3., 5.], [5., 5.]], dtype=np.float32)
    focalLength = 1
    refArea = 1
    with pytest.raises(ValueError):
        vfm.moveFace(oldPoints, newPoints, focalLength, refArea)



def test_moveFace_giveInfValueToNewPoints_raiseValueError():
    oldPoints = np.array([[2., 1.], [1., 1.], [1., 2.], [2., 2.]], dtype=np.float32)
    newPoints = np.array([[5, 3.], [3., np.float("inf")], [3., 5.], [5., 5.]], dtype=np.float32)
    focalLength = 1
    refArea = 1
    with pytest.raises(ValueError):
        vfm.moveFace(oldPoints, newPoints, focalLength, refArea)












