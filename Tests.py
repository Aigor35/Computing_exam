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



"""
#@settings(report_multiple_bugs= True)
@given(refDistance = st.floats(min_value = 10**(-10), max_value = 10**5, allow_nan = False,
                               allow_infinity = False,  exclude_min = True, exclude_max = True),
       refArea = st.floats(min_value = 10**(-10), max_value = 10**5, allow_nan = False,
                               allow_infinity = False,  exclude_min = True, exclude_max = True),
       refDetectedArea = st.floats(min_value = 10**(-10), max_value = 10**5, allow_nan = False,
                               allow_infinity = False,  exclude_min = True, exclude_max = True))
def test_getFocalLength_propertyBasedTest(refDistance, refArea, refDetectedArea):
    focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)**2
    assert focalLength - (refDistance**2)*refDetectedArea/refArea < 10**(-5)
"""

"""
@given(refDistance = st.floats(min_value = 0, allow_nan = False, allow_infinity = False, exclude_min = True),
       refArea = st.floats(min_value = 0, allow_nan = False, allow_infinity = False, exclude_min = True),
       x = st.floats(min_value = 0, allow_nan = False, allow_infinity = False, exclude_min = True),
       y = st.floats(min_value = 0, allow_nan = False, allow_infinity = False, exclude_min = True),
       width = st.floats(min_value = 0, allow_nan = False, allow_infinity = False, exclude_min = True),
       height = st.floats(min_value = 0, allow_nan = False, allow_infinity = False, exclude_min = True))
"""

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
    assert np.round(refDistance, 3) == np.round(vfm.getDistance(vfm.getFocalLength(refDistance, refArea, refDetectedArea),
                                                                refArea, boxPoints), 3)


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
    assert np.round(focalLength, 3) == np.round(vfm.getFocalLength(vfm.getDistance(focalLength, refArea, boxPoints),
                                                          refArea, refDetectedArea), 3)



def test_getDistance_giveParametersEqualToOne_returnOne():
    focalLength = 1
    refArea = 1
    boxPoints = np.array([[1, 0], [0, 0], [0, 1], [1, 1]], dtype = np.float32) # square of area 1
    distance = vfm.getDistance(focalLength, refArea, boxPoints)
    assert distance == 1


def test_getDistance_giveSpecificValues_returnSpecificResult():
    focalLength = 1
    refArea = 16
    boxPoints = np.array([[2, 0], [0, 0], [0, 2], [2, 2]], dtype = np.float32)
    distance = vfm.getDistance(focalLength, refArea, boxPoints)
    assert distance == 2


def test_getDistance_giveEmptyBoxPoints_raiseValueError():
    focalLength = 1
    refArea = 1
    boxPoints = np.array([])
    with pytest.raises(ValueError):
        vfm.getDistance(focalLength, refArea, boxPoints)



def test_getDistance_giveBoxPointsContainingNan_raiseValueError():
    focalLength = 1
    refArea = 1
    boxPoints = np.array([[1, 0], [0, float("nan")], [0, 1], [1, 1]], dtype = np.float32)
    with pytest.raises(ValueError):
        vfm.getDistance(focalLength, refArea, boxPoints)




"""
@given(focalLength = st.floats(min_value = 10**(-10), max_value = 10**5, allow_nan = False,
                               allow_infinity = False,  exclude_min = True, exclude_max = True),
       refArea = st.floats(min_value=10 ** (-10), max_value=10 ** 5, allow_nan=False,
                         allow_infinity = False, exclude_min = True, exclude_max = True),
       boxPoints = hypothesis.extra.numpy.arrays(np.float32, shape = (4, 2),
                                                 elements = st.floats(allow_nan = False, allow_infinity = False)))
def test_getDistance_propertyBasedTest(focalLength,refArea,boxPoints):
    distance = vfm.getDistance(focalLength,refArea,boxPoints)
    assert distance - (focalLength**2)*refArea/
"""


