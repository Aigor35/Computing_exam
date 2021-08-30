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


def test_getFocalLength_giveRefAreaEqualToZero_raiseExeption():
    refDistance = 1
    refArea = 0
    refDetectedArea = 1
    with pytest.raises(ZeroDivisionError):
        vfm.getFocalLength(refDistance, refArea, refDetectedArea)


def test_getFocalLength_giveRefDistanceLowerThanZero_raiseExeption():
    refDistance = -1
    refArea = 1
    refDetectedArea = 1
    with pytest.raises(ValueError):
        vfm.getFocalLength(refDistance, refArea, refDetectedArea)


def test_getFocalLength_giveRefAreaLowerThanZero_raiseExeption():
    refDistance = 1
    refArea = -1
    refDetectedArea = 1
    with pytest.raises(ValueError):
        vfm.getFocalLength(refDistance, refArea, refDetectedArea)


def test_getFocalLength_giveRefDetectedAreaLowerThanZero_raiseExeption():
    refDistance = 1
    refArea = 1
    refDetectedArea = -1
    with pytest.raises(ValueError):
        vfm.getFocalLength(refDistance, refArea, refDetectedArea)


def test_getFocalLength_giveRefDistanceEqualToNan_raiseExeption():
    refDistance = float("nan")
    refArea = 1
    refDetectedArea = 1
    with pytest.raises(ValueError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)



def test_getFocalLength_giveRefAreaEqualToNan_raiseExeption():
    refDistance = 1
    refArea = float("nan")
    refDetectedArea = 1
    with pytest.raises(ValueError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)



def test_getFocalLength_giveRefDetectedAreaEqualToNan_raiseExeption():
    refDistance = 1
    refArea = 1
    refDetectedArea = float("nan")
    with pytest.raises(ValueError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)


def test_getFocalLength_giveRefDistanceEqualToInf_raiseExeption():
    refDistance = float("inf")
    refArea = 1
    refDetectedArea = 1
    with pytest.raises(ValueError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)


def test_getFocalLength_giveRefAreaEqualToInf_raiseExeption():
    refDistance = 1
    refArea = float("inf")
    refDetectedArea = 1
    with pytest.raises(ValueError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)


def test_getFocalLength_giveRefDetectedAreaEqualToInf_raiseExeption():
    refDistance = 1
    refArea = 1
    refDetectedArea = float("inf")
    with pytest.raises(ValueError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)


def test_getFocalLength_giveRefDistanceNotNumerical_raiseExeption():
    refDistance = True
    refArea = 1
    refDetectedArea = 1
    with pytest.raises(TypeError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)


def test_getFocalLength_giveRefAreaNotNumerical_raiseExeption():
    refDistance = 1
    refArea = True
    refDetectedArea = 1
    with pytest.raises(TypeError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)



def test_getFocalLength_giveRefDetectedAreaNotNumerical_raiseExeption():
    refDistance = 1
    refArea = 1
    refDetectedArea = True
    with pytest.raises(TypeError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)




def test_getFocalLength_giveSpecificValues_returnSpecificResult():
    refDistance = 1
    refArea = 4
    refDetectedArea = 16
    focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)
    assert focalLength == 2




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