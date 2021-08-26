import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis import example
from hypothesis import assume
import Virtual_face_motion as vfm






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


def test_getFocalLength_giveSpecificValues_returnSpecificResult():
    refDistance = 1
    refArea = 4
    refDetectedArea = 16
    focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)
    assert focalLength == 2


def test_getFocalLength_giveNegativeParameters_raiseExeption():
    refDistance = 1
    refArea = -4
    refDetectedArea = 16
    with pytest.raises(ValueError):
        focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)


@given(refDistance = st.floats(), refArea = st.floats(), refDetectedArea = st.floats())
def test_hypothesis(refDistance, refArea, refDetectedArea):
    assume(refDistance > 0)
    assume(refArea > 0)
    assume(refArea > 0)
    assert vfm.getFocalLength(refDistance, refArea, refDetectedArea)**2 == (refDistance**2)*refDetectedArea/refArea


