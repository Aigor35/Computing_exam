"""
Objective
---------
This program contains all the tests relative
to the file Virtual_face_motion.py
The framework used to run the tests is pytest.
In order to increase the readability of the code,
the name of each test describes which function or property
is tested and which result is expected.
More information about the parameters passed to the tests
and the expected results can be found in the tests' documentation.

Needed libraries and subpackages
--------------------------------
hypothesis
pytest
vedo
Virtual_face_motion
numpy
opencv-python

Needed files with their path
----------------------------
Models/STL_Head.stl

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



def test_dataCluster_instanceInitializedCorrectly():
    """
    Given:
        an instance of the vfm.dataCluster class
    Expect:
        the attributes of the instance are correctly initialized.
        This implice that:
            - the attribute pointSelected is False
            - the attribute fourPointsSelected is False
            - the attribute oldPoints is a numpy.array of size 0
            - the position associated to the attribute faceMesh is equal to the array [0., 0., 0.]
    """
    cluster = vfm.dataCluster()
    assert cluster.pointSelected == False
    assert cluster.fourPointsSelected == False
    assert np.size(cluster.oldPoints) == 0
    assert np.all(cluster.faceMesh.pos() == np.array([0., 0., 0.]))



def test_dataCluster_pointSelecedRelatedFunctions_correctManagementOfTheAttributePointSelected():
    """
    Given:
        an instance of the vfm.dataCluster class
        the boolean "True"
    Expect:
        the attribute vfm.dataCluster.pointSelected is updated correctly
        the method vfm.dataCluster.getPointSelected() returns "True"
    :return:
    """
    cluster = vfm.dataCluster()
    cluster.updatePointSelected(True)
    assert cluster.getPointSelected() == True



def test_dataCluster_fourPointsSelectedRelatedFunctions_correctManagementOfTheAttributeFourPointsSelected():
    """
    Given:
        an instance of the vfm.dataCluster class
        the boolean "True"
    Expect:
        the attribute vfm.dataCluster.fourPointsSelected is updated correctly
        the method vfm.dataCluster.getFourPointsSelected() returns "True"
    :return:
    """
    cluster = vfm.dataCluster()
    cluster.updateFourPointsSelected(True)
    assert cluster.getFourPointsSelected() == True



def test_dataCluster_oldPointsRelatedFunctions_correctManagementOfTheAttributeOldPoints():
    """
    Given:
        an instance of the vfm.dataCluster class
        the numpy.array = [[3, 4]]
        the numpy.array = [[1, 2],[5, 6]]
    Expect:
        the function vfm.dataCluster.trackOneNewPoints() adds a new point to oldPoints
        the function vfm.dataCluster.updateOldPoints() correctly updates the points stored inside oldPoints
        the function vfm.dataCluster.getOldPoints() returns the array [[3, 4]]
            the first time it's called and returns the array [[1, 2], [5, 6]] the second time instead
    """
    cluster = vfm.dataCluster()
    point = np.array([3, 4])
    cluster.trackOneNewPoint(point)
    assert np.all(cluster.getOldPoints() == np.array([[3, 4]], dtype = np.float32))
    cluster.updateOldPoints(np.array([[1, 2], [5, 6]], dtype = np.float32))
    assert np.all(cluster.getOldPoints() == np.array([[1, 2], [5, 6]], dtype = np.float32))



def test_dataCluster_faceMeshRelatedFunctions_correctManagementOfTheAttributeFaceMesh():
    """
    Given:
        an instance of the vfm.dataCluster class
        the numpy.array [1., 2., 3.]
    Expect:
        the function vfm.dataCluster.updateFaceMesh() updates correctly the attribute faceMesh
        the function vfm.dataCluster.getFaceMesh().pos() returns [x, y, z]
    """
    cluster = vfm.dataCluster()
    position = np.array([1., 2., 3.])
    faceMesh = cluster.getFaceMesh()
    faceMesh.addPos(position[0], position[1], position[2])
    cluster.updateFaceMesh(faceMesh)
    assert np.all(cluster.getFaceMesh().pos() == position)



def test_dataCluster_getRefDetectedArea_returnCorrectValue():
    """
    Given:
        an instance of the vfm.dataCluster class
        the numpy.array ([[5., 1.], [1., 1.], [1., 5.], [5., 5.]], dtype = np.flaot32)
    Expect:
        the function vfm.dataCluster.getRefDetectedArea() returns 16
    :return:
    """
    cluster = vfm.dataCluster()
    box = np.array([[5., 1.],
                    [1., 1.],
                    [1., 5.],
                    [5., 5.]], dtype = np.float32)
    cluster.updateOldPoints(box)
    assert cluster.getRefDetectedArea() == 16



def test_dataCluster_resetTrackingData_dataResetCorrectly():
    """
    Given:
        an instance of the vfm.dataCluster class
        the boolean "True"
        the numpy.array ([[5., 1.], [1., 1.], [1., 5.], [5., 5.]], dtype = np.flaot32)
        the numpy array [1., 2., 3.]
    Expect:
        the three attributes pointSelected, oldPoints and faceMesh are
        reset to their original values
    """
    cluster = vfm.dataCluster()
    box = np.array([[5., 1.],
                    [1., 1.],
                    [1., 5.],
                    [5., 5.]], dtype=np.float32)
    position = np.array([1., 2., 3.])
    cluster.updatePointSelected(True)
    cluster.updateOldPoints(box)
    faceMesh = cluster.getFaceMesh()
    faceMesh.addPos(position[0], position[1], position[2])
    cluster.updateFaceMesh(faceMesh)

    cluster.resetTrackingData()
    assert cluster.getPointSelected() == False
    assert np.size(cluster.getOldPoints()) == 0
    assert np.all(cluster.getFaceMesh().pos() == [0., 0., 0.])

@settings(deadline = 1000, max_examples = 50)
@given(x=st.integers(min_value=0, max_value=640),
       y=st.integers(min_value=0, max_value=480))
def test_manageDataCluster_dataStoredCorrectly(x, y):
    """
    Given:
        an instance of the vfm.dataCluster class
        an integer x between 0 and 640
        an integer y between 480
        the variables event = 1, flags = 1
    Expect:
        the function vfm.manageDataCluster() detects a LBUTTONDOWN mouse event
        the values of the attributes of the vfm.dataCluster instance are updated correctly
    """
    cluster = vfm.dataCluster()
    event = 1
    flags = 1
    vfm.manageDataCluster(event, x, y, flags, params = cluster)
    assert cluster.getPointSelected() == True
    assert np.all(cluster.getOldPoints() == np.array([[x, y]], dtype = np.float32))

@settings(deadline = 1000, max_examples = 50)
@given(x=st.integers(min_value=0, max_value=640),
       y=st.integers(min_value=0, max_value=480),
       xPos=st.floats(allow_nan=False, allow_infinity=False),
       yPos=st.floats(allow_nan=False, allow_infinity=False),
       zPos=st.floats(allow_nan=False, allow_infinity=False))
def test_manageDataCluster_dataResetCorrectly(x, y, xPos, yPos, zPos):
    """
    Given:
        an instance of the vfm.dataCluster class
        the integer x, between 0 and 640
        the integer y, between 0 and 480
        three finite floating numbers xPos, yPos, zPos
        the variables event = 2, flags = 2
    Expect:
        the function vfm.manageDataCluster() detects a RBUTTONDOWN mouse event
        the values of the attributes of the vfm.dataCluster instance are reset correctly
    """
    cluster = vfm.dataCluster()
    cluster.updatePointSelected(True)
    cluster.trackOneNewPoint(np.array([x, y]))
    faceMesh = cluster.getFaceMesh()
    faceMesh.addPos(xPos, yPos, zPos)
    cluster.updateFaceMesh(faceMesh)
    event = 2
    flags = 2

    vfm.manageDataCluster(event, x, y, flags, params = cluster)
    assert cluster.getPointSelected() == False
    assert np.size(cluster.getOldPoints()) == 0
    assert np.all(cluster.getFaceMesh().pos() == [0., 0., 0.])



def test_checkString_giveCorrectString_returnExpectedValue():
    """
    Given:
        the string "45.7"
    Expect:
        the function vfm.checkString() returns (True, 45.7)
    """
    string = "45.7"
    assert np.all(vfm.checkString(string) == (True, 45.7))


def test_checkString_giveStringContainingNegativeNumber_returnExpectedValue():
    """
    Given:
        the string "-45.7"
    Expect:
        the function vfm.checkString() returns (False, 0)
    """
    string = "-45.7"
    assert np.all(vfm.checkString(string) == (False, 0))


def test_checkString_giveZero_returnExpectedValue():
    """
    Given:
        the string "0"
    Expect:
        the function vfm.checkString() returns (False, 0)
    """
    string = "0"
    assert np.all(vfm.checkString(string) == (False, 0))



def test_checkString_giveNan_returnExpectedValue():
    """
    Given:
        the string "nan"
    Expect:
        the function vfm.checkString() returns (False, 0)
    :return:
    """
    string = "nan"
    assert np.all(vfm.checkString(string) == (False, 0))



def test_checkString_giveInf_returnExpectedValue():
    """
    Given:
        the string "inf"
    Expect:
        the function vfm.checkString() returns (False, 0)
    :return:
    """
    string = "inf"
    assert np.all(vfm.checkString(string) == (False, 0))



def test_checkString_giveInvalidString_returnExpectedValue():
    """
    Given:
        the string "3h"
    Expect:
        the function vfm.checkString() returns (False, 0)
    :return:
    """
    string = "3h"
    assert np.all(vfm.checkString(string) == (False, 0))



def test_getFocalLength_giveSpecificValues_returnSpecificResult():
    """
    Given:
        float variable refDistance = 1.
        float variable refArea = 4.
        float variable refDetectedArea = 16.
    Expect:
        the function vfm.getFocalLength() returns 2.
    """
    refDistance = 1.
    refArea = 4.
    refDetectedArea = 16.
    focalLength = vfm.getFocalLength(refDistance, refArea, refDetectedArea)
    assert focalLength == 2.



@given(refDistance = st.floats(min_value = 10**(-5), max_value = 10**5, allow_nan = False,
                               allow_infinity = False,  exclude_min = True, exclude_max = True),
       refArea = st.floats(min_value = 10**(-5), max_value = 10**5, allow_nan = False,
                           allow_infinity = False,  exclude_min = True, exclude_max = True),
       width = st.integers(min_value = 1, max_value = 640),
       height = st.integers(min_value = 1, max_value = 480),
       x = st.integers(min_value = 0, max_value = 640),
       y = st.integers(min_value = 0, max_value = 480)
       )
def test_getDistance_isInverseOf_getFocalLength(refDistance, refArea, x, y, width, height):
    """
    Given:
        float variable refDistance, between 10**(-5) and 10**5
        float variable refArea, between 10**(-5) and 10**5
        int variable width, between 1 and 640
        int variable x, between 0 and 640
        int variable height, between 1 and 480
        int variable y, between 0 and 480
    Expect:
        the function vfm.getDistance() is the inverse of the function vfm.getFocalLength()
    """
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
    """
    Given:
        float variable focalLength, between 10**(-5) and 10**5
        float variable refArea, between 10**(-5) and 10**5
        int variable width, between 1 and 640
        int variable x, between 0 and 640
        int variable height, between 1 and 480
        int variable y, between 0 and 480
    Expect:
        the function vfm.getFocalLength() is the inverse of the function vfm.getDistance()
        """
    boxPoints = np.array([[x + width, y],
                          [x, y],
                          [x, y + height],
                          [x + width, y + height]], dtype=np.float32)
    refDetectedArea = width * height
    ratio = vfm.getCmOverPixelsRatio(refArea,boxPoints)
    assert np.round(focalLength, 3) == np.round(vfm.getFocalLength(vfm.getDistance(focalLength, ratio),
                                                                   refArea, refDetectedArea), 3)



def test_getCmOverPixelsRatio_giveSpecificValues_returnSpecificResult():
    """
    Given:
        float variable refArea = 16.
        numpy.array([[2, 0], [0, 0], [0, 2], [2, 2]], dtype = np.float32)
    Expect:
        the function vfm.getCmOverPixelsRatio() returns 2
    """
    refArea = 16.
    boxPoints = np.array([[2, 0], [0, 0], [0, 2], [2, 2]], dtype = np.float32)
    assert vfm.getCmOverPixelsRatio(refArea, boxPoints) == 2




def test_moveFace_giveSpecificValues_returnSpecificResult():
    """
    Given:
        an instance of the vfm.dataCluster class
        numpy.array oldPoints = np.array([[3.5, 2.5], [2.5, 2.5], [2.5, 3.5], [3.5, 3.5]], dtype = np.float32)
        numpy.array newPoints = np.array([[5., 3.], [3., 3.], [3., 5.], [5., 5.]], dtype = np.float32)
        float variable focalLength = 0.5
        float variable refArea = 16.
    Expect:
        the position associated to the attribute faceMesh of the vfm.dataCluster instance
        is equal to [[2., -2., 1.]]
    """
    cluster = vfm.dataCluster()
    oldPoints = np.array([[3.5, 2.5], [2.5, 2.5], [2.5, 3.5], [3.5, 3.5]], dtype=np.float32)
    newPoints = np.array([[5., 3.], [3., 3.], [3., 5.], [5., 5.]], dtype = np.float32)
    focalLength = 0.5
    refArea = 16.
    faceMesh = cluster.getFaceMesh()
    faceMesh = vfm.moveFace(faceMesh, oldPoints, newPoints, focalLength, refArea)
    assert np.all(faceMesh.pos() == np.array([[2., -2., 1.]]))



def test_checkIfInsideBoundary_ifInsideDoNothing():
    """
    Given:
        an instance of the vfm.dataCluster class
        numpy.array oldPoints = np.full((4, 2), 1., dtype=np.float32)
        the boolean "True"
        integer windowWidth = 640
        integer windowHeight = 480
    Expect:
        the function vfm.checkIfInsideBoundary() doesn't reset the vfm.dataCluster instance.
        This implies that:
            - the method vfm.dataCluster.getPointSelected() returns True
            - the method vfm.dataCluster.getOldPoints() returns np.full((4, 2), 1., dtype=np.float32)
            - the method vfm.dataCluster.getFaceMesh.pods() returns np.array([5., 5., 5.])
    """
    cluster = vfm.dataCluster()
    cluster.updatePointSelected(True)
    cluster.updateOldPoints(np.full((4, 2), 1., dtype=np.float32))
    faceMesh = cluster.getFaceMesh()
    faceMesh.addPos(5., 5., 5.)
    cluster.updateFaceMesh(faceMesh)
    windowWidth = 640
    windowHeight = 480
    vfm.checkIfInsideBoundary(cluster, cluster.getOldPoints(), windowWidth, windowHeight)
    assert cluster.getPointSelected() == True
    assert np.all(cluster.getOldPoints() == np.full((4, 2), 1., dtype=np.float32))
    assert np.all(cluster.getFaceMesh().pos() == np.array([5., 5., 5.]))



def test_checkIfInsideBoundary_ifPointIsFurtherThanMaxXBoundary_ResetData():
    """
    Given:
        an instance of the vfm.dataCluster class
        numpy.array oldPoints = np.array([[700., 1.], [1., 1.], [1., 1.], [1., 1.]], dtype = np.float32)
        the boolean "True"
        integer windowWidth = 640
        integer windowHeight = 480
    Expect:
        the function vfm.checkIfInsideBoundary() resets correctly the vfm.dataCluster instance.
        This implies that:
            - the method vfm.dataCluster.getPointSelected() returns False
            - the method vfm.dataCluster.getOldPoints() returns an array whose size is 0
            - the method vfm.dataCluster.getFaceMesh.pods() returns np.array([0., 0., 0.])
    """
    cluster = vfm.dataCluster()
    cluster.updatePointSelected(True)
    cluster.updateOldPoints(np.array([[700., 1.],
                                      [1., 1.],
                                      [1., 1.],
                                      [1., 1.]], dtype=np.float32))
    faceMesh = cluster.getFaceMesh()
    faceMesh.addPos(5., 5., 5.)
    cluster.updateFaceMesh(faceMesh)
    windowWidth = 640
    windowHeight = 480
    vfm.checkIfInsideBoundary(cluster, cluster.getOldPoints(), windowWidth, windowHeight)
    assert cluster.getPointSelected() == False
    assert np.size(cluster.getOldPoints()) == 0
    assert np.all(cluster.getFaceMesh().pos() == np.array([0., 0., 0.]))



def test_checkIfInsideBoundary_ifPointIsFurtherThanMaxYBoundary_ResetData():
    """
    Given:
        an instance of the vfm.dataCluster class
        numpy.array oldPoints = np.array([[1., 1.], [1., 1.], [1., 480.], [1., 1.]], dtype = np.float32)
        the boolean "True"
        integer windowWidth = 640
        integer windowHeight = 480
    Expect:
        the function vfm.checkIfInsideBoundary() resets correctly the vfm.dataCluster instance.
        This implies that:
            - the method vfm.dataCluster.getPointSelected() returns False
            - the method vfm.dataCluster.getOldPoints() returns an array whose size is 0
            - the method vfm.dataCluster.getFaceMesh.pods() returns np.array([0., 0., 0.])
    """
    cluster = vfm.dataCluster()
    cluster.updatePointSelected(True)
    cluster.updateOldPoints(np.array([[1., 1.],
                                      [1., 1.],
                                      [1., 480.],
                                      [1., 1.]], dtype=np.float32))
    faceMesh = cluster.getFaceMesh()
    faceMesh.addPos(5., 5., 5.)
    cluster.updateFaceMesh(faceMesh)
    windowWidth = 640
    windowHeight = 480
    vfm.checkIfInsideBoundary(cluster, cluster.getOldPoints(), windowWidth, windowHeight)
    assert cluster.getPointSelected() == False
    assert np.size(cluster.getOldPoints()) == 0
    assert np.all(cluster.getFaceMesh().pos() == np.array([0., 0., 0.]))



def test_checkIfInsideBoundary_ifPointTouchesMaxXBoundary_ResetData():
    """
    Given:
        an instance of the vfm.dataCluster class
        numpy.array oldPoints = np.array([[1., 1.], [1., 1.], [1., 1.], [640., 1.]], dtype=np.float32)
        the boolean "True"
        integer windowWidth = 640
        integer windowHeight = 480
    Expect:
        the function vfm.checkIfInsideBoundary() resets correctly the vfm.dataCluster instance.
        This implies that:
            - the method vfm.dataCluster.getPointSelected() returns False
            - the method vfm.dataCluster.getOldPoints() returns an array whose size is 0
            - the method vfm.dataCluster.getFaceMesh.pods() returns np.array([0., 0., 0.])
    """
    cluster = vfm.dataCluster()
    cluster.updatePointSelected(True)
    cluster.updateOldPoints(np.array([[1., 1.],
                                      [1., 1.],
                                      [1., 1.],
                                      [640., 1.]], dtype=np.float32))
    faceMesh = cluster.getFaceMesh()
    faceMesh.addPos(5., 5., 5.)
    cluster.updateFaceMesh(faceMesh)
    windowWidth = 640
    windowHeight = 480
    vfm.checkIfInsideBoundary(cluster, cluster.getOldPoints(), windowWidth, windowHeight)
    assert cluster.getPointSelected() == False
    assert np.size(cluster.getOldPoints()) == 0
    assert np.all(cluster.getFaceMesh().pos() == np.array([0., 0., 0.]))



def test_checkIfInsideBoundary_ifPointTouchesMaxYBoundary_ResetData():
    """
    Given:
        an instance of the vfm.dataCluster class
        numpy.array oldPoints = np.array([[1., 480.], [1., 1.], [1., 1.], [1., 1.]], dtype=np.float32)
        the boolean "True"
        integer windowWidth = 640
        integer windowHeight = 480
    Expect:
        the function vfm.checkIfInsideBoundary() resets correctly the vfm.dataCluster instance.
        This implies that:
            - the method vfm.dataCluster.getPointSelected() returns False
            - the method vfm.dataCluster.getOldPoints() returns an array whose size is 0
            - the method vfm.dataCluster.getFaceMesh.pods() returns np.array([0., 0., 0.])
    """
    cluster = vfm.dataCluster()
    cluster.updatePointSelected(True)
    cluster.updateOldPoints(np.array([[1., 480.],
                                      [1., 1.],
                                      [1., 1.],
                                      [1., 1.]], dtype=np.float32))
    faceMesh = cluster.getFaceMesh()
    faceMesh.addPos(5., 5., 5.)
    cluster.updateFaceMesh(faceMesh)
    windowWidth = 640
    windowHeight = 480
    vfm.checkIfInsideBoundary(cluster, cluster.getOldPoints(), windowWidth, windowHeight)
    assert cluster.getPointSelected() == False
    assert np.size(cluster.getOldPoints()) == 0
    assert np.all(cluster.getFaceMesh().pos() == np.array([0., 0., 0.]))



def test_checkIfInsideBoundary_ifPointIsFurtherThanMinXBoundary_ResetData():
    """
    Given:
        an instance of the vfm.dataCluster class
        numpy.array oldPoints = np.array([[1., 1.], [-30., 1.], [1., 1.], [1., 1.]], dtype=np.float32)
        the boolean "True"
        integer windowWidth = 640
        integer windowHeight = 480
    Expect:
        the function vfm.checkIfInsideBoundary() resets correctly the vfm.dataCluster instance.
        This implies that:
            - the method vfm.dataCluster.getPointSelected() returns False
            - the method vfm.dataCluster.getOldPoints() returns an array whose size is 0
            - the method vfm.dataCluster.getFaceMesh.pods() returns np.array([0., 0., 0.])
    """
    cluster = vfm.dataCluster()
    cluster.updatePointSelected(True)
    cluster.updateOldPoints(np.array([[1., 1.],
                                      [-30., 1.],
                                      [1., 1.],
                                      [1., 1.]], dtype=np.float32))
    faceMesh = cluster.getFaceMesh()
    faceMesh.addPos(5., 5., 5.)
    cluster.updateFaceMesh(faceMesh)
    windowWidth = 640
    windowHeight = 480
    vfm.checkIfInsideBoundary(cluster, cluster.getOldPoints(), windowWidth, windowHeight)
    assert cluster.getPointSelected() == False
    assert np.size(cluster.getOldPoints()) == 0
    assert np.all(cluster.getFaceMesh().pos() == np.array([0., 0., 0.]))



def test_checkIfInsideBoundary_ifPointIsFurtherThanMinYBoundary_ResetData():
    """
    Given:
        an instance of the vfm.dataCluster class
        numpy.array oldPoints = np.array([[1., 1.], [1., 1.], [1., -30.], [1., 1.]], dtype=np.float32)
        the boolean "True"
        integer windowWidth = 640
        integer windowHeight = 480
    Expect:
        the function vfm.checkIfInsideBoundary() resets correctly the vfm.dataCluster instance.
        This implies that:
            - the method vfm.dataCluster.getPointSelected() returns False
            - the method vfm.dataCluster.getOldPoints() returns an array whose size is 0
            - the method vfm.dataCluster.getFaceMesh.pods() returns np.array([0., 0., 0.])
    """
    cluster = vfm.dataCluster()
    cluster.updatePointSelected(True)
    cluster.updateOldPoints(np.array([[1., 1.],
                                      [1., 1.],
                                      [1., -30.],
                                      [1., 1.]], dtype=np.float32))
    faceMesh = cluster.getFaceMesh()
    faceMesh.addPos(5., 5., 5.)
    cluster.updateFaceMesh(faceMesh)
    windowWidth = 640
    windowHeight = 480
    vfm.checkIfInsideBoundary(cluster, cluster.getOldPoints(), windowWidth, windowHeight)
    assert cluster.getPointSelected() == False
    assert np.size(cluster.getOldPoints()) == 0
    assert np.all(cluster.getFaceMesh().pos() == np.array([0., 0., 0.]))



def test_checkIfInsideBoundary_ifPointTouchesMinXBoundary_ResetData():
    """
    Given:
        an instance of the vfm.dataCluster class
        numpy.array oldPoints = np.array([[1., 1.], [1., 1.], [0., 1.], [1., 1.]], dtype=np.float32)
        the boolean "True"
        integer windowWidth = 640
        integer windowHeight = 480
    Expect:
        the function vfm.checkIfInsideBoundary() resets correctly the vfm.dataCluster instance.
        This implies that:
            - the method vfm.dataCluster.getPointSelected() returns False
            - the method vfm.dataCluster.getOldPoints() returns an array whose size is 0
            - the method vfm.dataCluster.getFaceMesh.pods() returns np.array([0., 0., 0.])
    """
    cluster = vfm.dataCluster()
    cluster.updatePointSelected(True)
    cluster.updateOldPoints(np.array([[1., 1.],
                                      [1., 1.],
                                      [0., 1.],
                                      [1., 1.]], dtype=np.float32))
    faceMesh = cluster.getFaceMesh()
    faceMesh.addPos(5., 5., 5.)
    cluster.updateFaceMesh(faceMesh)
    windowWidth = 640
    windowHeight = 480
    vfm.checkIfInsideBoundary(cluster, cluster.getOldPoints(), windowWidth, windowHeight)
    assert cluster.getPointSelected() == False
    assert np.size(cluster.getOldPoints()) == 0
    assert np.all(cluster.getFaceMesh().pos() == np.array([0., 0., 0.]))



def test_checkIfInsideBoundary_ifPointTouchesMinYBoundary_ResetData():
    """
    Given:
        an instance of the vfm.dataCluster class
        numpy.array oldPoints = np.array([[1., 1.],
                          [1., 1.],
                          [1., 1.],
                          [1., 0.]], dtype=np.float32)
        the boolean "True"
        integer windowWidth = 640
        integer windowHeight = 480
    Expect:
        the function vfm.checkIfInsideBoundary() resets correctly the vfm.dataCluster instance.
        This implies that:
            - the method vfm.dataCluster.getPointSelected() returns False
            - the method vfm.dataCluster.getOldPoints() returns an array whose size is 0
            - the method vfm.dataCluster.getFaceMesh.pods() returns np.array([0., 0., 0.])
    """
    cluster = vfm.dataCluster()
    cluster.updatePointSelected(True)
    cluster.updateOldPoints(np.array([[1., 1.],
                                      [1., 1.],
                                      [1., 1.],
                                      [1., 0.]], dtype=np.float32))
    faceMesh = cluster.getFaceMesh()
    faceMesh.addPos(5., 5., 5.)
    cluster.updateFaceMesh(faceMesh)
    windowWidth = 640
    windowHeight = 480
    vfm.checkIfInsideBoundary(cluster, cluster.getOldPoints(), windowWidth, windowHeight)
    assert cluster.getPointSelected() == False
    assert np.size(cluster.getOldPoints()) == 0
    assert np.all(cluster.getFaceMesh().pos() == np.array([0., 0., 0.]))





@settings(deadline = 800, max_examples = 50)
@given(xCoordinates = hypothesis.extra.numpy.arrays(int, 4,
                                               elements = st.integers(min_value = 1, max_value= 639)),
       yCoordinates = hypothesis.extra.numpy.arrays(int, 4,
                                               elements = st.integers(min_value = 1, max_value= 479)))
def test_checkIfInsideBoundary_generateBoxPointInsideBoundary_doNothing(xCoordinates, yCoordinates):
    """
    Given:
        a numpy.array of four integers ranging from 1 to 639
        a numpy.array of four integers ranging from 1 to 479
        the boolean "True"
        numpy.array oldPoints = np.full((4, 2), 1., dtype = np.float32)
    Expect:
        the function vfm.checkIfInsideBoundary() doesn't reset the vfm.dataCluster instance.
        This implies that:
            - the method vfm.dataCluster.getPointSelected() returns True
            - the method vfm.dataCluster.getOldPoints() returns np.full((4, 2), 1., dtype=np.float32)
            - the method vfm.dataCluster.getFaceMesh.pods() returns np.array([5., 5., 5.])
    """
    cluster = vfm.dataCluster()
    cluster.updatePointSelected(True)
    oldPoints = np.array([], dtype = np.float32)
    for i in range(0, 4):
        oldPoints = np.append(oldPoints, xCoordinates[i])
        oldPoints = np.append(oldPoints, yCoordinates[i])
    oldPoints = np.reshape(oldPoints, (4, 2))
    cluster.updateOldPoints(oldPoints)
    faceMesh = cluster.getFaceMesh()
    faceMesh.addPos(5., 5., 5.)
    cluster.updateFaceMesh(faceMesh)
    vfm.checkIfInsideBoundary(faceMesh, oldPoints, 640, 480)
    assert cluster.getPointSelected() == True
    assert np.all(cluster.getOldPoints() == oldPoints)
    assert np.all(cluster.getFaceMesh().pos() == np.array([5., 5., 5.]))




def test_getRotationAngle_giveSpecificValues_rotateFaceWithSpecificAngle():
    """
    Given:
        float alpha = np.radians(20), describing the 20° angle expressed in radians
        numpy.array oldPoints = np.array([[100., 100.], [100., 200.],
                                        [150., 200.], [150., 100.]], dtype = np.float32)
        numpy.array rotationMatrix = np.array([[np.cos(alpha), -np.sin(alpha)],
                                              [np.sin(alpha), np.cos(alpha)]], dtype = np.float32)
    Expect:
        the function vfm.getRotationAngle() returns -20
    """
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
    """
    Given:
        float alpha = np.radians(20), describing the 20° angle expressed in radians
        numpy.array oldPoints = np.array([[100., 100.], [100., 200.],
                                        [150., 200.], [150., 100.]], dtype = np.float32)
    Expect:
        the function vfm.getRotationAngle() returns 0
    """
    alpha = np.radians(20)
    oldPoints = np.array([[100., 100.],
                          [100., 200.],
                          [150., 200.],
                          [150., 100.]], dtype=np.float32)
    oldRectangle = cv.minAreaRect(oldPoints)
    newRectangle = cv.minAreaRect(oldPoints)
    assert vfm.getRotationAngle(oldRectangle, newRectangle) == 0



def test_getRotationAngle_giveRotationAngleGraterThan60Degrees_returnZero():
    """
    Given:
        float alpha, describing an angle expressed in radians.
            Said angle is first defined as np.radians(10), then np.radians(70)
        numpy.array oldPoints = np.array([[100., 100.], [100., 200.],
                                        [150., 200.], [150., 100.]], dtype = np.float32)
        numpy.array rotationMatrix = np.array([[np.cos(alpha), -np.sin(alpha)],
                                              [np.sin(alpha), np.cos(alpha)]], dtype = np.float32)
    Expect:
        the function vfm.getRotationAngle() returns 0
    """
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
       height = st.integers(min_value = 1, max_value = 480),
       beta = st.integers(min_value = 1, max_value = 80))
def test_getRotationAngle_rotateRectangleCounterclockwiseAndClockwiseBySameAngle_obtainAgainInitialRectangle(x, y, width,
                                                                                                             height, beta):
    """
    Given:
        int x, between 0 and 640
        int y, between 0 and 480
        int width, between 1 and 640
        int height, between 1 and 480
        int beta, between 1 and 80
        float alpha = np.radians(10), describing the 10° angle expressed in radians
    Expect:
        if a set of points is rotated counterclockwise and then clockwise by the same angle,
        the function vfm.getRotationAngle() successfully detects a total rotation angle
        equal to 0
    """
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

    startingRectangle = cv.minAreaRect(startingPoints)
    clockwiseRotatedRectangle = cv.minAreaRect(clockwiseRotatedPoints)

    assert vfm.getRotationAngle(startingRectangle, clockwiseRotatedRectangle) == 0



@given(x = st.integers(min_value = 0, max_value = 640),
       y = st.integers(min_value = 0, max_value = 480),
       width = st.integers(min_value = 1, max_value = 640),
       height = st.integers(min_value = 1, max_value = 640),
       beta = st.integers(min_value = 1, max_value = 80))
def test_getRotationAngle_rotateRectangleClockwiseAndCounterclockwiseBySameAngle_obtainAgainInitialRectangle(x, y, width,
                                                                                                             height, beta):
    """
    Given:
        int x, between 0 and 640
        int y, between 0 and 480
        int width, between 1 and 640
        int height, between 1 and 480
        int beta, between 1 and 80
        float alpha = np.radians(-10), describing the -10° angle expressed in radians
    Expect:
        if a set of points is rotated clockwise and then counterclockwise by the same angle,
        the function vfm.getRotationAngle() successfully detects a total rotation angle
        equal to 0
    """
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

    startingRectangle = cv.minAreaRect(startingPoints)
    counterclockwiseRotatedRectangle = cv.minAreaRect(counterclockwiseRotatedPoints)

    assert vfm.getRotationAngle(startingRectangle, counterclockwiseRotatedRectangle) == 0


