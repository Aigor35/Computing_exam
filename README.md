# Computing_exam

The objective of this project was to write a program able to detect and mimic
the movements of the user's face along the x, y and z axis,
as well as the rotation of the user's face around the axis perpendicular
to the user's monitor.  
This objective has been achieved through the programs "Virtual_face_motion.py"
and "Tests.py".  
The first file, "Virtual_face_motion.py", represents the core of the project
since its role is exactly following the user's movements and mimicking them.  
More precisely, this file generates a virtual space containing an
object shaped like a human head.  
As the user moves, the virtual head repeats the movements in real time.  
The movement detection stops if the user's face goes outside the camera's reach
or if the user presses the right mouse button.
The program instead stops when the user presses the 'q' key.  
The second file, "Tests.py", contains all the tests to which the
"Virtual_face_motion.py" file has been subjected.  

How the detection works:
------------------------
This program detects movement by tracking a ROI defined by the user, and checking frame by frame how the position and the angle of the ROI changed.  
The user in fact will have to select a set of points that will be tracked.  
The smallest rectangle containing all the points will define the ROI detected by the program.  

Preliminary steps:
------------------
Before starting the program, the user has to decide the ROI in advance and measure its dimensions,  
as well as the initial distance between themselves and the camera.  
It's in fact necessary to know the effective height and width of the ROI,
and the distance of the user's face from the camer when they first select the points for the ROI.  
These quantities must be expressed in cm.
The procedure described in the next section suggests to use the pupils and the corners of the lips as points of reference for the ROI.  
If the user follows this advice, the width of the ROI will be approximately equal to the distance between the pupils,  
and the height of the ROI will be equal to the distance between the top of the nose and the mouth.  
Alternatively, in order to improve the accuracy, the user can attach small stickers to the face in proximity of the suggested locations,  
then select these locations as points of reference.  


How to use the program:  
-----------------------
At the start of the program, the user will be asked to enter the width of the ROI they have decided, its height,  
and the initial distance between the user's face and the camera.  
All these quantities must be expressed in cm.  
Once all three quantities are given, two windows will open.  
One will show the feed from the camera of the pc, the other will show a virtual reality containing a simulated human head.   
The user then will have to select four or more points in the camera window by pressing the left mouse button,
and the program will immediately start tracking them.  
Although the user can select more than four points, it's important to notice that the first four points selected will
define the reference area in pixels^2 of the ROI, and therefore the first four points should be the ones that define the corners of the ROI.  
A good and easy choice for these four points are the pupils and the corners of the lips.  
These elements in fact are easy to recognize, and it's easy to measure their distance.  
Once at least four points have been selected, a rectangular ROI will be formed, and the program will
move the virtual head according to its movements.  
If the user selects by mistake a point, they can delete all the points selected by pressing the right mouse button.  
Sometimes the first selected point get misplaced by the tracking algorithm immediately after the selection.  
If this happens, the user has to press the right mouse button and reselect the points.  
This problem can happen only for the first point, and it can only happen immediately after the selection.  
The points will also be deleted if the ROI goes outside the reach of the camera.  
To stop the program entirely the user has to hold down the 'q' key.  

Known problems
--------------
The program is rather sensible to the blinking of the eyes and the movements of the lips.  
For this reason, in case the user selects the pupils
and the corners of the lips as points of reference,
they should avoid moving the lips as much as possible.  
This problem can be solved by attaching small stickers
to the face in proximity of the suggested locations,
and to select these locations as points of reference.  

Needed libraries and subpackages
--------------------------------
vedo
opencv-python
numpy

Needed files with their path
----------------------------
Models/STL_Head.stl
