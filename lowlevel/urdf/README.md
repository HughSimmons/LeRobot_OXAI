This folder contains urdf for the robot with the camera mount attached. Values obtained using onshape

Define the joint to be at the motor screw hole that the mount covers on `wrist_roll_follower`
Center (relative to `wrist_roll_follower` origin:
x: -8.800mm
y: -24.218mm
z: 14.100mm

That was the stl origin, we want the origin which located in the wrist roll motor (that is how the origin is defined)

x: -8.80
y: 24.00
z: -13.15

The joint is, relative to the frame origin of the stl file of the camera mount:

Center:
x: 0.000mm
y: -18.400mm
z: 6.300mm

Orientation (relative to `wrist_roll_follower`):

|axis in mount|axis in wrist|
|+x|+-y|
|+z|+x|
|+y|-z|

giving rpy of (-pi/2,0,-pi/2)
mass: 13.56g

C.O.M : (-25.54,-30.62 ,17.07707 mm)
Relative to joint: (-25.54, -12.22,10.78) 
MoI (cm2g): [[25.505932,-14.951827,1.360207],[-14.951827,62.436672,1.075714],[1.360207,1.076,60.113901]]

need to convert MoI to m2kg (multiply by 1e-7)

`<visual xyz>` should be joint to stl origin, which is (0,18.4,-6.3)mm

board thickness: 2mm
camera base to tip length: 23mm

camera diameter: 15mm
