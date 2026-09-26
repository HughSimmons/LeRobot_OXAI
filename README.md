Software to turn SO101 arm into chess robot.

See:
- [lowlevel/README.md](lowlevel/README.md) -- simulation, trajectory-generation, and hardware-calibration code (start here to build/verify a move lookup or drive the real arm)
- [lowlevel/coord_finder/README.md](lowlevel/coord_finder/README.md) -- vision pipeline (board + piece detection from camera images)
- [lowlevel/kinematicscalibration/README.md](lowlevel/kinematicscalibration/README.md) -- PyBullet/URDF gripper-contact-point calibration tools


All the relevant code for this project is currently in lowlevel. There are yaml files to help with setting up the environments to run the different sections - see the appropriate readmes.