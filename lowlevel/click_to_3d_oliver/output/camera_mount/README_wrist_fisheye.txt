Wrist fisheye lens files for click_to_3d (m → fisheye):

  wrist_fisheye.yaml
    DEFAULT. OpenCV fisheye.calibrate on all a1-indexed grown corners
    (Nov-3 d4 set). fx≈541 fy≈547, free principal point, RMS≈0.99 px.
    Produced by run_standard_wrist_fisheye.py.

  wrist_fisheye_status_quo.yaml
    Alternate. Earlier dense-grid interim (f=540, cx/cy fixed at centre,
    RMS≈0.40 px). Keep for A/B if the default looks off on hardware.

The old f=1400 / k1=-1.6 bundle has been removed.
