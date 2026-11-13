# Lens calibration photos

Photos for recalibrating the wrist camera's lens (and the stationary C920's), for Asha. The
bundled wrist lens file (`click_to_3d*/bundle/wrist_fisheye.yaml`, f = 1400 px) comes from a grid
search that never tries below 700 px. Forward kinematics puts the real focal length near 516 px,
which is why the click tool's PnP puts the camera 2.7× too far from the board. Every photo shows
the same square (d4), with its corners marked the click-tool way: green on the corner nearest
a1, then counter-clockwise in the picture.

Everything here runs in the click3d venv, from this folder, and imports
`../click_to_3d_oliver/click3d_shared.py` and `../paths.py`.

| File | What it does |
| --- | --- |
| `plan_wrist_poses.py` | Plans 15 wrist-camera views and checks the routes to and from them (nothing moves) |
| `collect_wrist_photos.py` | Drives the arm through the plan and takes a raw photo at each view. A dry run without `--execute` |
| `label_photos.py` | Marks the square's corners in each photo, or takes hand-held photos (e.g. the C920) live |
| `output/` | `wrist_pose_plan.json` / `.png`, `wrist_photos_<time>/`, `<camera>_<time>/` |

## Wrist camera

```bash
~/venvs/click3d/bin/python plan_wrist_poses.py          # writes output/wrist_pose_plan.json and .png
~/venvs/click3d/bin/python collect_wrist_photos.py      # dry run: rebuilds and checks every route
~/venvs/click3d/bin/python collect_wrist_photos.py --execute
~/venvs/click3d/bin/python label_photos.py --camera wrist --images output/wrist_photos_<time>
```

The plan keeps to what the arm has already done with the camera mount on:

- **Joints** stay in the range recorded on the real arm with the mount fitted
  (`SAFE_LIMITS`: pan −18.5…97.5, lift −113…56, elbow −73.5…97.5). Wrist flex is capped at 95
  (the URDF limit, and the probe's via value). Wrist roll stays inside probe_xy_pose.py's
  −150…−30. Outside that range the camera once swung round into the motors. The views stay
  further in, with lift no lower than home and roll 5° from each end.
- **Routes** all pass through the via pose `probe_xy_pose.py` uses above d4, which the arm has
  run many times. The arm leaves rest the way the probe does, pans and rolls only at that
  height, then goes straight down to the view. Every move back reverses this.
- **Every 0.5° step** of every route is checked with forward kinematics. The fixed jaw, gripper,
  wrist, camera mount and camera must stay 25–45 mm above the board, or 10–30 mm above the table
  off the board, and nothing may pass the board's far edge, where the tripod stands. The
  smallest margin in the current plan is 44 mm over the board, plus 20 mm over the table at
  rest, which is the probe's own route.

`collect_wrist_photos.py --execute` is careful in these ways:

- **Camera first:** it opens the camera before anything else, and never touches the arm if that fails.
- **No torque change on connect:** it opens the serial port without lerobot's configure step. If
  the arm is limp, it must be at rest; the script sets every goal to the present position, then
  switches torque on. If the arm is already held, it carries on from the goal it holds.
- **Gripper:** it refuses an open gripper and never moves the gripper.
- **Confirmation:** before every move it rebuilds the route from the last command, checks it
  again and shows it. Enter moves, `s` skips the view, and `q` goes back to rest.
- **Speed:** 12°/s by default.
- **Stopping:** it stops and holds where it is, with torque on, if any joint falls more than
  10° behind its command (`--max-lag-deg`), if it doesn't settle on its target, on Ctrl-C, or
  on any error.

After a stop, support the arm, check the camera mount, then:

```bash
~/venvs/click3d/bin/python collect_wrist_photos.py --return-home --execute   # back to rest, then release
~/venvs/click3d/bin/python collect_wrist_photos.py --release --execute       # torque off, if that refuses
```

Every view must also show a complete 5×5 (or 4×4) block of inner corners round the square, at
least 20 px inside the frame and clear of the jaws, whether the focal length turns out to be 500
or 620 px. On the first real photo the detector took a jaw tip for a corner, which wrecked the fit.
Views that can't do that are left out: 10 of the 15 qualify. A simulated calibration from those
10 recovers the focal length to ±1.5–4 px (true 516 or 580, 0.3–1 px corner noise). The plan uses
550 px, so the real square sits near the planned outline rather than exactly on it. Clear the
pieces off the board first, because they hide corners. The side views only reach about 20° of tilt (wrist flex runs out), and
views from beyond the board are out of reach. The diagonal and robot-side views (38–45°) give
the focal length its tilt. Re-plan after the friction fit's joint offsets change; the
collector refuses a plan made with different offsets.

## Stationary C920

```bash
~/venvs/click3d/bin/python label_photos.py --camera stationary --square d4
```

Move the camera by hand: lift it off the tripod, or move the board. The tool suggests 15 views:
face-on with the board in each corner of the frame, tilted ~40° from all sides and both
diagonals, one rolled 45°, and one farther away. Keep it 35–80 cm from the board, because its
focus stays locked at 40, which was set for the tripod distance. Afterwards, put it back and
press `c` in the click tool.
