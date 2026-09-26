import sys
import struct
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
ASSETS_DIR = paths.ASSETS_DIR

def read_binary_stl_vertices(path):
    data = path.read_bytes()
    tri_count = struct.unpack_from("<I", data, 80)[0]
    verts = np.empty((tri_count * 3, 3), dtype=np.float64)
    offset = 84
    for t in range(tri_count):
        for v in range(3):
            x, y, z = struct.unpack_from("<fff", data, offset + 12 + v * 12)
            verts[t * 3 + v] = (x, y, z)
        offset += 50
    return verts

def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx

PARTS = [
    ("base_motor_holder_so101_v1.stl", (-0.00636471, -9.94414e-05, -0.0024), (1.5708, -1.67685e-15, 1.5708)),
    ("base_so101_v2.stl", (-0.00636471, -8.97657e-09, -0.0024), (1.5708, -2.78073e-29, 1.5708)),
    ("sts3215_03a_v1.stl", (0.0263353, -8.97657e-09, 0.0437), (-8.21148e-16, 7.84513e-18, 1.249e-15)),
    ("waveshare_mounting_plate_so101_v2.stl", (-0.0309827, -0.000199441, 0.0474), (1.5708, -1.35493e-14, 1.5708)),
]

best = None
for stl_name, xyz, rpy in PARTS:
    verts = read_binary_stl_vertices(ASSETS_DIR / stl_name)
    R = rpy_to_matrix(*rpy)
    verts_link = (R @ verts.T).T + np.array(xyz)
    zmin = verts_link[:, 2].min()
    idx = verts_link[:, 2].argmin()
    print(f"{stl_name:42s} zmin={zmin:+.5f}  at local xyz={verts_link[idx]}")
    if best is None or zmin < best[1]:
        best = (stl_name, zmin, verts_link[idx])

print()
print(f"TRUE lowest point of base_link: {best[0]}  z={best[1]:+.5f}  xyz={best[2]}")
