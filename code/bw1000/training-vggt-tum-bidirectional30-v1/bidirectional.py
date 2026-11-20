"""Deterministic reverse views of the two already frozen TUM training pairs.

The input uses OpenCV world-to-camera poses with frame zero as the world gauge.
No new media, labels, scene, or held-out example is collected here.
"""
import math


ORDER = ((0, "forward"), (1, "forward"), (0, "reverse"), (1, "reverse"))
FRAME_FIELDS = ("images", "depths", "intrinsics", "point_masks")


def checked_pose(value):
    rows = [[float(x) for x in row] for row in value]
    if len(rows) != 3 or any(len(row) != 4 for row in rows):
        raise ValueError("Expected a rigid 3x4 world-to-camera pose")
    if any(not math.isfinite(x) for row in rows for x in row):
        raise ValueError("Nonfinite pose")
    r = [row[:3] for row in rows]
    if any(abs(sum(r[i][k]*r[j][k] for k in range(3)) - int(i == j)) > 1e-6
           for i in range(3) for j in range(3)):
        raise ValueError("Pose rotation is not orthonormal")
    det = (r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
           - r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
           + r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    if abs(det-1) > 1e-6:
        raise ValueError("Pose rotation is not proper")
    return rows


def inverse_pose(value):
    p = checked_pose(value)
    return [[p[j][i] for j in range(3)] +
            [-sum(p[j][i]*p[j][3] for j in range(3))] for i in range(3)]


def reverse_pair(raw):
    if set(raw) != {*FRAME_FIELDS, "seq_name", "ids", "extrinsics"}:
        raise ValueError("Original TUM batch keys differ")
    if (len(raw["ids"]) != 2 or any(type(i) is not int for i in raw["ids"])
            or not raw["ids"][0] < raw["ids"][1]
            or len(raw["extrinsics"]) != 2
            or any(len(raw[k]) != 2 for k in FRAME_FIELDS)):
        raise ValueError("Expected one original forward ordered pair")
    first = checked_pose(raw["extrinsics"][0])
    identity = [[float(i == j) for j in range(4)] for i in range(3)]
    if any(abs(first[i][j]-identity[i][j]) > 1e-6 for i in range(3) for j in range(4)):
        raise ValueError("Original pair does not use the first-camera gauge")
    # New first camera is old camera 1.  E'_0=I, E'_1=E_0 inv(E_1).
    # Merely reversing the extrinsics array would change the target convention.
    result = {key: [raw[key][1], raw[key][0]] for key in FRAME_FIELDS}
    result.update(seq_name=raw["seq_name"], ids=list(reversed(raw["ids"])),
                  extrinsics=[identity, inverse_pose(raw["extrinsics"][1])])
    return result


def expand(originals, geometry):
    if len(originals) != 2 or len(geometry) != 2:
        raise ValueError("Exactly two pinned physical training pairs required")
    result, evidence = [], []
    for index, direction in ORDER:
        batch = originals[index] if direction == "forward" else reverse_pair(originals[index])
        result.append(batch)
        evidence.append(dict(source_pair_index=index, direction=direction,
                             physical_sequence=batch["seq_name"], input_frame_ids=list(batch["ids"]),
                             original_geometry=geometry[index],
                             gauge="first_camera_world_to_camera_OpenCV",
                             derived_media=True, new_media=False))
    return result, evidence
