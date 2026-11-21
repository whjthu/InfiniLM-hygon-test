"""Measure area-sampled distances to the actual triangles in a fixed frame."""

import numpy as np


POINTS_PER_MESH = 2048
POINT_BLOCK = 32
FACE_BLOCK = 512
THRESHOLD = 0.01


def xyz(values, name):
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0 or array.dtype.kind not in "fiu":
        raise ValueError(name + " must be nonempty N×3 coordinates")
    if not np.isfinite(array).all():
        raise ValueError(name + " contains nonfinite coordinates")
    return array.astype(np.float64, copy=False)


def mesh(vertices, faces):
    vertices = xyz(vertices, "mesh vertices")
    faces = np.asarray(faces)
    if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or faces.dtype.kind not in "iu"
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("mesh faces must be valid triangle indices")
    triangles = vertices[faces]
    origin = triangles[:, 0]
    ab = triangles[:, 1] - origin
    ac = triangles[:, 2] - origin
    normal = np.cross(ab, ac)
    normal_sq = np.einsum("ij,ij->i", normal, normal)
    if not np.isfinite(normal_sq).all() or np.any(normal_sq <= 0):
        raise ValueError("mesh contains degenerate triangles")
    return triangles, normal, normal_sq


def block_squared(points, triangles, normals, normal_sq):
    a, b, c = np.moveaxis(triangles, 1, 0)
    ab, ac = b - a, c - a
    ap = points[:, None, :] - a
    ab_sq = np.einsum("ij,ij->i", ab, ab)
    ac_sq = np.einsum("ij,ij->i", ac, ac)
    overlap = np.einsum("ij,ij->i", ab, ac)
    proj_ab = np.einsum("pij,ij->pi", ap, ab)
    proj_ac = np.einsum("pij,ij->pi", ap, ac)
    v = (ac_sq * proj_ab - overlap * proj_ac) / normal_sq
    w = (ab_sq * proj_ac - overlap * proj_ab) / normal_sq
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)
    plane = np.einsum("pij,ij->pi", ap, normals)
    best = np.where(inside, plane * plane / normal_sq, np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        rel = points[:, None, :] - start
        edge_sq = np.einsum("ij,ij->i", edge, edge)
        t = np.clip(np.einsum("pij,ij->pi", rel, edge) / edge_sq, 0.0, 1.0)
        delta = rel - t[..., None] * edge
        np.minimum(best, np.einsum("pij,pij->pi", delta, delta), out=best)
    return best.min(axis=1)


def distances(points, vertices, faces):
    points = xyz(points, "query samples")
    triangles, normals, normal_sq = mesh(vertices, faces)
    best = np.full(len(points), np.inf)
    for start in range(0, len(points), POINT_BLOCK):
        query = points[start:start + POINT_BLOCK]
        current = best[start:start + len(query)]
        for first in range(0, len(triangles), FACE_BLOCK):
            block = slice(first, first + FACE_BLOCK)
            np.minimum(current, block_squared(query, triangles[block], normals[block], normal_sq[block]),
                       out=current)
    if not np.isfinite(best).all():
        raise ValueError("invalid point-to-surface distance")
    return np.sqrt(best)


def metrics(reference, prediction):
    """Each arm holds (vertices, triangle indices, its fixed area samples)."""
    ref_v, ref_f, ref_points = reference
    pred_v, pred_f, pred_points = prediction
    if np.shape(ref_points) != (POINTS_PER_MESH, 3) or np.shape(pred_points) != (POINTS_PER_MESH, 3):
        raise ValueError("both full 2048-point denominators are required")
    forward = distances(pred_points, ref_v, ref_f)
    reverse = distances(ref_points, pred_v, pred_f)
    precision_count = int((forward <= THRESHOLD).sum())
    recall_count = int((reverse <= THRESHOLD).sum())
    precision = precision_count / POINTS_PER_MESH
    recall = recall_count / POINTS_PER_MESH
    fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return dict(sampled_surface_mean_L1=float((forward.mean() + reverse.mean()) / 2),
                sampled_surface_F_at_1percent=fscore, precision=precision, recall=recall,
                within_threshold_predicted=precision_count, within_threshold_reference=recall_count,
                max_predicted_distance=float(forward.max()), max_reference_distance=float(reverse.max()),
                predicted_points=len(forward), reference_points=len(reverse), threshold=THRESHOLD)
