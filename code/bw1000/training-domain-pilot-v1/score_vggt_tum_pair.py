"""Score a paired VGGT camera-head comparison on frozen development RGB-D pairs."""

from hashlib import sha256
import sys
from math import isfinite
from time import perf_counter

import numpy as np


BASE_HEAD = "90d2dc64acde99cd90bc4fb4a71a5fbda3de9267719c84153ff65a679cbaa202"
UPDATED_HEAD = "fb84a5c4a5678cf724f29fc2a9f7a2d8b0f4d10bfd5a04f2e2d8f684454dec11"
AGGREGATOR = "6590aab13e36a0309bd601226cc053c59fbef3f633ae3ce478fe329dd247aac4"
WEIGHT_SHA = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
PILOT_GEOMETRY_SHA = "b98b2473335cecd21c65be9dfb64d908b177802adfa74ce5750a5e4c4efeac50"
HEAD_STAT = (864719552, 23356225900, 1794863955000000000)


def relative_pose(extrinsics):
    """Use OpenCV world-to-camera extrinsics; the output maps first camera to second."""
    pose = np.asarray(extrinsics, dtype=np.float64)
    if pose.shape != (2, 3, 4) or not np.isfinite(pose).all():
        raise ValueError("expected two finite 3x4 camera matrices")
    rotations = pose[:, :, :3]
    if not (np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3)
            and np.allclose(np.linalg.det(rotations), 1, atol=1e-3)):
        raise ValueError("camera rotation differs from SO(3)")
    rotation = rotations[1] @ rotations[0].T
    translation = pose[1, :, 3] - rotation @ pose[0, :, 3]
    return rotation, translation


def angular_errors(predicted, truth):
    predicted_r, predicted_t = relative_pose(predicted)
    truth_r, truth_t = relative_pose(truth)
    disagreement = predicted_r @ truth_r.T
    cosine = (np.trace(disagreement) - 1) / 2
    rotation_deg = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    norms = np.linalg.norm(predicted_t), np.linalg.norm(truth_t)
    if norms[1] <= 1e-8:
        raise ValueError("ground truth camera displacement is degenerate")
    if norms[0] <= 1e-8:
        return dict(rotation_deg=rotation_deg, translation_direction_deg=None,
                    predicted_translation_degenerate=True, pair_error_deg=None)
    translation_deg = float(np.degrees(np.arccos(np.clip(predicted_t @ truth_t / np.prod(norms), -1, 1))))
    return dict(rotation_deg=rotation_deg, translation_direction_deg=translation_deg,
                predicted_translation_degenerate=False, pair_error_deg=max(rotation_deg, translation_deg))


def stat_identity(path):
    info = path.stat()
    return info.st_size, info.st_ino, info.st_mtime_ns


def stream_sha256(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_development(rows):
    """Report exact areas under empirical angular recall, retaining failed pairs."""
    identifiers = {(row["sequence"], tuple(row["frame_ids"])) for row in rows}
    if not rows or len(identifiers) != len(rows):
        raise ValueError("development results must identify distinct frame pairs")
    summary = {"pairs": len(rows), "sequences": len({row["sequence"] for row in rows})}
    for arm in ("baseline", "updated"):
        errors = [row[arm]["pair_error_deg"] for row in rows]
        if any(error is not None and (not isfinite(error) or not 0 <= error <= 180) for error in errors):
            raise ValueError("invalid development angular error")
        summary[arm] = dict(failed_pairs=errors.count(None),
                            **{f"development_AUC_at_{threshold}": sum(
                                max(0, 1 - error / threshold) for error in errors if error is not None
                            ) / len(rows) for threshold in (5, 15, 30)})
    return summary


def score_native_pairs(cases, *, source, weights, trained_head):
    """Run each frozen pair under both heads while keeping one model instance resident."""
    seen = set()
    for raw_pair, sequence, pair_sha in cases:
        identity = (sequence, tuple(raw_pair.get("frame_identity", raw_pair["ids"])))
        if (pair_sha == PILOT_GEOMETRY_SHA or sequence in (
                "freiburg3_structure_texture_near", "freiburg3_long_office_household")
                or raw_pair["seq_name"] != sequence or identity in seen):
            raise ValueError("development evaluation needs a distinct pair outside training and blind sequences")
        seen.add(identity)
    if not seen:
        raise ValueError("development evaluation needs a frozen pair")
    import torch
    from gencuda.backends.vggt_training import _module_sha, _tensor_batch, valid_depth_batch
    from gencuda.runtime import autocast_bf16, synchronize

    if torch.cuda.device_count() != 1 or not torch.version.hip:
        raise ValueError("evaluation requires the one leased HIP device")
    if stat_identity(trained_head) != HEAD_STAT:
        raise ValueError("original trained head file identity differs")
    sys.path.insert(0, str(source))
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if stream_sha256(weights) != WEIGHT_SHA:
        raise ValueError("original model bytes differ")
    batches = [_tensor_batch(torch, case[0]) for case in cases]
    if any(tuple(batch["images"].shape) != (1, 2, 3, 224, 224)
           or not valid_depth_batch(batch["point_masks"]) for batch in batches):
        raise ValueError("a frozen development batch has invalid geometry")
    raw_state = torch.load(weights, map_location="cpu", weights_only=True)
    state = raw_state.get("model", raw_state) if isinstance(raw_state, dict) else raw_state
    model = VGGT(enable_camera=True, enable_point=False, enable_depth=False, enable_track=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or not all(key.startswith(("point_head.", "depth_head.", "track_head.")) for key in unexpected):
        raise ValueError("baseline model state differs from the camera-only architecture")
    del state, raw_state
    if _module_sha(model.camera_head) != BASE_HEAD or _module_sha(model.aggregator) != AGGREGATOR:
        raise ValueError("baseline weights differ from the first-update original")
    model.requires_grad_(False).eval().cuda()
    def measure(batch):
        images = batch["images"].cuda()
        truth = batch["extrinsics"][0].numpy()
        synchronize()
        start = perf_counter()
        with torch.inference_mode(), autocast_bf16():
            output = model(images)
        synchronize()
        forward_s = perf_counter() - start
        if tuple(output["pose_enc"].shape) != (1, 2, 9) or not output["pose_enc"].is_cuda:
            raise ValueError("native camera prediction shape or device differs")
        extrinsics, _ = pose_encoding_to_extri_intri(output["pose_enc"].float(), images.shape[-2:],
                                                     build_intrinsics=False)
        return dict(forward_s=forward_s, **angular_errors(extrinsics[0].cpu().numpy(), truth))

    baseline = [measure(batch) for batch in batches]
    head_state = torch.load(trained_head, map_location="cpu", weights_only=True)
    if stat_identity(trained_head) != HEAD_STAT:
        raise ValueError("trained head changed while loading")
    model.camera_head.load_state_dict(head_state, strict=True)
    del head_state
    if _module_sha(model.camera_head) != UPDATED_HEAD:
        raise ValueError("trained head differs from the saved first update")
    updated = [measure(batch) for batch in batches]
    if _module_sha(model.aggregator) != AGGREGATOR or _module_sha(model.camera_head) != UPDATED_HEAD:
        raise ValueError("evaluation mutated model weights")
    return [dict(sequence=sequence, pair_sha256=pair_sha,
                 frame_ids=list(raw_pair.get("frame_identity", raw_pair["ids"])),
                 baseline=before, updated=after)
            for (raw_pair, sequence, pair_sha), before, after in zip(cases, baseline, updated, strict=True)]


def score_native(raw_pair, *, sequence, pair_sha, source, weights, trained_head):
    row, = score_native_pairs([(raw_pair, sequence, pair_sha)], source=source, weights=weights,
                              trained_head=trained_head)
    return dict(**row, paired_cases=1,
                formal_CO3D_eligible=False, formal_performance_eligible=False, optimizer_updates=0,
                orientation_convention="first camera to second, directed translation without sign folding")
