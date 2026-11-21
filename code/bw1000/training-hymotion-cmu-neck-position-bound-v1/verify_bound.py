"""Exact head-position cost for a neck-axis bound with the original parent and pivot fixed."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / 'code/bw1000'
LOCAL = REPO / 'local-only'
OUTPUT = LOCAL / 'training-hymotion-cmu-neck-position-bound-v1-original'
REVIEW = SOURCE / 'training-hymotion-cmu-neck-pivot-audit-v1'
Y = np.array([0., 1., 0.])
X = np.array([1., 0., 0.])


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def pinned(path, checksum):
    require(path.resolve(strict=True) == path, 'frozen path alias')
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == checksum, 'frozen source bytes differ: ' + path.name)
    return raw


def projection_in_y_plane(axis):
    transverse = Y - axis[..., 1:2] * axis
    norm = np.linalg.norm(transverse, axis=-1, keepdims=True)
    return np.where(norm > 1e-12, transverse/np.maximum(norm, 1e-12), X)


def frame_with_max_y_alignment(axis):
    transverse = projection_in_y_plane(axis)
    return np.stack((axis, transverse, np.cross(axis, transverse)), axis=-1)


def closest_head(rest, wanted, min_y_alignment):
    """Return an attainable minimizer and a law-of-cosines lower bound."""
    length = np.linalg.norm(rest)
    need_length = np.linalg.norm(wanted, axis=1)
    require(length > 0 and np.all(need_length > 0) and -1 <= min_y_alignment <= 1,
            'invalid rest, observed joint or channel bound')
    rest_unit, wanted_unit = rest/length, wanted/need_length[:,None]
    beta = np.arccos(np.clip(rest_unit[1], -1, 1))
    theta = np.arccos(np.clip(wanted_unit[:,1], -1, 1))
    alpha = np.arccos(min_y_alignment)
    best_theta = np.clip(theta, max(0, beta-alpha), min(np.pi, beta+alpha))
    radial = wanted_unit - wanted_unit[:,1:2]*Y
    norm = np.linalg.norm(radial, axis=1, keepdims=True)
    azimuth = np.where(norm > 1e-12, radial/np.maximum(norm, 1e-12), X)
    best_unit = np.cos(best_theta)[:,None]*Y + np.sin(best_theta)[:,None]*azimuth
    witness = frame_with_max_y_alignment(best_unit) @ frame_with_max_y_alignment(rest_unit).T
    delta = theta-best_theta
    lower = np.sqrt((need_length-length)**2 + 4*need_length*length*np.sin(delta/2)**2)
    require(np.max(np.abs(witness@rest_unit-best_unit)) < 1e-10
            and np.max(np.abs(witness.swapaxes(-1,-2)@witness-np.eye(3))) < 1e-10
            and np.max(np.abs(np.linalg.det(witness)-1)) < 1e-10
            and np.min(witness[:,1,1]) >= min_y_alignment-1e-10,
            'reconstructed closest point violates an attainable rotation constraint')
    actual = np.linalg.norm((witness@rest)-wanted, axis=1)
    require(np.max(np.abs(lower-actual)) < 1e-10, 'closest-endpoint witness and analytical lower bound differ')
    return lower, witness


def aggregate(values):
    return dict(rmse=float(np.sqrt(np.mean(values**2))), p95=float(np.percentile(values,95)), max=float(np.max(values)))


def main():
    pinned(REVIEW/'audit_pair.py', '96440d557b38b10c59924ade4d17e34c0886cc6a70ef1f92f0192476ff5e4358')
    sys.path.insert(0, str(REVIEW))
    from audit_pair import manifest, module

    root = SOURCE/'training-hymotion-cmu-wooden-root-v1'
    manifest(root,600,'d17872c6d1664b20af9dc6e4d16d5dcf8e9e99e6a002a084726913d03bc65745')
    geometry = module('frozen_bound_geometry', root/'assemble_motion.py')
    source_rig, decode = geometry.prior_math()
    names, _, _ = source_rig()
    rest, parents = geometry.full_skeleton()
    neck, head = (names.index(name) for name in ('Neck','Head'))
    torso = parents[neck]
    require(neck == 12 and parents[head] == neck and names[torso] == 'Spine3', 'original chain differs')
    source = LOCAL/'hymotion-cmu-wooden-root-v1-original/motion135-and-CPU-geometry201-preview-private.npz'
    pair = LOCAL/'training-hymotion-cmu-thorax-neck-pair-v1-original/neck-source-relative-pair-private.npz'
    with np.load(BytesIO(pinned(source,'1aa6b6292f8a56b6a9fcf77533389533288df942b1c218d4e03c53b0de86d548')),
                 allow_pickle=False) as frozen:
        features = frozen['cpu_geometry201']
        frames = frozen['source_frame_numbers']
    with np.load(BytesIO(pinned(pair,'12fba821e5634bdebb93559f73fcdb9347b163d1e6ac4441e0829c2693f7db44')),
                 allow_pickle=False) as original_pair:
        require(np.array_equal(frames,original_pair['frames']) and features.shape == (470,201), 'original frames differ')
        anchor, original_head = original_pair['observed_head'], original_pair['original_head']
        transported_head = original_pair['candidate_head']
        transported_neck_six = original_pair['candidate_neck']
    local = decode(features[:,3:135].reshape(470,22,6))
    rotations, positions = geometry.transform(local,rest,parents)
    require(np.array_equal(positions[:,head],original_head), 'independent original head replay differs')
    target_world = anchor + positions[:,0]
    wanted = (rotations[:,torso].swapaxes(1,2) @ (target_world-positions[:,neck])[...,None])[...,0]
    rest_edge = rest[head]-rest[neck]
    mean = np.load(BytesIO(geometry.fixed(geometry.OFFICIAL/'Mean.npy',1736,
                   'a4bbe5f68881fe793d509f262ad3dc698a91f79158a9bde9d6f24d46ef9652bd')),allow_pickle=False)
    std = np.load(BytesIO(geometry.fixed(geometry.OFFICIAL/'Std.npy',1736,
                   '6cc86d1ca4b47fbd0e84b677c4397e51b6615abcb34e38fdd9daabbe6a5685cc')),allow_pickle=False)
    channel = 3+neck*6+3
    require(channel == 78 and std[channel] > 1e-3 and mean[channel]+5*std[channel] >= 1,
            'the single-channel five-sigma limit has an additional upper constraint')
    threshold = float(mean[channel]-5*std[channel])
    lower, witness = closest_head(rest_edge, wanted, threshold)
    original_error = np.linalg.norm(original_head-target_world, axis=1)
    transported_error = np.linalg.norm(transported_head-target_world, axis=1)
    require(np.all(transported_neck_six[:,3] >= threshold) and np.all(lower <= transported_error + 1e-8),
            'frozen transported feasible point contradicts the minimum')
    improvement_possible = lower <= original_error + 1e-8
    necessary_pivot_shift = np.maximum(0, lower-original_error)
    scopes = {}
    for label, part in (('first128',slice(0,128)), ('later342_same_recording',slice(128,None)), ('whole470',slice(None))):
        scopes[label] = dict(original_head_error=aggregate(original_error[part]),
                             theoretical_best_head_error=aggregate(lower[part]),
                             previous_transported_head_error=aggregate(transported_error[part]),
                             frames_with_unavoidable_position_regression=int((~improvement_possible[part]).sum()),
                             pivot_shift_lower_bound_to_match_original=aggregate(necessary_pivot_shift[part]))
    OUTPUT.mkdir(mode=0o700)
    data = OUTPUT/'one_channel_bound_witness_private.npz'
    np.savez_compressed(data, frames=frames, fixed_pivot_minimum_head_error=lower,
                        minimum_pivot_shift_bound=necessary_pivot_shift, witness_local_neck=witness)
    report = dict(scope='EXACT_FIXED_PARENT_PIVOT_AND_LENGTH_HEAD_BOUND',frames=len(frames),
                  channel=channel,z_limit=5,required_R11_min=threshold,subsets=scopes,
                  all_six_channels_constrained=False,witness_is_training_candidate=False,
                  actual_GPU_steps=0,actual_optimizer_updates=0,candidate_training_ready=False,
                  witness_SHA256=sha256(data.read_bytes()).hexdigest())
    (OUTPUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(report,allow_nan=False))


if __name__ == '__main__':
    main()
