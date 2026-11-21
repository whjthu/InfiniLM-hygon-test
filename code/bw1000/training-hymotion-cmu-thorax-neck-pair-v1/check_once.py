"""Compare one source-relative neck transport on the original 470-frame CMU recording."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / 'code/bw1000'
LOCAL = REPO / 'local-only'
OUTPUT = LOCAL / 'training-hymotion-cmu-thorax-neck-pair-v1-original'
PINS = {
    'geometry': ('hymotion-cmu-wooden-root-v1-original/motion135-and-CPU-geometry201-preview-private.npz',
                 856445, '1aa6b6292f8a56b6a9fcf77533389533288df942b1c218d4e03c53b0de86d548'),
    'raw_FK': ('training-hymotion-cmu-asf-fk-candidate-v1-original/world-joints-native-meters-private.npz',
               2679326, '4ca26a839adf332c5525c78a3443f841f97700d608c0052e13763ac11e3d55fc'),
    'raw_hierarchy': ('training-hymotion-cmu-asf-fk-candidate-v1-original/hierarchy-and-conventions-private.json',
                      14452, 'e7374b7a18d16255506febbc6575272beca20803c8abfb8b00f256fd71032aa0'),
    'anchors': ('hymotion-cmu-to-wooden-rest-binding-v1-original/anchors-in-wooden-rest-basis-private.npz',
                410551, '39c1d9d7ec6d375c7fa4794ae93752ae107c65e87fbc722f767384198721cf35'),
}


def ensure(condition, detail):
    if not condition:
        raise ValueError(detail)


def fixed(path, size, digest):
    ensure(path.resolve(strict=True) == path and path.stat().st_size == size, 'frozen path or size differs: '+path.name)
    raw = path.read_bytes()
    ensure(sha256(raw).hexdigest() == digest, 'frozen input differs: '+path.name)
    return raw


def private(name):
    path, size, digest = PINS[name]
    return fixed(LOCAL/path, size, digest)


def metrics(positions, anchors, neck_values, mean, std):
    error = np.linalg.norm(positions - anchors, axis=1)
    z = (neck_values - mean) / std
    return dict(head_position_rmse=float(np.sqrt(np.mean(error**2))), head_position_p95=float(np.percentile(error, 95)),
                neck_rms_z=float(np.sqrt(np.mean(z*z))), neck_max_abs_z=float(np.max(np.abs(z))))


def independent_neck_source(rotations, endpoints, bases, metadata):
    names = metadata['joint_names']
    bones = {bone['name'][0]: bone for bone in metadata['bones']}
    source_axes = {}
    errors = {}
    for name in ('thorax','upperneck'):
        i = names.index(name)
        vector = np.array(bones[name]['direction'], dtype=np.float64)
        direction = vector / np.linalg.norm(vector)
        actual = endpoints[:,i] - bases[:,i-1]
        actual /= np.linalg.norm(actual, axis=1, keepdims=True)
        errors[name] = float(np.max(np.abs(rotations[:,i]@direction - actual)))
        source_axes[name] = direction
    ensure(max(errors.values()) < 1e-12, 'raw orientation disagrees with independently retained bone directions')
    torso, upper = (rotations[:,names.index(name)] for name in ('thorax','upperneck'))
    return torso.swapaxes(1,2)@upper, source_axes['upperneck'], errors


def main():
    review = SOURCE/'training-hymotion-cmu-neck-pivot-audit-v1'
    fixed(review/'audit_pair.py',12209,'96440d557b38b10c59924ade4d17e34c0886cc6a70ef1f92f0192476ff5e4358')
    sys.path.insert(0, str(review))
    from audit_pair import manifest, module

    root = SOURCE/'training-hymotion-cmu-wooden-root-v1'
    fit = SOURCE/'training-hymotion-cmu-wooden-position-fit-v1'
    bind = SOURCE/'training-hymotion-wooden-rest-binding-v1'
    manifest(root,600,'d17872c6d1664b20af9dc6e4d16d5dcf8e9e99e6a002a084726913d03bc65745')
    manifest(fit,739,'b74668bffe8303bfc655a1ed4ec627b3ea5280effffca2ca89026f6efc53239d')
    manifest(bind,355,'73214bf524166c7b86ca04d3a1461ece7ae51e0852b79f0156e6c77eb6d332e8')
    geometry = module('frozen_geometry', root/'assemble_motion.py')
    binding = module('frozen_binding', bind/'bind_rest.py')
    source_rig, decode = geometry.prior_math()
    names, rest22, _ = source_rig()
    rest, parents = geometry.full_skeleton()
    neck, head = (names.index(name) for name in ('Neck','Head'))
    ensure(neck == 12 and parents[head] == neck and names[parents[neck]] == 'Spine3'
           and np.flatnonzero(parents == neck).tolist() == [head] and not (parents == head).any(),
           'unexpected original leaf structure')

    with np.load(BytesIO(private('raw_FK')), allow_pickle=False) as raw:
        relative, source_axis, fk_error = independent_neck_source(raw['world_rotations'], raw['joint_endpoints_m'],
                                                                  raw['bone_bases_m'], json.loads(private('raw_hierarchy')))
    centers, sampled, _ = sys.modules['resample_rotations'].decimate(relative[:,None])
    with np.load(BytesIO(private('geometry')), allow_pickle=False) as original:
        frames = original['source_frame_numbers']
        features = original['cpu_geometry201']
    ensure(np.array_equal(frames,centers+1) and features.shape == (470,201), 'frozen 30 Hz timestamps differ')

    source_names, source_rest, _ = binding.cmu_rig()
    source_by_target = {name: source_names.index(anchor) for name,anchor in binding.ANCHORS.items()}
    src_hip = source_rest[source_by_target['L_Hip']] - source_rest[source_by_target['R_Hip']]
    src_spine = source_rest[source_by_target['Spine3']]
    dst_hip = rest22[names.index('L_Hip')] - rest22[names.index('R_Hip')]
    dst_spine = rest22[names.index('Spine3')] - rest22[0]
    Q = binding.basis(dst_hip,dst_spine) @ binding.basis(src_hip,src_spine).T
    target_axis = rest[head]-rest[neck]
    target_axis /= np.linalg.norm(target_axis)
    align, ambiguous = sys.modules['fit_poses'].minimum_turn((Q@source_axis)[None],target_axis[None])
    ensure(ambiguous == 0, 'rest-axis half-turn is ambiguous')
    mapping = align[0]@Q
    ensure(np.linalg.norm(mapping@source_axis-target_axis) < 1e-10 and abs(np.linalg.det(mapping)-1) < 1e-10,
           'neck mapping does not preserve the rest axis and handedness')

    original_six = features[:,3:135].reshape(470,22,6)
    candidate_six = original_six.copy()
    transported = mapping@sampled[:,0]@mapping.T
    candidate_six[:,neck] = transported[...,:2].reshape(470,6)
    _, positions = geometry.transform(decode(original_six), rest, parents)
    _, proposed = geometry.transform(decode(candidate_six), rest, parents)
    ensure(np.max(np.abs(positions[:,:22].reshape(470,66)-features[:,135:])) < 1e-6, 'original FK differs from preview')
    other = [i for i in range(len(rest)) if i != head]
    ensure(np.array_equal(positions[:,other],proposed[:,other]), 'unrelated joint positions changed')
    before_length = np.linalg.norm(positions[:,head]-positions[:,neck],axis=1)
    after_length = np.linalg.norm(proposed[:,head]-proposed[:,neck],axis=1)
    ensure(np.max(np.abs(before_length-after_length)) < 1e-9, 'original target neck-to-head length changed')

    with np.load(BytesIO(private('anchors')), allow_pickle=False) as obs:
        observed_names = obs['candidate_joint_names'].tolist()
        ensure(np.array_equal(frames,obs['frame_numbers'][centers]), 'observed anchor timestamps differ')
        anchor = obs['points'][centers,observed_names.index('Head')]
    mean = np.load(BytesIO(geometry.fixed(geometry.OFFICIAL/'Mean.npy',1736,
                   'a4bbe5f68881fe793d509f262ad3dc698a91f79158a9bde9d6f24d46ef9652bd')), allow_pickle=False)
    std = np.load(BytesIO(geometry.fixed(geometry.OFFICIAL/'Std.npy',1736,
                   '6cc86d1ca4b47fbd0e84b677c4397e51b6615abcb34e38fdd9daabbe6a5685cc')), allow_pickle=False)
    section = slice(3+neck*6,3+(neck+1)*6)
    mean, std = mean[section], np.where(std[section]<1e-3,1,std[section])
    scores = {}
    for name, subset in (('first128',slice(0,128)), ('later342_same_recording',slice(128,None)), ('whole470',slice(None))):
        scores[name] = {'original':metrics(positions[subset,head]-positions[subset,0],anchor[subset],
                                           original_six[subset,neck],mean,std),
                        'transported':metrics(proposed[subset,head]-proposed[subset,0],anchor[subset],
                                              candidate_six[subset,neck],mean,std)}
    later = scores['later342_same_recording']
    compatibility = all(later['transported'][field] <= later['original'][field]
                        for field in ('head_position_rmse','head_position_p95','neck_max_abs_z'))
    OUTPUT.mkdir()
    pair = OUTPUT/'neck-source-relative-pair-private.npz'
    np.savez_compressed(pair, frames=frames, original_neck=original_six[:,neck], candidate_neck=candidate_six[:,neck],
                        original_head=positions[:,head], candidate_head=proposed[:,head], observed_head=anchor)
    report = dict(scope='ONE_REAL_SOURCE_RELATIVE_NECK_TRANSPORT', frames=len(frames), source_frames=len(relative),
                  raw_orientation_endpoint_max_error=fk_error, sample_groups=scores,
                  candidate_does_not_worsen_later_geometry_and_channel_tail=compatibility,
                  unchanged_non_head_positions=len(other), head_length_preserved=True, candidate_training_ready=False,
                  optimizer_updates=0, GPU_steps=0, private_pair_SHA256=sha256(pair.read_bytes()).hexdigest())
    (OUTPUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(report,allow_nan=False))


if __name__ == '__main__':
    main()
