"""Fit public WoodenMesh joint positions to the verified CMU anchor trajectories."""

from hashlib import sha256
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRIVATE = REPO/'local-only'
SOURCE = PRIVATE/'hymotion-cmu-to-wooden-rest-binding-v1-original/anchors-in-wooden-rest-basis-private.npz'
SECTION = HERE.parent/'training-hymotion-wooden-rest-binding-v1'
OUTPUT = PRIVATE/'hymotion-cmu-wooden-position-fit-v1-original'
UPSTREAM = HERE.parent/'training-hymotion-target-meta-CPU-v2/upstream/hymotion/pipeline/body_model.py'
SWEEPS = 12


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, length, sha):
    with path.open('rb') as stream:
        data = stream.read(length+1)
    need(len(data) == length and sha256(data).hexdigest() == sha, 'pinned input differs: '+path.name)
    return data


def source_rig():
    manifest = json.loads(pinned(SECTION/'SOURCE_FILES.json',355,
                                 '73214bf524166c7b86ca04d3a1461ece7ae51e0852b79f0156e6c77eb6d332e8'))
    entry = SECTION/'bind_rest.py'
    member = manifest['files'][entry.name]
    pinned(entry,member['bytes'],member['SHA256'])
    specification = importlib.util.spec_from_file_location('pinned_cmu_wooden_rest',entry)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.wooden_rig()


def skew(x):
    a,b,c = x.T
    zero = np.zeros_like(a)
    return np.stack((zero,-c,b,c,zero,-a,-b,a,zero),axis=-1).reshape(-1,3,3)


def minimum_turn(source, target):
    cross = np.cross(source,target)
    dot = np.clip((source*target).sum(axis=1),-1.,1.)
    opposite = dot < -1+1e-10
    k = skew(cross)
    result = np.eye(3)+k+(k@k)/np.where(opposite,1.,1+dot)[:,None,None]
    if np.any(opposite):
        x = source[opposite]
        # The shortest direction for an exact half-turn is underdetermined.
        fallback = np.eye(3)[np.argmin(np.abs(x),axis=1)]
        axis = np.cross(x,fallback)
        axis /= np.linalg.norm(axis,axis=1)[:,None]
        result[opposite] = 2*axis[:,:,None]*axis[:,None,:]-np.eye(3)
    return result,int(opposite.sum())


def best_rotation(before, after):
    cross_covariance = before.swapaxes(1,2)@after
    u,s,vt = np.linalg.svd(cross_covariance)
    v = vt.swapaxes(1,2)
    ut = u.swapaxes(1,2)
    correction = np.ones_like(s)
    correction[:,2] = np.where(np.linalg.det(v@ut)<0,-1.,1.)
    rotation = (v*correction[:,None,:])@ut
    underdetermined = s[:,1] < np.maximum(s[:,0]*1e-10,1e-15)
    rotation[underdetermined], opposite = minimum_turn(u[underdetermined,:,0],v[underdetermined,:,0])
    rotation[s[:,0]<1e-15] = np.eye(3)
    return rotation,int(underdetermined.sum()),opposite


def descendant_mask(parents):
    count = len(parents)
    desc = np.eye(count,dtype=bool)
    for child in range(count-1,0,-1):
        desc[parents[child]] |= desc[child]
    return desc


def position_fit(rest,parents,anchor_ids,observations,sweeps=SWEEPS):
    frames = observations.shape[0]
    joints = len(rest)
    estimate = np.broadcast_to(rest-rest[0],(frames,joints,3)).copy()
    world = np.broadcast_to(np.eye(3),(frames,joints,3,3)).copy()
    desc = descendant_mask(parents)
    active = []
    for j in reversed(range(joints)):
        offspring = desc[j].copy()
        offspring[j] = False
        mask = offspring[anchor_ids]
        if np.any(mask):
            active.append((j,offspring,mask))
    errors = [float(np.mean((estimate[:,anchor_ids]-observations)**2))]
    ambiguous = opposite = 0
    for _ in range(sweeps):
        for joint,offspring,mask in active:
            origin = estimate[:,[joint]]
            rotation,singular,half_turns = best_rotation(estimate[:,anchor_ids[mask]]-origin,observations[:,mask]-origin)
            estimate[:,offspring] = (estimate[:,offspring]-origin)@rotation.swapaxes(1,2)+origin
            world[:,desc[joint]] = rotation[:,None]@world[:,desc[joint]]
            ambiguous += singular
            opposite += half_turns
        errors.append(float(np.mean((estimate[:,anchor_ids]-observations)**2)))
        need(errors[-1] <= errors[-2]+1e-10, 'same-anchor fit objective increased')
    local = world.copy()
    local[:,1:] = world[:,parents[1:]].swapaxes(2,3)@world[:,1:]
    return estimate,local,errors,ambiguous,opposite,[j for j,_,_ in active]


def replay_fk(local, rest, parents):
    """Rebuild with homogeneous pose multiplication as in the pinned body model."""
    shape = local.shape[:2]
    transforms = np.broadcast_to(np.eye(4),(*shape,4,4)).copy()
    transforms[:,:,:3,:3] = local
    transforms[:,1:,:3,3] = rest[1:]-rest[parents[1:]]
    for j in range(1,len(rest)):
        transforms[:,j] = transforms[:,parents[j]]@transforms[:,j]
    return transforms[:,:,:3,3]


def summarize(estimate,external,ids,names):
    errors = np.linalg.norm(estimate[:,ids]-external,axis=-1)
    return dict(rms=float(np.sqrt(np.mean(errors**2))),p95=float(np.percentile(errors,95)),max=float(errors.max()),
                by_joint_rms={name:float(np.sqrt(np.mean(errors[:,i]**2))) for i,name in enumerate(names)})


def real_recording():
    begin = time.monotonic()
    # The linked native implementation is the basis of the independent FK replay.
    body = UPSTREAM.read_bytes()
    need(sha256(body).hexdigest() == '01eebf66d87f0ee10ebbd2e7b8ab1a2dd4c581661be99d0597c5286854e95443',
         'original body model code differs')
    names,rest,parents = source_rig()
    raw = pinned(SOURCE,410551,'39c1d9d7ec6d375c7fa4794ae93752ae107c65e87fbc722f767384198721cf35')
    with np.load(BytesIO(raw),allow_pickle=False) as captured:
        frames = captured['frame_numbers']
        matched = captured['candidate_joint_names'].tolist()
        observations = captured['points']
    need(observations.shape == (940,20,3) and np.isfinite(observations).all() and np.array_equal(frames,np.arange(1,941))
         and set(names)-set(matched)=={'L_Collar','R_Collar'} and matched[0]=='Pelvis' and not observations[:,0].any(),
         'the entire verified, pelvic-centered capture is required')
    ids = np.array([names.index(n) for n in matched])
    baseline = np.broadcast_to(rest-rest[0],(len(frames),len(rest),3))
    fit,local,errors,singular,opposite,active = position_fit(rest,parents,ids,observations)
    replay = replay_fk(local,rest,parents)
    fk_delta = float(np.max(np.abs(replay-fit)))
    residual = np.max(np.abs(local.swapaxes(2,3)@local-np.eye(3)))
    determinant = np.max(np.abs(np.linalg.det(local)-1))
    bone_lengths = np.linalg.norm(rest[1:]-rest[parents[1:]],axis=-1)
    drift = float(np.max(np.abs(np.linalg.norm(fit[:,1:]-fit[:,parents[1:]],axis=-1)-bone_lengths)))
    need(max(fk_delta,residual,determinant,drift)<1e-10 and np.isfinite(fit).all(), 'fit fails positional geometry')
    delta = local[1:]@local[:-1].swapaxes(2,3)
    angles = np.degrees(np.arccos(np.clip((np.trace(delta,axis1=2,axis2=3)-1)/2,-1,1)))
    OUTPUT.mkdir()
    private = OUTPUT/'positional-fit-not-training-private.npz'
    np.savez_compressed(private,frame_numbers=frames,joint_names=names,local_rotations=local,positions=fit)
    body = private.read_bytes()
    result = dict(scope='GEOMETRIC_POSITION_ONLY_LOCAL_CANDIDATE',frames=len(frames),source_fps=60,sweeps=SWEEPS,
                  target_body_joints=len(names),measured_anchor_candidates=len(matched),optimized_joints=[names[j] for j in active],
                  unobserved_collars_estimated_from_descendants=sorted(set(names)-set(matched)),
                  rank1_or_degenerate_updates=singular,ambiguous_half_turn_updates=opposite,
                  anchor_squared_component_loss=errors,baseline=summarize(baseline,observations,ids,matched),
                  position_fit=summarize(fit,observations,ids,matched),position_units='WoodenMesh rig units',
                  independent_FK_replay_max_abs=fk_delta,maximum_bone_length_error=drift,
                  local_rotation_orthogonality_max_abs=float(residual),local_rotation_det_max_abs=float(determinant),
                  adjacent_frame_rotation_p95_degrees=float(np.percentile(angles,95)),
                  adjacent_frame_rotation_max_degrees=float(angles.max()),
                  private_arrays=dict(bytes=len(body),SHA256=sha256(body).hexdigest()),wall_seconds=time.monotonic()-begin,
                  absolute_root_trajectory_qualified=False,model_feature135_qualified=False,cached201_qualified=False,
                  actual_resampling=False,actual_text_embeddings=0,actual_GPU_steps=0,actual_optimizer_updates=0,READY_added=0)
    with (OUTPUT/'result.json').open('x',encoding='utf8',newline='\n') as handle:
        json.dump(result,handle,allow_nan=False,sort_keys=True,indent=2)
        handle.write('\n')
    return result


if __name__ == '__main__':
    value = real_recording()
    print(json.dumps({k:value[k] for k in ('frames','target_body_joints','measured_anchor_candidates',
        'independent_FK_replay_max_abs','adjacent_frame_rotation_max_degrees','baseline','position_fit','READY_added')},allow_nan=False))
