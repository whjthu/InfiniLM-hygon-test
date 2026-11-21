"""Choose a continuous representative of rotations not determined by joint positions."""

from hashlib import sha256
from io import BytesIO
import json
import time

import numpy as np

from fit_poses import PRIVATE, minimum_turn, need, pinned, replay_fk, source_rig


ORIGINAL = PRIVATE/'hymotion-cmu-wooden-position-fit-v1-original'
OUTPUT = PRIVATE/'hymotion-cmu-wooden-position-twist-v1-original'
FIT_SHA = 'e5e20c3a7d55b3e39db360afcf041e23a6e08f8d5a41e9494c494b9eec7ad294'


def retwist(local, rest, parents):
    before = local.copy()
    for j in range(1,len(rest)):
        before[:,j] = before[:,parents[j]]@local[:,j]
    after = before.copy()
    children = [np.flatnonzero(parents==j) for j in range(len(rest))]
    single = np.array([j for j,c in enumerate(children) if len(c)==1],dtype=int)
    leaf = [j for j,c in enumerate(children) if j and not len(c)]
    child = [children[j][0] for j in single]
    rest_axis = rest[child]-rest[single]
    length = np.linalg.norm(rest_axis,axis=1)
    need((length > 0).all(), 'zero-length bone has no observable direction')
    rest_axis /= length[:,None]
    desired = (before[:,single]@rest_axis[None,:,:,None])[...,0]
    previous = np.broadcast_to(np.eye(3),(len(single),3,3))
    half_turns = 0
    for t,column in enumerate(desired):
        start = (previous@rest_axis[:,:,None])[...,0]
        turn,opposite = minimum_turn(start,column)
        previous = turn@previous
        after[t,single] = previous
        half_turns += opposite
    if leaf:
        after[:,leaf] = after[:,parents[leaf]]
    updated = after.copy()
    updated[:,1:] = after[:,parents[1:]].swapaxes(2,3)@after[:,1:]
    return updated,single.tolist(),leaf,half_turns


def rotation_summary(local,names):
    delta = local[1:]@local[:-1].swapaxes(2,3)
    angles = np.degrees(np.arccos(np.clip((np.trace(delta,axis1=2,axis2=3)-1)/2,-1.,1.)))
    frame, joint = np.unravel_index(angles.argmax(),angles.shape)
    return dict(p95_degrees=float(np.percentile(angles,95)),max_degrees=float(angles[frame,joint]),
                max_joint=names[joint],max_between_source_frames=[int(frame+1),int(frame+2)],
                above20=int((angles>20).sum()),above45=int((angles>45).sum()))


def main():
    began = time.monotonic()
    record = json.loads(pinned(ORIGINAL/'result.json',3586,FIT_SHA))
    need(record['scope']=='GEOMETRIC_POSITION_ONLY_LOCAL_CANDIDATE' and record['frames']==940 and
         record['cached201_qualified'] is False, 'expected the original local positional fit')
    asset = record['private_arrays']
    inputs = pinned(ORIGINAL/'positional-fit-not-training-private.npz',asset['bytes'],asset['SHA256'])
    with np.load(BytesIO(inputs),allow_pickle=False) as sample:
        local,position = sample['local_rotations'],sample['positions']
        frames,original_names = sample['frame_numbers'],sample['joint_names'].tolist()
    names,rest,parents = source_rig()
    need(names==original_names and local.shape==(940,22,3,3) and position.shape==(940,22,3) and
         np.isfinite(local).all() and np.isfinite(position).all() and np.array_equal(frames,np.arange(1,941)),
         'original animation scope differs')
    stable,single,leaf,turns = retwist(local,rest,parents)
    replay = replay_fk(stable,rest,parents)
    error = float(np.max(np.abs(replay-position)))
    so3 = float(np.max(np.abs(stable.swapaxes(2,3)@stable-np.eye(3))))
    determinant = float(np.max(np.abs(np.linalg.det(stable)-1)))
    need(max(error,so3,determinant)<1e-9, 'continuity operation changed the original positional fit')
    OUTPUT.mkdir()
    reconstructed = OUTPUT/'continuous-twist-not-training-private.npz'
    np.savez_compressed(reconstructed,frame_numbers=frames,joint_names=names,positions=replay,local_rotations=stable)
    raw = reconstructed.read_bytes()
    out = dict(scope='LOCAL_WOODEN_BODY_ROTATION_REPRESENTATIVE',source_position_result_SHA256=FIT_SHA,
               frames=len(frames),frame_rate=60,body_joint_count=len(names),
               single_child_twist_representatives=[names[j] for j in single],
               leaf_relative_rotation_identity=[names[j] for j in leaf],exact_half_turn_fallbacks=turns,
               before=rotation_summary(local,names),after=rotation_summary(stable,names),
               max_position_change_in_rig_units=error,orthogonality_max_abs=so3,determinant_max_abs=determinant,
               artifact=dict(bytes=len(raw),SHA256=sha256(raw).hexdigest()),wall_seconds=time.monotonic()-began,
               source_twist_groundtruth_observed=False,resampled_30Hz=False,model_feature135_qualified=False,
               cached201_qualified=False,actual_optimizer_updates=0,actual_GPU_steps=0,READY_added=0)
    with (OUTPUT/'result.json').open('x',encoding='utf8',newline='\n') as handle:
        json.dump(out,handle,allow_nan=False,sort_keys=True,indent=2)
        handle.write('\n')
    print(json.dumps({k:out[k] for k in ('before','after','max_position_change_in_rig_units','READY_added')},allow_nan=False))


if __name__ == '__main__':
    main()
