"""Evaluate one constant, position-invariant neck twist using only calibration frames."""

from io import BytesIO
import json
import time

import numpy as np

from assemble_motion import (CALIBRATION_FRAMES, LOCAL, OFFICIAL, fixed, full_skeleton, need, pack_201,
                             pin, prior_math, rotations, transform)


INPUT = LOCAL/'hymotion-cmu-wooden-root-v1-original'
OUTPUT = LOCAL/'hymotion-cmu-wooden-neck-prior-v1-original'
GRID = np.arange(-180,180,2)


def constant_twist(axis,degrees):
    theta = np.radians(degrees)
    x,y,z = axis
    k = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(theta)[...,None,None]*k+(1-np.cos(theta))[...,None,None]*(k@k)


def objective(local,axis,mean,std):
    rotated = local[:CALIBRATION_FRAMES,None]@constant_twist(axis,GRID)
    six = rotated[...,:2].reshape(len(rotated),len(GRID),6)
    return np.mean(((six-mean)/std)**2,axis=(0,2))


def score(geometry,mean,denominator,neck):
    row = geometry[CALIBRATION_FRAMES:]
    deviation = np.abs((row-mean)/denominator)
    block = deviation[:,3+neck*6:3+(neck+1)*6]
    return dict(neck_abs_max=float(block.max()),neck_RMS=float(np.sqrt(np.mean(block**2))),
                all_rot6d_abs_max=float(deviation[:,3:135].max()),
                all_rot6d_above_10=int((deviation[:,3:135]>10).sum()))


def main():
    began = time.monotonic()
    proof = json.loads((INPUT/'result.json').read_bytes())
    need(proof['scope']=='REAL_CMU_PUBLIC_RIG_CPU_GEOMETRY_PREVIEW' and proof['calibration_frames']==CALIBRATION_FRAMES
         and proof['domain_training_READY'] is False and proof['source_135_shape']==[470,135], 'original root preview scope differs')
    source = proof['geometry_file']
    old = fixed(INPUT/'motion135-and-CPU-geometry201-preview-private.npz',source['bytes'],source['SHA256'])
    with np.load(BytesIO(old),allow_pickle=False) as d:
        original,root,source_frames = d['cpu_geometry201'],d['motion135'][:,:3],d['source_frame_numbers']
    _,decode6 = prior_math()
    frames,rot6d,names,_ = rotations()
    need(np.array_equal(frames,source_frames),'root and rotation frames differ')
    rest,parents = full_skeleton()
    neck,head = (names.index(n) for n in ('Neck','Head'))
    need(np.flatnonzero(parents==neck).tolist()==[head] and not (parents==head).any(),
         'neck must have a single terminal successor for this transform')
    axis = rest[head]-rest[neck]
    axis /= np.linalg.norm(axis)
    mean = np.load(BytesIO(fixed(OFFICIAL/'Mean.npy',1736,'a4bbe5f68881fe793d509f262ad3dc698a91f79158a9bde9d6f24d46ef9652bd')),allow_pickle=False).astype(np.float32)
    std = np.load(BytesIO(fixed(OFFICIAL/'Std.npy',1736,'6cc86d1ca4b47fbd0e84b677c4397e51b6615abcb34e38fdd9daabbe6a5685cc')),allow_pickle=False).astype(np.float32)
    denominator = np.where(std<1e-3,np.float32(1),std)
    section = slice(3+neck*6,3+(neck+1)*6)
    base = decode6(rot6d)
    objectives = objective(base[:,neck],axis,mean[section],denominator[section])
    index = int(np.argmin(objectives))
    proposal = base[:,neck]@constant_twist(axis,GRID[index])
    candidate = rot6d.copy()
    candidate[:,neck] = proposal[...,:2].reshape(-1,6).astype(np.float32)
    _,base_points = transform(base,rest,parents)
    _,after_points = transform(decode6(candidate),rest,parents)
    position_change = float(np.max(np.abs(after_points[:,:22]-base_points[:,:22])))
    need(position_change<1e-6,'single-child twist changed stored joint geometry')
    _,preview = pack_201(root,candidate,after_points)
    before = score(original,mean,denominator,neck)
    after = score(preview,mean,denominator,neck)
    relative = base[1:,neck]@base[:-1,neck].swapaxes(1,2)
    rotated_delta = proposal[1:]@proposal[:-1].swapaxes(1,2)
    speed_error = float(np.max(np.abs(relative-rotated_delta)))
    need(speed_error<1e-12,'constant local twist changed neck velocity')
    OUTPUT.mkdir()
    r = dict(scope='NECK_POSITION_INVARIANT_CONSTANT_TWIST_DIAGNOSTIC',train_clip_frames=len(frames),
             first_calibration_frames=CALIBRATION_FRAMES,subsequent_same_recording_frames=len(frames)-CALIBRATION_FRAMES,
             candidate_angles=len(GRID),selected_angle_degrees=int(GRID[index]),
             calib_neck_standardized_MSE_at_zero=float(objectives[90]),calib_neck_standardized_MSE_at_selected=float(objectives[index]),
             later_original=before,later_candidate=after,max_body_position_change_rig_units=position_change,
             max_neck_relative_step_matrix_change=speed_error,source_reference=pin(old),
             acted_on_unobservable_joint_axis=True,actor_twist_groundtruth_observed=False,
             production_training_cache_qualified=False,READY_added=0,actual_GPU_steps=0,actual_optimizer_updates=0,
             wall_seconds=time.monotonic()-began)
    with (OUTPUT/'result.json').open('x',encoding='utf8',newline='\n') as f:
        f.write(json.dumps(r,allow_nan=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(r,allow_nan=False))


if __name__ == '__main__':
    main()
