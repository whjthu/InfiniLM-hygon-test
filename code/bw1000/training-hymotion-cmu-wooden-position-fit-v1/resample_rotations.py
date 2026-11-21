"""Decimate the fitted rotations on SO(3) and verify the native 6D layout."""

import ast
from hashlib import sha256
from io import BytesIO
import json
import time

import numpy as np

from fit_poses import HERE, PRIVATE, need, pinned, replay_fk, skew, source_rig
from transport_twist import rotation_summary


INPUT = PRIVATE/'hymotion-cmu-wooden-position-twist-v1-original'
OUTPUT = PRIVATE/'hymotion-cmu-wooden-rotation30-v1-original'
SOURCE = HERE.parent/'training-hymotion-target-meta-CPU-v2/upstream/hymotion/utils/geometry.py'
WEIGHTS = np.array([1,8,28,56,70,56,28,8,1])/256


def log_local(delta):
    off = delta-delta.swapaxes(-1,-2)
    sine_v = .5*np.stack((off[...,2,1],off[...,0,2],off[...,1,0]),axis=-1)
    sine = np.linalg.norm(sine_v,axis=-1)
    cosine = np.clip((np.trace(delta,axis1=-1,axis2=-2)-1)/2,-1.,1.)
    theta = np.arctan2(sine,cosine)
    need(np.max(theta)<np.pi-1e-6,'rotation window crosses the principal logarithm boundary')
    return sine_v/np.sinc(theta/np.pi)[...,None], float(np.max(theta))


def exp_local(tangent):
    theta = np.linalg.norm(tangent,axis=-1)
    cross = skew(tangent.reshape(-1,3)).reshape((*theta.shape,3,3))
    return np.eye(3)+np.sinc(theta/np.pi)[...,None,None]*cross+(.5*np.sinc(theta/(2*np.pi))**2)[...,None,None]*(cross@cross)


def decimate(rotations):
    centers = np.arange(0,len(rotations),2)
    # Reflect the time indices at both endpoints. The within-window mean uses
    # tangent coordinates around the rotation sampled at the output timestamp.
    windows = np.pad(np.arange(len(rotations)),(4,4),mode='reflect')[centers[:,None]+np.arange(9)]
    base = rotations[centers]
    relative = base.swapaxes(-1,-2)[:,None]@rotations[windows]
    logarithms, max_angle = log_local(relative)
    tangent = np.einsum('w,fwjd->fjd',WEIGHTS,logarithms)
    return centers, base@exp_local(tangent), max_angle


def encode6(rotations):
    return rotations[...,:2].reshape((*rotations.shape[:-2],6)).astype(np.float32)


def decode6(values):
    six = values.astype(np.float64).reshape((*values.shape[:-1],3,2))
    first, second = six[...,0], six[...,1]
    e1 = first/np.linalg.norm(first,axis=-1)[...,None]
    perp = second-(second*e1).sum(axis=-1)[...,None]*e1
    e2 = perp/np.linalg.norm(perp,axis=-1)[...,None]
    return np.stack((e1,e2,np.cross(e1,e2)),axis=-1)


def source_convention():
    raw = SOURCE.read_bytes()
    digest = sha256(raw).hexdigest()
    need(digest=='15d0c3c093e6fbe7464ce54697c6d909050217d2af8af0813a21cd64a7b3c540', 'original upstream geometry code differs')
    functions = {n.name:ast.unparse(n) for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}
    encoder,decoder = (functions[name] for name in ('rotation_matrix_to_rot6d','rot6d_to_rotation_matrix'))
    need('rotation_matrix[..., 0:1]' in encoder and 'rotation_matrix[..., 1:2]' in encoder and
         'torch.cat([v1, v2], dim=-1).reshape' in encoder and '3, 2)' in decoder and
         'torch.stack((b1, b2, b3), dim=-1)' in decoder,'original upstream 6D layout differs')
    return digest


def main():
    begin = time.monotonic()
    original_report = pinned(INPUT/'result.json',1619,'3a8409718a516d12106bdedd7fa2a60c80f57c6220e400ae7e8487bddf0b8cbb')
    result = json.loads(original_report)
    need(result['scope']=='LOCAL_WOODEN_BODY_ROTATION_REPRESENTATIVE' and result['frames']==940 and
         result['frame_rate']==60 and result['cached201_qualified'] is False and result['after']['max_degrees']<8,
         'expected the verified original continuous 60 Hz candidate')
    item = result['artifact']
    blob = pinned(INPUT/'continuous-twist-not-training-private.npz',item['bytes'],item['SHA256'])
    with np.load(BytesIO(blob),allow_pickle=False) as original:
        motions,positions,frames = original['local_rotations'], original['positions'], original['frame_numbers']
        source_names = original['joint_names'].tolist()
    names,rest,parents = source_rig()
    need(source_names==names and motions.shape==(940,22,3,3) and positions.shape==(940,22,3) and
         np.array_equal(frames,np.arange(1,941)) and np.isfinite(motions).all(), 'original motion does not match target body')
    source_sha = source_convention()
    centers,smoothed,angle = decimate(motions)
    six = encode6(smoothed)
    decoded = decode6(six)
    fk = replay_fk(decoded,rest,parents)
    error = float(np.max(np.abs(smoothed-decoded)))
    difference = np.linalg.norm(fk-positions[centers],axis=-1)
    need(len(centers)==470 and six.shape==(470,22,6) and np.isfinite(six).all() and error<1e-6,
         '30Hz 6D rotation round trip differs')
    OUTPUT.mkdir()
    trajectory = OUTPUT/'rot6d-30Hz-no-root-no-201-private.npz'
    np.savez_compressed(trajectory,source_frame_numbers=frames[centers],joint_names=names,rot6d=six,
                        relative_joint_positions=fk,frame_rate_hz=np.array(30,dtype=np.int32))
    array_raw = trajectory.read_bytes()
    unfiltered, filtered = (rotation_summary(value,names) for value in (motions[centers],decoded))
    for summary in (unfiltered,filtered):
        summary['max_between_source_frames'] = [int(frames[centers[i-1]]) for i in summary['max_between_source_frames']]
    out = dict(scope='30HZ_PUBLIC_RIG_ROTATION_COMPONENT_CANDIDATE',source_frames=940,output_frames=470,
               source_fps=60,output_fps=30,first_source_frame=int(frames[centers[0]]),last_source_frame=int(frames[centers[-1]]),
               local_tangent_binomial_weights=WEIGHTS.tolist(),boundary_rule='reflect',
               max_rotation_window_radius_degrees=float(np.degrees(angle)),official_geometry_source_SHA256=source_sha,
               official_target_runtime_executed=False,rot6d_roundtrip_matrix_max_abs=error,
               unfiltered_decimated_rotation=unfiltered,filtered_decimated_rotation=filtered,position_units='WoodenMesh rig units',
               filter_displacement_rms=float(np.sqrt(np.mean(difference**2))),filter_displacement_p95=float(np.percentile(difference,95)),
               filter_displacement_max=float(difference.max()),private_result=dict(bytes=len(array_raw),SHA256=sha256(array_raw).hexdigest()),
               source_result_SHA256=sha256(original_report).hexdigest(),source_twist_groundtruth_observed=False,
               root_translation_qualified=False,model_feature135_qualified=False,cached201_qualified=False,
               READY_added=0,actual_optimizer_updates=0,actual_GPU_steps=0,wall_seconds=time.monotonic()-begin)
    with (OUTPUT/'result.json').open('x',encoding='utf8',newline='\n') as f:
        f.write(json.dumps(out,allow_nan=False,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('output_frames','last_source_frame','rot6d_roundtrip_matrix_max_abs',
        'filter_displacement_p95','filter_displacement_max','filtered_decimated_rotation','READY_added')},allow_nan=False))


if __name__ == '__main__':
    main()
