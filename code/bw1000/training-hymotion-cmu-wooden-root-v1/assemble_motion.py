"""Calibrate an observed CMU root trajectory against the public WoodenMesh rest rig."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOCAL = REPO/'local-only'
OUTPUT = LOCAL/'hymotion-cmu-wooden-root-v1-original'
ROTATIONS = LOCAL/'hymotion-cmu-wooden-rotation30-v1-original'
CMU = LOCAL/'training-hymotion-cmu-asf-fk-candidate-v1-original'
OFFICIAL = LOCAL/'hymotion-wooden-official-lfs-4e426f5-original'
MESH = LOCAL/'hymotion-wooden-official-foot-mesh-4e426f5-original'
BINDING = LOCAL/'hymotion-cmu-to-wooden-rest-binding-v1-original'
CALIBRATION_FRAMES = 128
CONTACT_HEIGHT_M = .03
CONTACT_SPEED_M_S = .05
WEIGHTS = np.array([1,8,28,56,70,56,28,8,1])/256


def need(condition,message):
    if not condition:
        raise ValueError(message)


def pin(raw):
    return dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())


def fixed(path,count,digest):
    with path.open('rb') as stream:
        b = stream.read(count+1)
    need(len(b)==count and sha256(b).hexdigest()==digest, 'pinned input differs: '+path.name)
    return b


def prior_math():
    root = HERE.parent/'training-hymotion-cmu-wooden-position-fit-v1'
    content = fixed(root/'SOURCE_FILES.json',739,'b74668bffe8303bfc655a1ed4ec627b3ea5280effffca2ca89026f6efc53239d')
    for n,p in json.loads(content)['files'].items():
        fixed(root/n,p['bytes'],p['SHA256'])
    sys.path.insert(0,str(root))
    try:
        from fit_poses import source_rig
        from resample_rotations import decode6
    finally:
        sys.path.pop(0)
    return source_rig,decode6


def rotations():
    data = json.loads(fixed(ROTATIONS/'result.json',1850,'2fb4b419dd803d17f36e65333aa54273d795412e34e2a93eabefef7172d22957'))
    item = data['private_result']
    raw = fixed(ROTATIONS/'rot6d-30Hz-no-root-no-201-private.npz',item['bytes'],item['SHA256'])
    with np.load(BytesIO(raw),allow_pickle=False) as store:
        frames, pose = store['source_frame_numbers'],store['rot6d']
        names, points = store['joint_names'].tolist(),store['relative_joint_positions']
    need(np.array_equal(frames,np.arange(1,940,2)) and pose.shape==(470,22,6) and pose.dtype==np.float32 and
         int(data['output_fps'])==30 and np.isfinite(pose).all(), 'prior 470 rotation frames differ')
    return frames,pose,names,points


def original_motion(centers):
    block = fixed(CMU/'world-joints-native-meters-private.npz',2679326,'4ca26a839adf332c5525c78a3443f841f97700d608c0052e13763ac11e3d55fc')
    hierarchy = json.loads(fixed(CMU/'hierarchy-and-conventions-private.json',14452,
                                'e7374b7a18d16255506febbc6575272beca20803c8abfb8b00f256fd71032aa0'))
    names = hierarchy['joint_names']
    with np.load(BytesIO(block),allow_pickle=False) as store:
        points,source_frames = store['joint_endpoints_m'],store['frame_numbers']
    need(np.isfinite(points).all() and points.shape==(940,31,3) and names[0]=='root' and
         np.array_equal(source_frames[centers],centers+1), 'original CMU motion differs')
    columns = [names.index(n) for n in ('root','ltoes','rtoes')]
    windows = np.pad(np.arange(len(points)),(4,4),mode='reflect')[centers[:,None]+np.arange(9)]
    observations = np.einsum('w,fwjd->fjd',WEIGHTS,points[:,columns][windows])
    toe_height = observations[:,1:,1]
    floor = np.quantile(toe_height[:CALIBRATION_FRAMES],.05)
    velocity = np.linalg.norm(np.gradient(observations[:,1:],1/30.,axis=0),axis=-1)
    supports = (np.abs(toe_height-floor)<=CONTACT_HEIGHT_M)&(velocity<=CONTACT_SPEED_M_S)
    return observations[:,0],supports,float(floor),points[:,columns[0]],hierarchy


def full_skeleton():
    j = fixed(OFFICIAL/'j_template.bin',624,'4f488cc9d0b650a816f5a8eda49eda7bc796f490e9130a6e0dec5be137d7b929')
    p = fixed(OFFICIAL/'kintree.bin',208,'98a20fa3b53193790b63d9ac3a9c917f2f70fbe6e053dca495c75317ff4b756a')
    return np.frombuffer(j,'<f4').reshape(52,3).astype(np.float64),np.frombuffer(p,'<i4')


def transform(poses,rest,parents):
    world = np.broadcast_to(np.eye(3),(len(poses),len(rest),3,3)).copy()
    world[:,:poses.shape[1]] = poses
    points = np.empty((len(poses),len(rest),3))
    points[:,0] = rest[0]
    need(parents[0]==-1 and all(0<=p<j for j,p in enumerate(parents[1:],1)), 'invalid rig parent topology')
    for j in range(1,len(rest)):
        p = parents[j]
        points[:,j] = points[:,p]+world[:,p]@(rest[j]-rest[p])
        world[:,j] = world[:,p]@world[:,j]
    return world,points


def skin(world,points,rest,vertices,weights,bones):
    offsets = points-(world@rest[None,:,:,None])[...,0]
    joint_to_vertex = (world[:,bones]@vertices[None,:,None,:,None])[...,0]+offsets[:,bones]
    return (joint_to_vertex*weights[None,:,:,None]).sum(axis=2)


def feet_on_rig(rest,names):
    source = json.loads(fixed(MESH/'result.json',1606,'d3c64b2463b0723244979e9b87b36e5c2ea92cbbcedc790ca92bbdc842bb8a8e'))
    files = {}
    for name,field in source['fixed_payloads'].items():
        need(field['two_successful_GETs'] is True and field['expected_SHA256']==field['SHA256'], 'missing pinned mesh')
        files[name] = fixed(MESH/name,field['bytes'],field['SHA256'])
    vertices = np.frombuffer(files['v_template.bin'],'<f4').reshape(-1,3).astype(np.float64)
    weights = np.frombuffer(files['skinWeights.bin'],'<f4').reshape(len(vertices),4).astype(np.float64)
    indices = np.frombuffer(files['skinIndice.bin'],'<u2').reshape(len(vertices),4)
    need(np.isfinite(vertices).all() and np.isfinite(weights).all() and (weights>=0).all() and indices.max()<len(rest) and
         np.max(np.abs(weights.sum(axis=1)-1))<1e-5, 'mesh vertex weights differ')
    regions = []
    for side in ('L','R'):
        foot_bones = [names.index(side+'_'+p) for p in ('Ankle','Foot')]
        mask = (weights*np.isin(indices,foot_bones)).sum(axis=1)>=.8
        low = np.quantile(vertices[mask,1],.15)
        selected = mask&(vertices[:,1]<=low)
        need(int(selected.sum())>10, 'insufficient original sole vertices')
        regions.append((vertices[selected],weights[selected],indices[selected]))
    return len(vertices),regions


def source_root_in_rig(hierarchy,root,rest,names,scale):
    vectors = {row['name'][0]:np.array(row['direction'],dtype=float)*float(row['length'][0])
               for row in hierarchy['bones']}
    lateral_source = vectors['lhipjoint']-vectors['rhipjoint']
    lateral_target = rest[names.index('L_Hip')]-rest[names.index('R_Hip')]
    def heading(lateral):
        right = lateral.copy()
        right[1] = 0
        right /= np.linalg.norm(right)
        y = np.array([0.,1.,0.])
        return np.stack((right,y,np.cross(right,y)),axis=1)
    rotation = heading(lateral_target)@heading(lateral_source).T
    need(np.allclose(rotation@rotation.T,np.eye(3),rtol=0,atol=1e-12) and
         np.array_equal(rotation[1],[0,1,0]) and abs(np.linalg.det(rotation)-1)<1e-12, 'expected yaw-only basis')
    converted = (root-root[0])@rotation.T*scale
    return converted,rotation


def pack_201(root,rot6d,model_joints):
    frames = len(root)
    features = np.concatenate((root,rot6d.reshape(frames,132)),axis=1).astype(np.float32)
    # Native WoodenMesh applies global translation to vertices only, not keypoints3d.
    return features,np.concatenate((features,model_joints[:,:22].reshape(frames,66)),axis=1).astype(np.float32)


def write_sources(directory, features, points):
    dataset = directory/'candidate-native-source-only'
    for name in ('joints_6d','new_joints','texts'):
        (dataset/name).mkdir(parents=True,exist_ok=True)
    key = 'cmu79-81-root-preview'
    (dataset/'train.txt').write_text(key+'\n',encoding='utf8')
    (dataset/'texts'/f'{key}.txt').write_text('brushing teeth\n',encoding='utf8')
    np.save(dataset/'joints_6d'/f'{key}.npy',features)
    np.save(dataset/'new_joints'/f'{key}.npy',points.astype(np.float32))


def main():
    begin = time.monotonic()
    source_rig,decode6 = prior_math()
    frames,rot6d,names,relative = rotations()
    target,rest_body,parents_body = source_rig()
    rest,parents = full_skeleton()
    need(names==target and np.array_equal(rest[:22],rest_body) and np.array_equal(parents[:22],parents_body),
         'the new skin assets must use the original 22-node body template')
    root,contacts,source_floor,raw_source,hierarchy = original_motion(frames-1)
    binding = json.loads((BINDING/'result.json').read_bytes())
    scale = binding['source_to_target_paired_leg_length_scale']
    need(abs(scale-0.9132775972368371)<1e-14,'original geometric scale differs')
    translation, heading = source_root_in_rig(hierarchy,root,rest,names,scale)
    world,joints = transform(decode6(rot6d),rest,parents)
    relative_error = float(np.max(np.abs(joints[:,:22]-rest[0]-relative)))
    need(relative_error<1e-10,'body forward kinematics differs')
    total_vertices,feet = feet_on_rig(rest,names)
    foot_heights = []
    for region in feet:
        foot_heights.append(skin(world,joints,rest,*region)[...,1].min(axis=1))
    floor_side = np.stack(foot_heights,axis=1)+translation[:,1,None]
    selected = contacts[:CALIBRATION_FRAMES]
    need(int(selected.sum())>=16 and int(contacts[CALIBRATION_FRAMES:].sum())>=16,'insufficient observed stationary contact')
    offset = -float(np.median(floor_side[:CALIBRATION_FRAMES][selected]))
    translation[:,1] += offset
    final_floor = floor_side+offset
    test = final_floor[CALIBRATION_FRAMES:][contacts[CALIBRATION_FRAMES:]]
    features,geometry = pack_201(translation,rot6d,joints)
    mean = np.load(BytesIO(fixed(OFFICIAL/'Mean.npy',1736,'a4bbe5f68881fe793d509f262ad3dc698a91f79158a9bde9d6f24d46ef9652bd')),allow_pickle=False).astype(np.float32)
    std = np.load(BytesIO(fixed(OFFICIAL/'Std.npy',1736,'6cc86d1ca4b47fbd0e84b677c4397e51b6615abcb34e38fdd9daabbe6a5685cc')),allow_pickle=False).astype(np.float32)
    normalized = (geometry-mean)/np.where(std<1e-3,np.float32(1),std)
    need(features.shape==(470,135) and geometry.shape==(470,201) and np.isfinite(normalized).all(), 'native input composition differs')
    # Compare relative motion rather than the unrelated calibration origin.
    direct = (raw_source[frames-1]-raw_source[0])@heading.T*scale
    relative_root_error = float(np.max(np.linalg.norm((translation-translation[0])-direct,axis=1)))
    OUTPUT.mkdir()
    dest = OUTPUT/'motion135-and-CPU-geometry201-preview-private.npz'
    np.savez_compressed(dest,source_frame_numbers=frames,motion135=features,cpu_geometry201=geometry,
                        cpu_normalized201=normalized)
    write_sources(OUTPUT,features,joints[:,:22])
    result = dict(scope='REAL_CMU_PUBLIC_RIG_CPU_GEOMETRY_PREVIEW',frames=len(frames),sample_fps=30,
                  physical_source_floor_estimate_m=source_floor,original_mesh_vertex_count=total_vertices,
                  used_left_right_sole_vertices=[len(f[0]) for f in feet],calibration_frames=CALIBRATION_FRAMES,
                  contact_speed_cutoff_m_s=CONTACT_SPEED_M_S,contact_height_delta_cutoff_m=CONTACT_HEIGHT_M,
                  observed_calibration_contact_side_frames=int(selected.sum()),observed_later_contact_side_frames=len(test),
                  later_same_recording_contact_RMS_rig_units=float(np.sqrt(np.mean(test**2))),
                  later_contact_abs_p95_rig_units=float(np.percentile(np.abs(test),95)),
                  later_contact_min_rig_units=float(test.min()),later_contact_max_rig_units=float(test.max()),
                  root_filter_max_relative_motion_change_rig_units=relative_root_error,
                  root_y_support_offset_rig_units=offset,relative_fk_vs_previous_max_abs=relative_error,
                  source_135_shape=list(features.shape),shadow_201_shape=list(geometry.shape),
                  normalized_geometry_abs_p95=float(np.percentile(np.abs(normalized),95)),
                  normalized_geometry_max_abs=float(np.max(np.abs(normalized))),
                  global_translation_applied_to_66_keypoints=False,
                  geometry_file=pin(dest.read_bytes()),wall_seconds=time.monotonic()-begin,
                  native_wooden_model_executions=0,native_text_encoder_executions=0,
                  deployed_stats_match_known=False,domain_training_READY=False,formal_T08_qualified=False,
                  actual_GPU_steps=0,actual_optimizer_updates=0)
    with (OUTPUT/'result.json').open('x',encoding='utf8',newline='\n') as f:
        f.write(json.dumps(result,allow_nan=False,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,allow_nan=False))


if __name__ == '__main__':
    main()
