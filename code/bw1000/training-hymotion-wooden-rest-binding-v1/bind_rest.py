"""Read the real Tencent rest skeleton and measure CMU anatomical anchor candidates."""

import hashlib
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOCAL = REPO/'local-only'
FK = LOCAL/'training-hymotion-cmu-asf-fk-candidate-v1-original'
RAW = LOCAL/'hymotion-wooden-official-assets-4e426f5-original'
LFS = LOCAL/'hymotion-wooden-official-lfs-4e426f5-original'
OUTPUT = LOCAL/'hymotion-cmu-to-wooden-rest-binding-v1-original'
REV = '4e426f5a1021cbcf7f375458c37b840ee7225229'
CONFIG_BYTES = 36790
CONFIG_SHA = '6b4d8a42486c7c4eb38a9addaa58bcf57377b735acf2fda2a704c85c65a1ef58'
ANCHORS = {
    'Pelvis':'root', 'L_Hip':'lhipjoint', 'R_Hip':'rhipjoint', 'Spine1':'lowerback',
    'L_Knee':'lfemur', 'R_Knee':'rfemur', 'Spine2':'upperback', 'L_Ankle':'ltibia',
    'R_Ankle':'rtibia', 'Spine3':'thorax', 'L_Foot':'lfoot', 'R_Foot':'rfoot',
    'Neck':'lowerneck', 'Head':'upperneck', 'L_Shoulder':'lclavicle', 'R_Shoulder':'rclavicle',
    'L_Elbow':'lhumerus', 'R_Elbow':'rhumerus', 'L_Wrist':'lradius', 'R_Wrist':'rradius',
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path, size, expected):
    with path.open('rb') as stream:
        data = stream.read(size+1)
    need(len(data) == size and sha(data) == expected, 'pinned input differs: '+path.name)
    return data


def wooden_rig():
    names = json.loads(read(RAW/'joint_names.json',698,'f29139c07a3e8c160cb3512428aac6e272697ade096a7e682024c47424e064d1'))
    data = read(LFS/'j_template.bin',624,'4f488cc9d0b650a816f5a8eda49eda7bc796f490e9130a6e0dec5be137d7b929')
    joints = np.frombuffer(data,dtype='<f4').reshape(52,3).astype(np.float64)
    edges = read(LFS/'kintree.bin',208,'98a20fa3b53193790b63d9ac3a9c917f2f70fbe6e053dca495c75317ff4b756a')
    parents = np.frombuffer(edges,dtype='<i4')
    need(len(names) == len(set(names)) == 52 and np.isfinite(joints).all() and parents[0] == -1 and
         all(0 <= parent < node for node,parent in enumerate(parents[1:],1)), 'invalid target rest skeleton')
    need(set(names[:22]) == set(ANCHORS)|{'L_Collar','R_Collar'} and np.max(parents[1:22]) < 22,
         'first 22 joints do not describe the native body')
    return names[:22],joints[:22],parents[:22]


def stats():
    mean = np.load(BytesIO(read(LFS/'Mean.npy',1736,'a4bbe5f68881fe793d509f262ad3dc698a91f79158a9bde9d6f24d46ef9652bd')),allow_pickle=False)
    std = np.load(BytesIO(read(LFS/'Std.npy',1736,'6cc86d1ca4b47fbd0e84b677c4397e51b6615abcb34e38fdd9daabbe6a5685cc')),allow_pickle=False)
    need(mean.shape == std.shape == (201,) and np.isfinite(mean).all() and np.isfinite(std).all() and
         (std >= 0).all(), 'official 201-value statistics differ')
    return dict(dimension=201,original_mean_dtype=str(mean.dtype),original_std_dtype=str(std.dtype),
                producer_FP32_low_std_replaced=int((std.astype(np.float32)<1e-3).sum()))


def cmu_rig():
    hierarchy = json.loads(read(FK/'hierarchy-and-conventions-private.json',14452,
                                 'e7374b7a18d16255506febbc6575272beca20803c8abfb8b00f256fd71032aa0'))
    names = hierarchy['joint_names']
    parents = hierarchy['parent_indices']
    unit = hierarchy['physical_raw_unit_to_meters']
    need(names[0] == 'root' and len(names) == 31 and unit == 0.0254/0.45, 'unexpected CMU calibration')
    offsets = {bone['name'][0]: np.array(bone['direction'],dtype=np.float64)*float(bone['length'][0])*unit
               for bone in hierarchy['bones']}
    rest = np.zeros((31,3))
    for idx,name in enumerate(names[1:],1):
        need(0 <= parents[idx] < idx, 'unexpected original CMU edge order')
        rest[idx] = rest[parents[idx]]+offsets[name]
    data = read(FK/'world-joints-native-meters-private.npz',2679326,'4ca26a839adf332c5525c78a3443f841f97700d608c0052e13763ac11e3d55fc')
    with np.load(BytesIO(data),allow_pickle=False) as store:
        frames, points = store['frame_numbers'], store['joint_endpoints_m']
    need(np.array_equal(frames,np.arange(1,941)) and points.shape == (940,31,3) and np.isfinite(points).all(),
         'the complete verified CMU capture is required')
    return names,rest,points


def basis(lateral, upward):
    length = np.linalg.norm(lateral)
    need(length > 0, 'degenerate hip line')
    x = lateral/length
    y = upward-np.dot(upward,x)*x
    vertical = np.linalg.norm(y)
    need(vertical > 0, 'degenerate pelvis-to-spine vector')
    y /= vertical
    return np.stack((x,y,np.cross(x,y)),axis=1)


def fit_anchors(source, source_rest, source_points, target, target_rest, target_parents):
    src = {name:source.index(value) for name,value in ANCHORS.items()}
    dst = {name:target.index(name) for name in ANCHORS}
    geometry = []
    for rest,index in ((source_rest,src),(target_rest,dst)):
        pelvis,lhip,rhip,spine = (rest[index[k]] for k in ('Pelvis','L_Hip','R_Hip','Spine3'))
        legs = sum(np.linalg.norm(rest[index[f'{side}_{a}']]-rest[index[f'{side}_{b}']])
                   for side in ('L','R') for a,b in (('Hip','Knee'),('Knee','Ankle')))
        need(legs > 0, 'degenerate original leg lengths')
        geometry.append((basis(lhip-rhip,spine-pelvis),legs))
    rotation = geometry[1][0]@geometry[0][0].T
    need(np.allclose(rotation@rotation.T,np.eye(3),rtol=0,atol=1e-12) and abs(np.linalg.det(rotation)-1)<1e-12,
         'rest basis would reflect or distort the capture')
    scale = geometry[1][1]/geometry[0][1]
    names = [name for name in target if name in src]
    indexes = [src[name] for name in names]
    rest = (source_rest[indexes]-source_rest[src['Pelvis']])@rotation.T*scale
    target_relative = target_rest[[dst[name] for name in names]]-target_rest[dst['Pelvis']]
    points = (source_points[:,indexes]-source_points[:,[src['Pelvis']]])@rotation.T*scale
    angle_differences = {}
    for name in names:
        parent = int(target_parents[dst[name]])
        if parent < 0 or target[parent] not in src:
            continue
        prior = target[parent]
        source_vector = rest[names.index(name)]-rest[names.index(prior)]
        target_vector = target_rest[dst[name]]-target_rest[parent]
        denominator = np.linalg.norm(source_vector)*np.linalg.norm(target_vector)
        need(denominator > 0, 'degenerate source or target edge')
        angle_differences[name] = float(np.degrees(np.arccos(np.clip(source_vector@target_vector/denominator,-1.,1.))))
    result = dict(matched_anchor_candidates=len(names),unmapped_target=[n for n in target if n not in src],
                  checked_target_edges=len(angle_differences),rest_edge_angles_degrees=angle_differences,
                  rest_anchor_median_distance_in_target_units=float(np.median(np.linalg.norm(rest-target_relative,axis=1))),
                  source_to_target_paired_leg_length_scale=float(scale),proper_rotation=True)
    return result, points, names


def main():
    started = time.monotonic()
    target,target_rest,parents = wooden_rig()
    source,source_rest,source_points = cmu_rig()
    output, points, names = fit_anchors(source,source_rest,source_points,target,target_rest,parents)
    need(np.isfinite(points).all() and points.shape == (940,20,3), 'invalid candidate anchor trajectory')
    configuration = json.loads(read(REPO/'code/bw1000/training-hymotion-target-meta-CPU-v2/META_SPEC.json',
                                     CONFIG_BYTES,CONFIG_SHA))
    fps = configuration['model_config']['train_pipeline_args']['output_mesh_fps']
    need(fps == 30, 'target source FPS differs')
    normalization = stats()
    OUTPUT.mkdir()
    trajectory = OUTPUT/'anchors-in-wooden-rest-basis-private.npz'
    np.savez_compressed(trajectory, frame_numbers=np.arange(1,941), candidate_joint_names=names, points=points)
    artifact = trajectory.read_bytes()
    count = points.shape[0]
    source_fps = 60
    result = dict(output, official_revision=REV, official_statistics=normalization, actual_CMU_frames=count,
        actual_source_fps=source_fps,target_config_fps=fps,time_grid_intersections_at_origin=len(range(0,count,source_fps//fps)),
        actual_resampling=False,public_template_matches_deployed_target=None,evaluated_anchors_only=True,HY135_or201_mapping=False,
        training_GO=False,READY_added=0,actual_GPU_steps=0,
        private_trajectory=dict(bytes=len(artifact),SHA256=sha(artifact)),wall_seconds=time.monotonic()-started)
    encoded = json.dumps(result,ensure_ascii=False,allow_nan=False,sort_keys=True,indent=2)+'\n'
    with (OUTPUT/'result.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(encoded)
    print(encoded,end='')


if __name__ == '__main__':
    main()
