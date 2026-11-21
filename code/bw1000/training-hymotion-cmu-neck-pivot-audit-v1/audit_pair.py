"""One source-orientation neck gauge, with a fixed-position R11 feasibility bound.

This never fits to Mean/Std, changes anchors, or creates a training cache.
"""
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
import time

import numpy as np

D=Path(__file__).resolve().parent
G=D.parents[2]
L=G/'local-only'
OUT=L/'training-hymotion-cmu-neck-pivot-audit-v1-original'
SOURCE=G/'code/bw1000'
CALIBRATION=128


def need(ok,reason):
    if not ok:raise ValueError(reason)


def pin(raw):return {'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}


def fixed(path,size,sha):
    raw=path.read_bytes();need(pin(raw)=={'bytes':size,'SHA256':sha},'frozen input differs: '+path.name);return raw


def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def manifest(directory,size,sha):
    m=json.loads(fixed(directory/'SOURCE_FILES.json',size,sha))
    for name,p in m['files'].items():fixed(directory/name,p['bytes'],p['SHA256'])


def score(values,mean,denom,neck):
    z=(values-mean)/denom;section=z[:,3+neck*6:3+(neck+1)*6]
    return {'frames':len(values),'Neck_RMS_z':float(np.sqrt(np.mean(section**2))),
        'Neck_abs_max_z':float(np.max(np.abs(section))),'Neck_entries_above10':int(np.sum(np.abs(section)>10)),
        'channel78_abs_max_z':float(np.max(np.abs(z[:,78]))),'all_rot6d_RMS_z':float(np.sqrt(np.mean(z[:,3:135]**2)))}


def feasibility(local,axis,mean,std,neck):
    # R*a=b fixes the outgoing bone direction. For any proper R satisfying this,
    # R_yy is in a_y*b_y +/- sqrt(1-a_y^2)*sqrt(1-b_y^2). This is an exact
    # SO(2) gauge interval, not a sampled angle search or a distribution prior.
    b=(local@axis[:,None])[...,0]
    center=axis[1]*b[:,1]
    radius=np.sqrt(max(0.,1-axis[1]**2))*np.sqrt(np.maximum(0.,1-b[:,1]**2))
    low,high=center-radius,center+radius
    index=3+neck*6+3;need(index==78,'frozen channel78 is no longer Neck R11')
    distance=np.maximum(np.maximum(low-mean[index],mean[index]-high),0)/std[index]
    return {'channel':index,'meaning':'Neck local R11 (second row/second column)',
        'fixed_position_possible_R11_min':float(low.min()),'fixed_position_possible_R11_max':float(high.max()),
        'max_irreducible_absolute_z':float(distance.max()),'RMS_irreducible_absolute_z':float(np.sqrt(np.mean(distance**2))),
        'frames_every_position_preserving_rotation_above5':int((distance>5).sum()),
        'frames_every_position_preserving_rotation_above10':int((distance>10).sum())},low,high


def main():
    began=time.monotonic();OUT.mkdir()
    result={'schema':'CMU_WOODEN_NECK_PIVOT_ONE_GAUGE_PAIR_V1','passed':False,'planned_frames':470,
        'calibration_frames':128,'later_same_recording_frames':342,'candidate_count':1,'actual_HTTP_GPU_training':0,
        'formal_T08_changed':False,'READY_added':0,'phase':'source-and-input-identity'}
    try:
        root=SOURCE/'training-hymotion-cmu-wooden-root-v1';fit=SOURCE/'training-hymotion-cmu-wooden-position-fit-v1';bind=SOURCE/'training-hymotion-wooden-rest-binding-v1'
        manifest(root,600,'d17872c6d1664b20af9dc6e4d16d5dcf8e9e99e6a002a084726913d03bc65745')
        manifest(fit,739,'b74668bffe8303bfc655a1ed4ec627b3ea5280effffca2ca89026f6efc53239d')
        manifest(bind,355,'73214bf524166c7b86ca04d3a1461ece7ae51e0852b79f0156e6c77eb6d332e8')
        old=module('frozen_root_geometry',root/'assemble_motion.py');rig=module('frozen_rest_binding',bind/'bind_rest.py')
        source_rig,decode=old.prior_math();names,rest22,parent22=source_rig();rest,parent=old.full_skeleton()
        frames,six,rotation_names,_=old.rotations();need(names==rotation_names and len(frames)==470,'original30Hz inputs differ')
        blob=fixed(L/'hymotion-cmu-wooden-root-v1-original/motion135-and-CPU-geometry201-preview-private.npz',856445,
            '1aa6b6292f8a56b6a9fcf77533389533288df942b1c218d4e03c53b0de86d548')
        with np.load(BytesIO(blob),allow_pickle=False) as z:
            original=z['cpu_geometry201'];root_positions=z['motion135'][:,:3];need(np.array_equal(z['source_frame_numbers'],frames),'source time differs')
        need(np.array_equal(original[:,:3],root_positions) and np.array_equal(original[:,3:135].reshape(470,22,6),six),'original135 pose identity differs')
        h=json.loads(fixed(L/'training-hymotion-cmu-asf-fk-candidate-v1-original/hierarchy-and-conventions-private.json',14452,
            'e7374b7a18d16255506febbc6575272beca20803c8abfb8b00f256fd71032aa0'))
        cmu=fixed(L/'training-hymotion-cmu-asf-fk-candidate-v1-original/world-joints-native-meters-private.npz',2679326,
            '4ca26a839adf332c5525c78a3443f841f97700d608c0052e13763ac11e3d55fc')
        with np.load(BytesIO(cmu),allow_pickle=False) as z:
            cmu_world=z['world_rotations'];cmu_bases=z['bone_bases_m'];cmu_ends=z['joint_endpoints_m']
        cn=h['joint_names'];bones={b['name'][0]:b for b in h['bones']};cmu_rest={'root':np.zeros(3)}
        for n,p in zip(cn[1:],h['parent_indices'][1:]):cmu_rest[n]=cmu_rest[cn[p]]+np.asarray(bones[n]['direction'],float)*float(bones[n]['length'][0])*h['physical_raw_unit_to_meters']
        neck,head=(names.index(n) for n in ('Neck','Head'));need(neck==12 and parent[head]==neck and np.flatnonzero(parent==neck).tolist()==[head] and not (parent==head).any(),'single terminal head successor needed')
        result['phase']='pivot-and-source-frame-audit'
        lower,upper,source_head=(cn.index(n) for n in ('lowerneck','upperneck','head'))
        lower_to_upper=float(np.max(np.abs(cmu_ends[:,lower]-cmu_bases[:,upper-1])))
        upper_to_head=float(np.max(np.abs(cmu_ends[:,upper]-cmu_bases[:,source_head-1])))
        need(lower_to_upper==upper_to_head==0,'actual CMU pivot/base relation differs')
        sp=rig.basis(cmu_rest['lhipjoint']-cmu_rest['rhipjoint'],cmu_rest['thorax'])
        tp=rig.basis(rest22[names.index('L_Hip')]-rest22[names.index('R_Hip')],rest22[names.index('Spine3')]-rest22[0])
        Q=tp@sp.T
        need(np.max(np.abs(Q.T@Q-np.eye(3)))<1e-12 and abs(np.linalg.det(Q)-1)<1e-12,'same rest basis is not proper')
        a=rest[head]-rest[neck];a/=np.linalg.norm(a)
        b=Q@np.asarray(bones['upperneck']['direction'],float);b/=np.linalg.norm(b)
        minimum_turn=sys.modules['fit_poses'].minimum_turn
        A,opposite=minimum_turn(a[None],b[None]);need(opposite==0,'rest neck alignment underdetermined half-turn')
        local=decode(six);world,points=old.transform(local,rest,parent)
        need(float(np.max(np.abs(points[:,:22].reshape(470,66)-original[:,135:])))<1e-6,'frozen keypoint preview differs')
        source30=sys.modules['resample_rotations'].decimate(cmu_world[:,[upper]])
        centers,sourceR,max_angle=source30;need(np.array_equal(centers+1,frames),'same binomial timestamps changed')
        reference=Q@sourceR[:,0]@Q.T@A[0]
        fitted_direction=(world[:,neck]@a[:,None])[...,0]
        reference_direction=(reference@a[:,None])[...,0]
        swing,half_turn=minimum_turn(reference_direction,fitted_direction);need(half_turn==0,'source gauge projection half-turn ambiguity')
        corrected_world=swing@reference
        candidate=six.copy();newlocal=world[:,parent[neck]].swapaxes(1,2)@corrected_world
        candidate[:,neck]=newlocal[...,:2].reshape(470,6).astype(np.float32)
        candidate_local=decode(candidate);_,after=old.transform(candidate_local,rest,parent)
        change=float(np.max(np.abs(after-points)));need(change<1e-6,'single twist projection changed geometry')
        # Freeze all old non-neck data bytes. FK is separately checked above so
        # reused keypoints cannot conceal a geometrically inconsistent new pose.
        paired=original.copy();section=slice(3+neck*6,3+(neck+1)*6);paired[:,section]=candidate[:,neck]
        other=np.ones(201,bool);other[section]=False
        need(np.array_equal(paired[:,other],original[:,other]),'non-neck source channels changed')
        mean=np.load(BytesIO(old.fixed(old.OFFICIAL/'Mean.npy',1736,'a4bbe5f68881fe793d509f262ad3dc698a91f79158a9bde9d6f24d46ef9652bd')),allow_pickle=False).astype(np.float32)
        std=np.load(BytesIO(old.fixed(old.OFFICIAL/'Std.npy',1736,'6cc86d1ca4b47fbd0e84b677c4397e51b6615abcb34e38fdd9daabbe6a5685cc')),allow_pickle=False).astype(np.float32)
        denom=np.where(std<1e-3,np.float32(1),std)
        scope={'whole470':slice(None),'first128':slice(0,128),'later342_same_recording':slice(128,None)}
        paired_scores={n:{'before':score(original[s],mean,denom,neck),'candidate':score(paired[s],mean,denom,neck)} for n,s in scope.items()}
        bounds={};low_full,high_full=None,None
        for n,s in scope.items():
            bounds[n],lo,hi=feasibility(local[s,neck],a,mean,denom,neck)
            if n=='whole470':low_full,high_full=lo,hi
        old_R11=local[:,neck,1,1];new_R11=candidate_local[:,neck,1,1]
        need(np.all(old_R11>=low_full-1e-10) and np.all(old_R11<=high_full+1e-10) and
             np.all(new_R11>=low_full-1e-7) and np.all(new_R11<=high_full+1e-7),'exact fixed-direction interval implementation error')
        source_pivot={'CMU_lowerneck_tip_equals_upperneck_base_all940':lower_to_upper,'CMU_upperneck_tip_equals_head_base_all940':upper_to_head,
            'current_Wood_Neck_anchor':'CMU lowerneck distal endpoint = upperneck pivot',
            'current_Wood_Head_anchor':'CMU upperneck distal endpoint = head pivot',
            'outgoing_CMU_segment_for_Wood_Neck_to_Head':'upperneck; lowerneck rotation acts at preceding pivot',
            'Wood_Neck_to_Head_rest_unit_axis':a.tolist(),'proper_rest_basis_Q':Q.tolist(),
            'source_CMU_upperneck_axis_and_DOF':{'axis':bones['upperneck']['axis'],'dof':bones['upperneck']['dof']},
            'all52_position_change_max_rig_units':change,'physical_anatomical_crossrig_equivalence_proven':False}
        np.savez_compressed(OUT/'neck-source-gauge-pair-not-training-private.npz',source_frame_numbers=frames,
            original201=original,candidate201=paired,original_neck_local=local[:,neck],candidate_neck_local=candidate_local[:,neck],
            channel78_possible_min=low_full,channel78_possible_max=high_full)
        artifact=pin((OUT/'neck-source-gauge-pair-not-training-private.npz').read_bytes())
        result.update(passed=True,state='ONE_POSITION_INVARIANT_SOURCE_ORIENTATION_PAIR_DIAGNOSTIC',phase='complete',
            completed_frames=470,pivot_evidence=source_pivot,paired_scores=paired_scores,
            exact_fixed_position_R11_feasibility=bounds,non_neck_195_channels_bitwise_preserved=True,
            root_keypoints_time_and_all_other_joints_unchanged=True,original_recoverable_exact=True,
            source_orientation_projection_uses_statistics=False,candidate_new_grid_searches=0,source_upperneck_rotation_max_filter_radius_degrees=float(np.degrees(max_angle)),
            actor_target_twist_groundtruth_proven=False,domain_training_READY=False,formal_T08=False,
            maximum_neck_local_orthogonality_error=float(np.max(np.abs(newlocal.swapaxes(1,2)@newlocal-np.eye(3)))),
            candidate_artifact=artifact,wall_seconds=time.monotonic()-began)
        result['candidate_adopted']=False
        result['decision']='Do not adopt; positional gauge audit only. Large channel78 fixed-position lower bound cannot be repaired by selecting a neck twist. Cross-rig pivot equivalence and actual target reference remain unqualified.'
    except Exception as e:
        import traceback
        (OUT/'failure-private.json').write_text(json.dumps({'phase':result['phase'],'error_type':type(e).__name__,'reason':str(e),'trace':traceback.format_exc()},indent=2)+'\n',encoding='utf8')
        result.update(state='FAILED_PARTIAL_PRESERVED',error_type=type(e).__name__,failure=pin((OUT/'failure-private.json').read_bytes()))
    (OUT/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['passed','state','planned_frames','READY_added']},sort_keys=True))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
