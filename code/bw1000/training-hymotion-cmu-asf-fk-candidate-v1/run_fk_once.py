"""Offline 940-frame FK and actual official C++ parser/FK cross-check."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

D=Path(__file__).resolve().parent
G=D.parents[2]
RUN=G/'local-only/training-hymotion-cmu-asf-fk-candidate-v1-original'
REF=G/'local-only/training-hymotion-cmu-asf-fk-candidate-v1-reference-original/AMCViewer.zip'
ASF=G/'local-only/training-hymotion-public-indoor-motion-candidate-v1-original/79.asf'
AMC=G/'local-only/training-hymotion-public-indoor-motion-candidate-amc-only-v2-original/79_81.amc'
PARSER=G/'code/bw1000/training-hymotion-public-indoor-motion-candidate-amc-only-v2/parse_asf_amc.py'
COMPILER=Path('C:/Software/Perl/c/bin/g++.exe')
EXPECTED={str(REF):'0cc57874e66a77263857a7b7b4af2b3a35a4c033775aded84091eb725a3734f0',
    str(ASF):'aa7fb254b0e7b9155df59e0f0cf76cbe34c476c6f4b3f015e63d5ae170d2fe2d',
    str(AMC):'1b0886a7148900121bfe2716c8825fa481c80163d1e873b0edb2ea89246fe489',
    str(PARSER):'cc53c606986d1bc57630d79319da6b01ea1fc01dcf165edc9736df5cbf8c779e'}
MEMBERS=['Library/Skeleton.cpp','Library/Skeleton.hpp','Library/ReadSkeleton.cpp','Library/ReadSkeleton.hpp',
    'Library/Reader.cpp','Library/Reader.hpp','Character/Character.cpp','Character/Character.hpp',
    'Character/pose_utils.cpp','Character/pose_utils.hpp','Vector/Vector.hpp','Vector/Quat.hpp']


def need(ok,why):
    if not ok:raise ValueError(why)


def pin(path):
    b=path.read_bytes();return dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())


def save(name,obj):
    (RUN/name).write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf8')


def execute(argv,name,deadline):
    left=deadline-time.time()-5;need(left>0,'original offline epoch exhausted')
    with (RUN/(name+'.stdout')).open('xb') as out,(RUN/(name+'.stderr')).open('xb') as err:
        p=subprocess.run(argv,stdout=out,stderr=err,timeout=left,cwd=RUN)
    record=dict(argv=argv,returncode=p.returncode,stdout=pin(RUN/(name+'.stdout')),stderr=pin(RUN/(name+'.stderr')))
    save(name+'-receipt.json',record);need(p.returncode==0,name+' returned failure');return record


def worker(deadline):
    while not (RUN/'job-gate.json').exists():
        need(time.time()<deadline,'Job gate deadline');time.sleep(.01)
    result=dict(schema='CMU_ASF_ACTUAL940_FK_OFFLINE_V1',passed=False,planned_frames=940,completed_frames=0,
        planned_bones=30,planned_segment_observations=28200,actual_new_GETs=0,actual_GPU_or_Torch_or_training=0,
        T08_formal_recipe_changed=False,HY135_or201_mapping=False,READY=False,phase='input-pins',
        frozen_point_tolerance_m=0.00001,frozen_rotation_tolerance=0.00001,frozen_nominal_bone_length_tolerance_m=0.000001)
    try:
        gate=json.loads((RUN/'job-gate.json').read_bytes());need(gate['CPU_count']==2 and gate['memory_limit_bytes']==2147483648,'Job gate differs')
        for name,sha in EXPECTED.items():need(pin(Path(name))['SHA256']==sha,'frozen input differs: '+Path(name).name)
        result['input_pins']={str(Path(p).relative_to(G)):pin(Path(p)) for p in EXPECTED}
        spec=importlib.util.spec_from_file_location('frozen_native_parser',PARSER);parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
        skeleton=parser.parse_asf(ASF.read_bytes())
        root={};section=None
        for line in ASF.read_text().splitlines():
            line=line.partition('#')[0].strip()
            if line.startswith(':'):section=line[1:].split()[0]
            elif section=='root' and line:
                k,*values=line.split();root[k]=values
        need(root=={'order':['TX','TY','TZ','RX','RY','RZ'],'axis':['XYZ'],'position':['0','0','0'],'orientation':['0','0','0']},'actual root convention differs; no guessed transformation')
        import numpy as np
        from fk_matrix import solve
        motion=np.asarray(parser.parse_amc(AMC.read_bytes(),skeleton),dtype=np.float64)
        need(motion.shape==(940,62),'actual complete940 native sequence required')
        result['phase']='matrix-forward-kinematics';fk=solve(skeleton,motion)
        need(np.isfinite(fk['joint_endpoints_m']).all() and np.isfinite(fk['world_rotations']).all(),'nonfinite FK')
        result['phase']='official-CMU-source-extraction';reference=RUN/'reference-source';reference.mkdir()
        extracted={}
        with zipfile.ZipFile(REF) as z:
            for name in MEMBERS:
                raw=z.read('AMCViewer/'+name);need(len(raw)<=256*1024,'single reference source bound')
                p=reference/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);extracted[name]=pin(p)
        original=(reference/'Character/pose_utils.cpp').read_bytes()
        start=original.index(b'void get_world_bones(');stop=original.index(b'float world_distance(',start)
        function=original[start:stop]
        isolated=reference/'official_world_bones_isolated.cpp'
        isolated.write_bytes(b'#include <Library/Skeleton.hpp>\n#include <Character/pose_utils.hpp>\nnamespace Character {\n'+function+b'\n}\n')
        result['official_reference']={'archive':pin(REF),'members':extracted,'world_bones_function_exact_bytes':len(function),
            'world_bones_function_SHA256':hashlib.sha256(function).hexdigest(),'isolated_wrapper':pin(isolated),
            'original_math_or_parser_body_modified':False,'CXX_float_pose_coordinates':True,'reference_internal_length_scale':.45,
            'reference_internal_to_meters':.0254/(.45*.45),'physical_raw_to_meters':.0254/.45}
        save('reference-source-pins.json',result['official_reference'])
        result['phase']='compile-official-CMU-CPU-oracle';executable=RUN/'official_fk_oracle.exe'
        cmd=[str(COMPILER),'-std=gnu++98','-DWINDOWS','-O0','-I'+str(reference),str(D/'oracle_harness.cpp'),
            str(reference/'Library/Skeleton.cpp'),str(reference/'Library/ReadSkeleton.cpp'),str(reference/'Library/Reader.cpp'),
            str(reference/'Character/Character.cpp'),str(isolated),'-lws2_32','-o',str(executable)]
        result['compile']=execute(cmd,'compile',deadline)
        # Official ReadAnimation first looks for a same-name .bmc. New isolated input
        # contains exactly the two pinned originals; the .bmc alternative cannot exist.
        source=RUN/'oracle-input';source.mkdir();shutil.copyfile(ASF,source/'79.asf');shutil.copyfile(AMC,source/'79_81.amc')
        need(not (source/'79_81.bmc').exists() and pin(source/'79.asf')==pin(ASF) and pin(source/'79_81.amc')==pin(AMC),'oracle originals differ')
        result['phase']='actual-official-CMU-parser-and-FK'
        result['oracle']=execute([str(executable),str(source/'79.asf'),str(source/'79_81.amc'),str(RUN/'oracle-world-private.bin'),
            str(RUN/'oracle-skeleton-private.txt'),str(RUN/'oracle-channels-private.bin')],'oracle',deadline)
        metadata=(RUN/'oracle-skeleton-private.txt').read_text().splitlines();need(metadata[0]=='0.45000000000000001 62 XYZxyz','official root/units/order differs')
        reference_bones=[line.split() for line in metadata[1:]];need(len(reference_bones)==30,'official bone denominator differs')
        order=[b[0] for b in reference_bones];our={b['name'][0]:b for b in skeleton['bones']};parents={c:p for p,c in skeleton['hierarchy_edges']}
        channels=np.fromfile(RUN/'oracle-channels-private.bin',dtype='<f8').reshape(940,62)
        errors=[float(np.max(np.abs(channels[:,:3]/.45-motion[:,:3]))),float(np.max(np.abs(channels[:,3:6]-motion[:,3:6])))]
        cursor=6;ourslices={}
        for b in skeleton['bones']:
            name=b['name'][0];n=len(b.get('dof',[]));ourslices[name]=slice(cursor,cursor+n);cursor+=n
        for b in reference_bones:
            name,parent,ofs,dof,length,*rest=b;definition=our[name];n=len(definition.get('dof',[]))
            need(parent==parents[name] and dof==(''.join(d[1] for d in definition.get('dof',[])) or '-') and rest[6]=='xyz','official hierarchy/DOF/axis order differs')
            need(np.array_equal(np.asarray(rest[:3],float),np.asarray(definition['direction'],float)) and np.array_equal(np.asarray(rest[3:6],float),np.asarray(definition['axis'][:3],float)),'official direction/axis parse differs')
            need(abs(float(length)/.45-float(definition['length'][0]))<1e-12,'official bone length parse differs')
            if n:errors.append(float(np.max(np.abs(channels[:,int(ofs):int(ofs)+n]-motion[:,ourslices[name]]))))
        need(max(errors)<1e-12,'official independent AMC parser values differ')
        reference_array=np.fromfile(RUN/'oracle-world-private.bin',dtype='<f8').reshape(940,303)
        ouridx=[fk['names'].index(n) for n in order];official=reference_array[:,3:].reshape(940,30,10)
        root_error=float(np.max(np.abs(reference_array[:,:3]-fk['joint_endpoints_m'][:,0])))
        base_error=float(np.max(np.abs(official[:,:,:3]-fk['bone_bases_m'][:,np.asarray(ouridx)-1])))
        tip_error=float(np.max(np.abs(official[:,:,3:6]-fk['joint_endpoints_m'][:,ouridx])))
        q=official[:,:,6:];q=q/np.linalg.norm(q,axis=-1,keepdims=True);x,y,z,w=[q[:,:,i] for i in range(4)]
        refR=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(940,30,3,3)
        rotation_error=float(np.max(np.abs(refR-fk['world_rotations'][:,ouridx])))
        lengths=np.asarray([float(b['length'][0])*skeleton['root_length_to_meters'] for b in skeleton['bones']])
        actual_lengths=np.linalg.norm(fk['joint_endpoints_m'][:,1:]-fk['bone_bases_m'],axis=-1)
        direction_norm=np.asarray([np.linalg.norm(np.asarray(b['direction'],float)) for b in skeleton['bones']])
        nominal_error=float(np.max(np.abs(actual_lengths-lengths)))
        offset_error=float(np.max(np.abs(actual_lengths-lengths*direction_norm)))
        length_drift=float(np.max(np.ptp(actual_lengths,axis=0)))
        orthogonal=float(np.max(np.abs(np.swapaxes(fk['world_rotations'],-1,-2)@fk['world_rotations']-np.eye(3))))
        determinant=float(np.max(np.abs(np.linalg.det(fk['world_rotations'])-1)))
        result.update(completed_frames=940,actual_bones=30,actual_joint_endpoints_including_root=31,
            complete_parent_edges=30,actual_segment_observations=28200,actual_position_scalar_values=87420,
            official_parser_max_native_channel_error=max(errors),official_root_max_abs_m=root_error,
            official_bases_max_abs_m=base_error,official_tips_max_abs_m=tip_error,official_rotation_max_abs=rotation_error,
            bone_length_max_vs_ASF_nominal_m=nominal_error,bone_length_max_vs_raw_direction_offset_m=offset_error,
            bone_length_max_across940frames_drift_m=length_drift,ASF_direction_max_norm_deviation=float(np.max(abs(direction_norm-1))),
            directions_renormalized=False,orthogonality_max_abs=orthogonal,determinant_max_abs_error=determinant,
            root_units='meters; .0254/.45 applied equally to ASF lengths and AMC root translations',
            coordinate_convention='right-handed native CMU Y-up; column vectors; XYZ=Rz@Ry@Rx; no camera/SMPL transform',
            bounds={'point_m':1e-5,'rotation':1e-5,'nominal_bone_length_m':1e-6,'exact_direction_bone_length_m':1e-12,'rotation_orthogonal':1e-12},
            CXX_oracle_is_actual_executed=True,synthetic_frames_or_padding=0,body_model_or_HY201_proof=False)
        need(max(root_error,base_error,tip_error)<1e-5 and rotation_error<1e-5,'actual official FK discrepancy')
        need(nominal_error<1e-6 and offset_error<1e-12 and length_drift<1e-12 and max(orthogonal,determinant)<1e-12,'bone length/rigid rotation discrepancy')
        np.savez_compressed(RUN/'world-joints-native-meters-private.npz',frame_numbers=np.arange(1,941,dtype=np.int32),
            joint_endpoints_m=fk['joint_endpoints_m'],bone_bases_m=fk['bone_bases_m'],world_rotations=fk['world_rotations'],
            joint_names=np.asarray(fk['names']),parent_indices=np.asarray(fk['parents'],dtype=np.int32),source_fps=np.asarray(60))
        hierarchy={'root_convention':root,'node_definition':'root position plus named bone distal endpoints; each bone base is its parent endpoint, not SMPL anatomical labels',
            'joint_names':fk['names'],'parent_indices':fk['parents'],'bones':skeleton['bones'],'hierarchy_edges':skeleton['hierarchy_edges'],
            'editing_only_bones':skeleton['editing_only_bones'],'physical_raw_unit_to_meters':skeleton['root_length_to_meters']}
        save('hierarchy-and-conventions-private.json',hierarchy)
        result['artifacts']={name:pin(RUN/name) for name in ['world-joints-native-meters-private.npz','hierarchy-and-conventions-private.json','oracle-world-private.bin','oracle-channels-private.bin','oracle-skeleton-private.txt','reference-source-pins.json']}
        result.update(passed=True,state='ACTUAL940_NATIVE_ASF_FK_CXX_ORACLE_VERIFIED_NOT_HY201',phase='complete')
    except Exception as e:
        import traceback
        save('failure-private.json',dict(phase=result['phase'],error_type=type(e).__name__,reason=str(e)[:1024],traceback=traceback.format_exc()[-4096:]))
        result.update(state='FAILED_PARTIAL_PRESERVED',error_type=type(e).__name__,failure=pin(RUN/'failure-private.json'))
    save('result.json',result)
    print(json.dumps({k:result[k] for k in ['passed','state','phase','planned_frames','completed_frames','READY']},sort_keys=True))
    return 0 if result['passed'] else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--deadline',type=float);a=p.parse_args()
    if a.worker:return worker(a.deadline)
    epoch=time.time();mono=time.monotonic();RUN.mkdir()
    for path in D.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    save('intent.json',dict(request_id='8a03b9b02d7a45f49c5e2b0bd5b16066',epoch=epoch,deadline=epoch+180,CPU=2,Job_commit_cap=2147483648,
        source_pins={p.name:pin(p) for p in D.iterdir() if p.is_file()},planned_frames=940,planned_segment_observations=28200,GETs=0,training_GPU_or_HY201=False))
    helper=G/'code/bw1000/training-USGS-multievent-render144-v2/job_guard.py'
    need(pin(helper)['SHA256']=='c52244b5c239852c09ec2b7c5bd6aa1ffa4056eaf4e822946aaac46c9cd12b71','qualified Job helper differs')
    spec=importlib.util.spec_from_file_location('qualified_fk_job',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    job=m.guarded_run([sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--worker','--deadline',str(epoch+180)],RUN/'job-gate.json',RUN,epoch+180,mono+180)
    save('outcome.json',dict(state='KNOWN_TERMINAL',epoch=epoch,deadline=epoch+180,Job=job,result=pin(RUN/'result.json') if (RUN/'result.json').exists() else None))
    print(json.dumps(dict(returncode=job['returncode'],wall_seconds=job['wall_seconds'],output=str(RUN)),sort_keys=True));return job['returncode']


if __name__=='__main__':raise SystemExit(main())
