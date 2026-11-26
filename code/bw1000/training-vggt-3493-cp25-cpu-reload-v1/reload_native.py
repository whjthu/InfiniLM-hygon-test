"""Validate the exact project-owned CP25 and restore its camera head and AdamW on CPU."""
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

CP=Path('/checkpoint-25.pt')
FRAMEWORK=Path('/pilot/framework')
UPSTREAM=Path('/vggt-source')
BACKEND=FRAMEWORK/'generation/src/gencuda/backends/vggt_training.py'
BACKEND_SHA='762fc0d74a6ebb3ebbd052a8bdfbe302477a518a6868ec322bd89ef07e16c998'
UPSTREAM_SHA='64380b341ce377fb7e4b24c3f4f5986b6557b5339f4f7db57950dfa7e66b04c0'
EXTENT=2594191443
FILE_SHA='720b4adb7eea6173dd3fb2b9b834862fc371126031932fbe24e084efbeb352f0'
ORIGINAL_STAT=(23198152565,EXTENT,1795620615000000000,1795620615000000000,154459)
PINS=dict(model_SHA256='80e2f455942fadd91de23311c99500bc823636d1290e3be1ae0f5302a93e060b',
          optimizer_SHA256='b07973fed1b7c9a13105f822e2598c1c5248899bb83dc25a24618a3aeb308ea3',
          placements_SHA256='683c1e25e5ab7ba0d6e1eaa5824167282bebe17157ea8b4e18c3e55fe3cc5069',
          rng_SHA256='36bea4350d2e7d5c0f86eeffebf1fc308943bfb2cd478fbdcd5ff36878af7756',
          full_boundary_SHA256='27e3e628dbb163cc96e8f0ca17d6978a1453dbd88c8f776d8e4973faa7302d6c')


def need(ok,message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def stamp(s):
    return s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_uid


def frozen_bytes(path,wanted,size=None):
    need(path.resolve(strict=True)==path,'CANONICAL_FROZEN_MEMBER')
    with path.open('rb') as f:
        original=os.fstat(f.fileno())
        need(stat.S_ISREG(original.st_mode) and original.st_size<=2<<20
             and (size is None or original.st_size==size),'BOUNDED_FROZEN_MEMBER_EXTENT')
        data=f.read(original.st_size+1)
        after=os.fstat(f.fileno())
        need(len(data)==original.st_size and original.st_dev==after.st_dev
             and stamp(original)==stamp(after)==stamp(path.lstat()) and digest(data)==wanted,'EXACT_FROZEN_MEMBER_BYTES')
    return data


def checkpoint(torch,path=CP,expected_stamp=ORIGINAL_STAT,wanted=FILE_SHA):
    need(path.resolve(strict=True)==path,'CANONICAL_EXACT_CP25_BIND')
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as f:
        original=os.fstat(fd)
        need(stat.S_ISREG(original.st_mode) and stamp(original)==expected_stamp and
             os.fstatvfs(fd).f_flag&os.ST_RDONLY,'EXACT_READONLY_CP25_FD')
        h=sha256()
        length=0
        for chunk in iter(lambda:f.read(8<<20),b''):
            length+=len(chunk)
            need(length<=expected_stamp[1],'CP25_EXTENT_NO_GROWTH')
            h.update(chunk)
        need(length==expected_stamp[1] and h.hexdigest()==wanted and stamp(os.fstat(fd))==expected_stamp,
             'FULL_CP25_SHA_BEFORE_DESERIALIZATION')
        f.seek(0)
        payload=torch.load(f,map_location='cpu',weights_only=False)
        now=path.lstat()
        need(stamp(os.fstat(fd))==stamp(now)==expected_stamp and now.st_dev==original.st_dev,'EXACT_CP25_SAME_FD_AND_PATH_AFTER_LOAD')
        return payload


def original_placements(placements):
    need(type(placements) is dict and bool(placements),'SAVED_ORIGINAL_PLACEMENTS')
    for row in placements.values():
        need(type(row) is dict and set(row)=={'step','exp_avg','exp_avg_sq'} and
             row['step'] in ('cpu','cuda:0') and row['exp_avg']==row['exp_avg_sq']=='cuda:0',
             'SAVED_ADAMW_GPU_PLACEMENT_NOT_REPRODUCED_ON_CPU')
    return {i:{k:'cpu' for k in value} for i,value in placements.items()}


def apply_native(torch,b,model,payload):
    tree=b.resume_common.tree_sha256
    need(type(payload) is dict and set(payload)=={'metadata','state','rng','integrity_sha256'},'EXACT_NATIVE_PAYLOAD_SCHEMA')
    metadata=payload['metadata']
    contract=metadata['contract']
    need(contract['seed']==20260911 and contract['device_ids']==[0] and contract['warmup_steps']==5 and
         contract['max_steps']==20 and len(contract['view_order'])==8 and
         metadata==b._vggt_boundary_metadata(contract,25) and b._vggt_next_coordinate(metadata) is None,
         'EXACT_BOUNDARY_25_NO_STEP26')
    state=payload['state']
    need(type(state) is dict and set(state)=={'camera_head','optimizer','optimizer_placements'},'EXACT_SAVED_CAMERA_AND_ADAMW_SCHEMA')
    actual=dict(model_SHA256=tree(state['camera_head']),optimizer_SHA256=tree(state['optimizer']),
                placements_SHA256=tree(state['optimizer_placements']),rng_SHA256=tree(payload['rng']),
                full_boundary_SHA256=tree({k:v for k,v in payload.items() if k!='integrity_sha256'}))
    need(actual==PINS and payload['integrity_sha256']==PINS['full_boundary_SHA256'],'ALL_ORIGINAL_BOUNDARY_DIGESTS')
    b._vggt_validate_rng(payload['rng'],torch,[0],native=False)
    shadow=original_placements(state['optimizer_placements'])
    head,key=b._vggt_bind_trainable_scope(torch,model,'camera_head')
    need(key=='camera_head' and all(p.device.type=='cpu' and not p.requires_grad for p in model.aggregator.parameters()),
         'NATIVE_FROZEN_AGGREGATOR_CPU')
    params=[p for p in head.parameters() if p.requires_grad]
    optim=torch.optim.AdamW(params,lr=2e-5,weight_decay=.01)
    b._vggt_assert_optimizer_binding(optim,params)
    b._vggt_validate_state(dict(state,optimizer_placements=shadow),head,optim,25,torch)
    head.load_state_dict(state['camera_head'],strict=True)
    optim.load_state_dict(state['optimizer'])
    b._vggt_assert_optimizer_state_initialized(optim,params)
    b._vggt_assert_optimizer_binding(optim,params)
    live=optim.state_dict()
    places={i:{k:str(v.device) for k,v in optim.state[p].items()}
            for sg,lg in zip(live['param_groups'],optim.param_groups,strict=True)
            for i,p in zip(sg['params'],lg['params'],strict=True)}
    live_model=tree(b.resume_common.cpu_tree(head.state_dict()))
    live_optimizer=tree(b.resume_common.cpu_tree(live))
    need(places==shadow and all(p.device.type=='cpu' for p in model.parameters()) and
         (live_model,live_optimizer)==(PINS['model_SHA256'],PINS['optimizer_SHA256']),
         'FRESH_CPU_HEAD_ADAMW_HASH_AND_LIVE_PLACEMENT')
    return dict(saved_hashes=actual,live_model_SHA256=live_model,live_optimizer_SHA256=live_optimizer,
                live_CPU_placements_SHA256=tree(places),camera_named_tensors=len(head.state_dict()),
                optimizer_parameters=len(params),optimizer_slot_count=sum(map(len,places.values())),
                device_restore_verified=False,original_GPU_rng_and_placements_record_verified=True,
                warmup_steps=5,max_steps=20,global_step=25,next_batch_cursor=1,no_step26=True,
                optimizer_updates=0,forward_calls=0,original_frozen_weight_payload_read=False)


def main():
    step=sys.argv[1]
    need(re.fullmatch(r'1103606\.[0-9]+',step) is not None and not os.environ.get('SLURM_STEP_GPUS')
         and all(os.environ.get(k)=='' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')),
         'ZERO_GRES_ONE_NUMERIC_CPU_STEP')
    for mount in (CP,FRAMEWORK,UPSTREAM,Path('/audit-source')):
        need(os.statvfs(mount).f_flag&os.ST_RDONLY,'ORIGINAL_AND_AUDIT_MOUNTS_READONLY')
    items=json.loads(frozen_bytes(Path('/audit-source/UPSTREAM_FILES.json'),UPSTREAM_SHA))['files']
    names=set()
    for item in items:
        p=PurePosixPath(item['name'])
        need(not p.is_absolute() and '..' not in p.parts and str(p)==item['name'] and p not in names,'FROZEN_UPSTREAM_SCOPE')
        names.add(p)
        frozen_bytes(UPSTREAM/p,item['SHA256'],item['bytes'])
    need(len(names)==69,'ALL_69_NATIVE_UPSTREAM_SOURCES')
    frozen_bytes(BACKEND,BACKEND_SHA)
    import torch
    need(torch.version.hip=='6.3.26113' and importlib.metadata.version('torch')==
         '2.10.0+das.opt1.dtk2604.2606161421.g91a580','EXACT_3493_NATIVE_TORCH')
    torch.set_num_threads(4)
    from gencuda.backends import vggt_training as b
    sys.path.insert(0,str(UPSTREAM))
    from vggt.models.vggt import VGGT
    need(Path(b.__file__)==BACKEND and
         Path(sys.modules[VGGT.__module__].__file__)==UPSTREAM/'vggt/models/vggt.py','EXACT_IMPORTED_MODEL_BACKEND_SOURCE')
    payload=checkpoint(torch)
    model=VGGT(enable_camera=True,enable_point=False,enable_depth=False,enable_track=False)
    result=apply_native(torch,b,model,payload)
    print('VGGT_CP25_CPU '+json.dumps(dict(result,passed=True,actual_GPU=0,step=step,upstream=69,
          backend_SHA256=BACKEND_SHA,checkpoint_bytes=EXTENT,checkpoint_SHA256=FILE_SHA),sort_keys=True),flush=True)


if __name__=='__main__':
    main()
