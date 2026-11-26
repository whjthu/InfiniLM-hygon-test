"""Verify only nineteen frozen household pairs and run canonical VGGT CPU tensor conversion."""
import argparse
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import re
import stat
import sys

BACKEND=Path('/pilot/framework/generation/src/gencuda/backends/vggt_training.py')
EXPECTED_BACKEND='762fc0d74a6ebb3ebbd052a8bdfbe302477a518a6868ec322bd89ef07e16c998'
EXPECTED_MATH='c453a9c561c6411b8c2f25652f5c004c928db423c03ca0a45eba7d47ed9e3962'
LAYOUT={'images':(1,2,3,224,224),'depths':(1,2,224,224),'extrinsics':(1,2,3,4),
        'intrinsics':(1,2,3,3),'point_masks':(1,2,224,224)}
ARRAY_ORDER=('images','depths','intrinsics','extrinsics','point_masks')
SHA=re.compile(r'[0-9a-f]{64}')
PNG=re.compile(r'(rgb|depth)/[0-9]+\.[0-9]+\.png')


def need(ok,reason):
    if not ok:raise ValueError(reason)


def hash_bytes(raw):
    return sha256(raw).hexdigest()


def doc(data):
    return (json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def find(path,digest,expected=None,cap=4<<20):
    need(SHA.fullmatch(digest) and path.resolve(strict=True)==path,'EXACT_CANONICAL_BOUND_INPUT')
    before=path.lstat()
    meta=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=cap and
         (expected is None or before.st_size==expected),'EXACT_BOUND_INPUT_EXTENT')
    with path.open('rb') as f:
        opened=os.fstat(f.fileno())
        raw=f.read(before.st_size+1)
        after=os.fstat(f.fileno())
    need(meta(before)==meta(opened)==meta(after)==meta(path.lstat()) and
         len(raw)==before.st_size and hash_bytes(raw)==digest,'EXACT_BOUND_INPUT_BYTES')
    return raw


def manifest(source,sha):
    spec=json.loads(find(source/'INPUT_FILES.json',sha,cap=65536))
    slots=spec['slots'];files=spec['files']
    need(spec['schema']=='VGGT_EXACT_HOUSEHOLD_19_OF_20_NATIVE_INPUTS_V1' and
         spec['original_backend_SHA256']==EXPECTED_BACKEND and spec['geometry_math_SHA256']==EXPECTED_MATH
         and spec['model_or_score_GO'] is False and spec['planned_common']==20 and spec['planned_scoring']==40
         and [r['slot'] for r in slots]==list(range(1,21)) and [r['slot'] for r in slots if r['status']=='FROZEN_UNAVAILABLE']==[15]
         and sum(r['status']=='FROZEN_QUALIFIED' for r in slots)==spec['qualified_for_native_check']==19
         and len(files)==79,'EXACT_FIXED_NINETEEN_OF_TWENTY_DENOMINATOR')
    metadata={'rgb.txt','depth.txt','groundtruth.txt'}
    need(metadata<=files.keys() and all(PNG.fullmatch(p) for p in files.keys()-metadata),'ONLY_ORIGINAL_SELECTED_MEDIA_AND_THREE_INDICES')
    need({name for r in slots if r['status']=='FROZEN_QUALIFIED' for name in r['pair']['files_SHA256']}==set(files),
         'EACH_STAGED_MEMBER_BELONGS_TO_A_FROZEN_PAIR')
    return spec


def audit_assets(data,spec):
    actual={p.relative_to(data).as_posix() for p in data.rglob('*') if p.is_file()}
    need(actual==set(spec['files']) and all(not p.is_symlink() for p in data.rglob('*')),'EXACT_REDUCED_ASSET_TREE')
    size=0
    for name,pin in spec['files'].items():
        size+=len(find(data/name,pin['SHA256'],pin['bytes']))
    return size


def arrays_sha(batch):
    h=sha256()
    for key in ARRAY_ORDER:
        for val in batch[key]:
            h.update(doc([key,str(val.dtype),list(val.shape)]))
            h.update(val.tobytes(order='C'))
    return h.hexdigest()


def capture(torch,backend,geometry,asset_root,spec):
    result=[]
    captured=[]
    for row in spec['slots']:
        stage=dict(slot=row['slot'])
        if row['status']=='FROZEN_UNAVAILABLE':
            result.append(dict(stage,state='FROZEN_UNAVAILABLE',reason=row['reason']))
            continue
        try:
            raw,_=geometry['prepare_pair'](asset_root,row['pair'])
            need(arrays_sha(raw)==row['geometry_arrays_SHA256'],'ACTUAL_GEOMETRY_ARRAYS_CHANGED')
            batch=backend._tensor_batch(torch,raw)
            need(batch['seq_name']==row['pair']['sequence'] and batch['ids']==row['pair']['rgb_indices']
                 and all(tuple(batch[key].shape)==shape and batch[key].device.type=='cpu' and
                         batch[key].dtype==(torch.bool if key=='point_masks' else torch.float32)
                         and not batch[key].requires_grad and bool(torch.isfinite(batch[key]).all())
                         for key,shape in LAYOUT.items()) and backend.valid_depth_batch(batch['point_masks']),
                 'ORIGINAL_NATIVE_CPU_TENSOR_BATCH_CONTRACT')
            per_pair=backend._vggt_batch_content_sha256([batch])
            need(SHA.fullmatch(per_pair),'EXACT_NATIVE_INPUT_HASH')
            captured.append(batch)
            result.append(dict(stage,state='NATIVE_CPU_QUALIFIED',native_input_SHA256=per_pair))
        except (OSError,ValueError,TypeError,KeyError,RuntimeError) as error:
            result.append(dict(stage,state='NATIVE_CPU_FAILED',error_type=type(error).__name__,
                               error_SHA256=hash_bytes(str(error).encode())))
    return dict(passed=len(captured)==19,planned_common_pairs=20,planned_scoring_slots=40,qualified_input_pairs=len(captured),
        frozen_unavailable=1,failed_eligible=19-len(captured),rows=result,
        ordered_native_batch_SHA256=backend._vggt_batch_content_sha256(captured) if len(captured)==19 else None,
        ready_for_scoring=False,actual_predictions=0,model_weights_opened=0,actual_GPU=0)


def run(a):
    source=Path(__file__).resolve().parent
    for x in (a.assets,source):
        need(x.resolve(strict=True)==x,'USE_CANONICAL_READONLY_MOUNTS')
    native=manifest(source,a.manifest_sha256)
    if a.verify_only:
        return dict(passed=True,verified_objects=79,verified_bytes=audit_assets(a.assets,native),actual_native_pairs=0,
                    planned_common_pairs=20,model_weights_opened=0,actual_predictions=0,actual_GPU=0)
    need(os.environ.get('SLURM_JOB_ID')=='1103606' and re.fullmatch(r'[0-9]+',os.environ.get('SLURM_STEP_ID','')) and
         a.step=='1103606.'+os.environ['SLURM_STEP_ID'] and not os.environ.get('SLURM_STEP_GPUS')
         and all(os.environ.get(k)=='' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')),
         'EXACT_ONE_NUMERIC_CPU_STEP_WITH_NO_DEVICE_VISIBILITY')
    for x in (a.assets,source,BACKEND):
        need(os.statvfs(x).f_flag&os.ST_RDONLY,'READONLY_EFFECTIVE_MOUNTS')
    audit_assets(a.assets,native)
    helper={'__name__':'exact_original_tum_pair_private'}
    exec(compile(find(source/'tum_vggt_pair.py',EXPECTED_MATH),'exact_tum_geometry','exec'),helper)
    find(BACKEND,EXPECTED_BACKEND)
    import torch
    need(torch.version.hip=='6.3.26113' and importlib.metadata.version('torch')==
         '2.10.0+das.opt1.dtk2604.2606161421.g91a580' and torch.get_default_dtype()==torch.float32
         and not torch.cuda.is_initialized(),'EXACT_NATIVE_CPU_TORCH')
    torch.set_num_threads(4)
    from gencuda.backends import vggt_training as backend
    need(Path(backend.__file__)==BACKEND,'ORIGINAL_VGGT_BACKEND_IS_SOURCE')
    result=capture(torch,backend,helper,a.assets,native)
    need(not torch.cuda.is_initialized(),'NO_GPU_INITIALIZATION')
    return dict(result,numeric_step=a.step,input_manifest_SHA256=a.manifest_sha256,backend_SHA256=EXPECTED_BACKEND)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--assets',type=Path,required=True)
    cli.add_argument('--manifest-sha256',required=True)
    cli.add_argument('--step')
    cli.add_argument('--verify-only',action='store_true')
    result=run(cli.parse_args())
    print('VGGT_HOUSEHOLD_NATIVE_CPU '+json.dumps(result,sort_keys=True),flush=True)
    raise SystemExit(0 if result['passed'] else 2)
