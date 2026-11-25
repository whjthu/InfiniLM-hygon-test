"""One original production Torch import and bounded in-process library mapping readback."""
import argparse
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import re
import sys
import time

from maps_probe import MAX_MAPS, inspect_maps, sha
from map_files_probe import observe as read_map_files

REQUEST='b90cb0007e68492ab99fd131a30677f9'
PARENT='1103606'
TORCH_ROOT='/usr/local/lib/python3.10/dist-packages/torch'
VERSION_SOURCE_SHA='d324ffd50535bca68131043b76605e36d588a85ec41b2aed9d798ece471aa534'


def need(ok,code):
    if not ok:raise ValueError(code)


def read_maps():
    with Path('/proc/self/maps').open('rb') as stream:
        raw=stream.read(MAX_MAPS+1)
    need(0<len(raw)<=MAX_MAPS,'BOUNDED_CURRENT_IMPORTED_PROCESS_MAPS')
    return raw


def source_spec():
    spec=PathFinder.find_spec('torch',sys.path)
    need(spec is not None and spec.origin is not None,'ACTUAL_PRODUCTION_TORCH_RESOLUTION')
    path=Path(spec.origin).resolve(strict=True)
    version=path.parent/'version.py'
    need(path==Path(TORCH_ROOT)/'__init__.py' and version.resolve(strict=True)==version
         and 'torch' not in sys.modules,'EXACT_NOT_YET_IMPORTED_PRODUCTION_TORCH_PATH')
    with version.open('rb') as stream:
        raw=stream.read(65537)
    need(0<len(raw)<=65536 and sha(raw)==VERSION_SOURCE_SHA,'ORIGINAL_PRODUCTION_TORCH_VERSION_SOURCE_SHA')
    return path


def observe(report,deadline):
    path=source_spec()
    original,malformed,related=inspect_maps(read_maps())
    report['before_import']=dict(provider_groups=len(original),malformed_provider_segments=len(malformed))
    report['phase']='EXECUTE_ORIGINAL_TORCH_IMPORT'
    # No torch.cuda or other device APIs are called by this diagnostic.
    import torch
    report['torch_imported']=True
    need(Path(torch.__file__).resolve(strict=True)==path and torch.__version__=='2.10.0'
         and torch.version.hip=='6.3.26113' and torch.version.cuda is None,'ACTUAL_IMPORTED_TORCH_IDENTITY_DIFFERS')
    report['provider']=dict(torch_version=torch.__version__,hip_version=torch.version.hip,
                            production_path=TORCH_ROOT,version_source_SHA256=VERSION_SOURCE_SHA)
    report['phase']='OBSERVE_EXACT_EXECUTABLE_PROC_SELF_MAP_FILES_LINK'
    limit=time.monotonic()+max(0,deadline-time.time())
    observed=read_map_files(read_maps,limit)
    report['after_import']=observed
    opened=observed['map_files']
    status=opened['status'] if opened is not None else observed['status']
    report.update(phase='COMPLETE',passed=True,backing_qualified=bool(opened and opened['matched_pinned_SHA256']),
                  mapping_outcome=status,actual_miopen_call_trace='UNOBSERVED_NO_OPERATOR_CALL',
                  original_3587_mapping_backfilled=False,canonical_path_hash_fallback_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args()
    report=dict(request=REQUEST,phase='ENTRY',passed=False,torch_imported=False,library_actively_loaded_by_probe=False,
        explicit_GPU_SDK_device_calls=0,actual_GPU_allocated=0,actual_models_loaded=0,actual_operators_called=0,
        actual_inference_or_training_calls=0,current_scope_can_qualify_prior_GPU_scope=False)
    try:
        step=os.environ.get('SLURM_STEP_ID','')
        need(sys.dont_write_bytecode and sys.pycache_prefix is None and os.environ.get('SLURM_JOB_ID')==PARENT
             and step.isdecimal() and 0<args.deadline-time.time()<=120
             and all(os.environ.get(key)=='' for key in ('ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES')),
             'ADMITTED_ZERO_GRES_IMPORT_ONLY_RUNTIME')
        report['step']=PARENT+'.'+step
        observe(report,args.deadline)
    except (OSError,ImportError,ValueError,RuntimeError,TypeError,KeyError) as error:
        reason=str(error)
        report.update(error_type=type(error).__name__,errno=getattr(error,'errno',None),
                      error_code=reason if re.fullmatch(r'[A-Z0-9_]{1,96}',reason) else None)
    raw=json.dumps(report,sort_keys=True,allow_nan=False)
    need(len(raw.encode())<=128*1024,'BOUNDED_PUBLIC_SAFE_IMPORT_OBSERVATION')
    print(raw,flush=True)
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
