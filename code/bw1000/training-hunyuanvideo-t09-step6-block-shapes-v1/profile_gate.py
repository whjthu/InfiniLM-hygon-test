"""Pin the real device profiler and reduce the finite T09 prefix evidence."""

from hashlib import sha256
import importlib
from importlib.abc import MetaPathFinder,Loader
from importlib.util import spec_from_file_location
import inspect
import json
from math import isfinite
from pathlib import Path
import sys

from contract import FRAMEWORK, LABEL, NODE, REQUEST, RUN, load, need, read, save


RUN=Path(RUN)
CPU_SCOPE=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/t09-backward67-overlay-cpu-d2b34ae9bc52')
ORIGINAL_3640=RUN.parent/'training-hunyuanvideo-t09-sft-repeat2-ro393-miopen-untraced-v3/outcome.json'
CPU_RECEIPT_SHA='c9f071765f0085e141d6e7b7fee5fa6252a3f72d42c36b485d196bbb2b3d6fc2'
CPU_STDOUT_SHA='43c3da70440ca520eb8b698fe2689f0dae5a7bacd52b54c5af7d45b915e9519c'
PROFILER=('profiler/profiler.py',47745,'101a1d2d9fc0b27b87e36537ec01a04a98cb6896a41b7f6c1c14b2f0c16f4042')
API={'profile':PROFILER,'supported_activities':PROFILER,
     'record_function':('autograd/profiler.py',50143,'b44b1e13f8ff8fb89df6ff5388e544738bca007f725f895d095a61778a6dfe10')}
BACKEND_MODULE='gencuda.backends.hunyuanvideo_training_walltime'
OBSERVER_MODULE='gencuda.backends.t09_backward67_observer'
# The pinned hashes here are the accepted CPU overlay. The finite backend is separately manifest-bound.
CPU_IMPORTS={BACKEND_MODULE:dict(SHA256='4eb03a95f76d0affb74f2589afced199d991c894299b36c8c399c09a8ea70d2e',bytes=31974),
             OBSERVER_MODULE:dict(SHA256='4cbea11835e72de19d222e8953c1be78e56f7286db10db20532d0a2455ae79e0',bytes=7889)}
NEW_IMPORTS={BACKEND_MODULE:{'bytes': 33053, 'SHA256': '5b3542994c47b4a190981058f0e8f531532515033e2251dd54a3143c2031560a'},
             OBSERVER_MODULE:{'bytes': 26110, 'SHA256': '9c226434cec0836289c3dfe0a77628d030d0b8f15bc733e18063df602eacee4a'}}
ALIASES={BACKEND_MODULE:Path('/t09-profile-backend.py'),OBSERVER_MODULE:Path('/t09-profile-observer.py')}
NAMES={BACKEND_MODULE:'backward_backend.py',OBSERVER_MODULE:'backward_observer.py'}
MAX_TRACE=256<<20


def previous_3640():
    value,pin=load(ORIGINAL_3640,32768)
    need((pin['bytes'],pin['SHA256'])==(22283,'e712324838d1a34cf9802f7654cc54a22bef005ea5fda566c5d55b68de6dcdff')
         and value['request_id']=='f911ca35708844559588ee70ecd88912' and value['passed'] is True
         and value['release_permitted'] is True and value['completed_updates']==25
         and {k:v['row'][0] for k,v in value['receipts'].items()}==
             {'pre':'1103606.3639','gpu':'1103606.3640','post':'1103606.3641'},'EXACT_ORIGINAL_3640_AND_3641_PHYSICAL_RELEASE')
    return pin


def cpu_prerequisite():
    raw,pin=read(CPU_SCOPE/'private.stdout',1<<20)
    need(pin['SHA256']==CPU_STDOUT_SHA,'EXACT_3660_CPU_RAW_RESULT')
    mark=b'T09_OVERLAY_CPU_RESULT '
    results=[json.loads(line[len(mark):]) for line in raw.splitlines() if line.startswith(mark)]
    need(len(results)==1,'ONE_EXACT_3660_PERSISTED_CPU_RESULT')
    outcome=results[0]
    need(outcome['step']=='1103606.3660' and outcome['passed'] is True
         and outcome['framework_before']==outcome['framework_after']==393
         and outcome['miopen_same_readonly_mapped_fd'] is True and outcome['imports']==CPU_IMPORTS
         and outcome['profiler']['activity_names']==['CPU'],'ORIGINAL_CPU_IMPORTS_ONLY_NOT_GPU_CAPABILITY')
    return dict(actual_CPU_step=outcome['step'],original_authenticated_receipt_SHA256=CPU_RECEIPT_SHA,
                original_stdout_SHA256=CPU_STDOUT_SHA)


def check_api(torch):
    root=Path(torch.__file__).resolve(strict=True).parent
    p=torch.profiler
    for key,(relative,size,sha) in API.items():
        fn=getattr(p,key)
        path=Path(inspect.getsourcefile(fn)).resolve(strict=True)
        _,pin=read(path,size)
        need(path==root/relative and (pin['bytes'],pin['SHA256'])==(size,sha) and fn.__qualname__==key,
             'SAME_3660_TORCH_PROFILER_CALLABLE_SOURCE')
    need({'activities','record_shapes','profile_memory','with_stack','with_flops'}<=set(inspect.signature(p.profile).parameters)
         and callable(getattr(p.profile,'export_chrome_trace',None)), 'EXACT_NATIVE_KINETO_PROFILER_API')
    activities=p.supported_activities()
    need({p.ProfilerActivity.CPU,p.ProfilerActivity.CUDA}<=activities and torch.cuda.device_count()==1,
         'CURRENT_ASSIGNED_GPU_KINETO_ACTIVITY_NOT_AVAILABLE')
    return dict(activity_names=sorted(a.name for a in activities),API=API,actual_native_kernels_yet_verified=False)


class TwoFiles(MetaPathFinder,Loader):
    def __init__(self,bytes_by_name): self.blobs=bytes_by_name
    def find_spec(self,fullname,path=None,target=None):
        if fullname in self.blobs:
            need(target is None,'NO_PARALLEL_OVERLAY_RELOAD')
            return spec_from_file_location(fullname,ALIASES[fullname],loader=self)
    def create_module(self,spec): return None
    def exec_module(self,module):
        name=module.__name__
        exec(compile(self.blobs[name],str(ALIASES[name]),'exec'),module.__dict__)


def admit_imports():
    from mount_observation import explicit_readonly,observe
    need(not any(name in sys.modules for name in ALIASES),'NO_PREEXISTING_PROFILED_BACKENDS')
    mf,_=load(RUN/'src/SOURCE_FILES.json')
    pins,blobs={},{}
    for name,path in ALIASES.items():
        item=mf['files'][NAMES[name]]
        original=RUN/'src'/NAMES[name]
        evidence=observe(path)
        raw,pin=read(path,65536)
        need(explicit_readonly(evidence) and evidence['object_type']=='REGULAR'
             and evidence['dev_inode']==[original.stat().st_dev,original.stat().st_ino]
             and pin['bytes']==item['bytes'] and pin['SHA256']==item['SHA256'],'EXACT_GPU_TWO_FILE_OVERLAY_ASSIGNMENT')
        blobs[name]=raw
        pins[name]=item
    need(pins==NEW_IMPORTS,'EXACT_NEW_BOUND_CHECKPOINT_AND_TRACE_OBSERVER')
    finder=TwoFiles(blobs)
    sys.path.insert(0,str(Path(FRAMEWORK)/'generation/src'))
    sys.meta_path.insert(0,finder)
    try:
        modules=[importlib.import_module(name) for name in ALIASES]
        need(all(Path(mod.__file__)==ALIASES[mod.__name__] for mod in modules),'EXACT_ACTUAL_GPU_OVERLAY_IMPORT_ORIGINS')
    finally:
        sys.meta_path.remove(finder)
    return pins


class RequestedPrefixComplete(BaseException):
    pass


def stop_after(output):
    need(Path(output)==RUN/'outputs/hunyuanvideo-1.5/train'/LABEL,'EXACT_PRIVATE_DIAGNOSTIC_OUTPUT')
    raise RequestedPrefixComplete()


def project():
    output=RUN/'outputs/hunyuanvideo-1.5/train'/LABEL
    answer=dict(planned_updates=6,completed_updates=0,missing_updates=6,completed_warmup=0,profiled_steps=[],
        planned_profiled_steps=[6],prefix_stop_verified=False,device_attribution_qualified=False,
        recomputation_device_time='UNKNOWN',formal_performance=False,no_checkpoint_deliverable=True)
    try:
        step_log,log_pin=read(output/'steps.ndjson',65536)
        records=[json.loads(line) for line in step_log.splitlines()]
        for pos,row in enumerate(records,1):
            need(row['global_step']==pos<=6 and isfinite(row['loss']) and row['loss']>=0
                 and isfinite(row['grad_norm']) and row['grad_norm']>0,'EXACT_NATIVE_PREFIX_STEP_ORDER_AND_NUMERICS')
            answer.update(completed_updates=pos,missing_updates=6-pos,completed_warmup=min(5,pos))
        answer['step_log_SHA256']=log_pin['SHA256']
        for step in (6,):
            folder=output/'t09-backward67'/f'step-{step}'
            try:
                observation,_=load(folder/'summary.json',65536)
                raw,pin=read(folder/'backward.trace.json',MAX_TRACE)
            except FileNotFoundError:
                continue
            # Execute only the independently pinned reducer over original raw trace bytes.
            code,src=read(RUN/'src/backward_observer.py',65536)
            need({k:src[k] for k in ('bytes','SHA256')}==NEW_IMPORTS[OBSERVER_MODULE],'EXACT_READONLY_REDUCER')
            ns={'__name__':'t09_independent_backward_reducer'}
            exec(compile(code,'t09_independent_backward_reducer','exec'),ns)
            payload=json.loads(raw)
            reduced=ns['reduce_trace'](payload,step)
            blocks,_=load(folder/'blocks.json',65536)
            reduced['step6_blocks']=ns['block_trace'](payload,step,blocks)
            memory,_=load(folder/'capacity.json',16384)
            need(observation==dict(reduced,private_trace_bytes=pin['bytes'],private_trace_SHA256=pin['SHA256']),
                 'EXACT_ORIGINAL_PER_STEP_PROFILER_REDUCTION')
            providers,_=load(folder/'provider.json',65536)
            expected={k:dict(relative_path=v[0],bytes=v[1],SHA256=v[2]) for k,v in API.items()}
            need(providers['providers']==expected and {'CPU','CUDA'}<=set(providers['supported_activities']),
                 'SAME_3660_PROVIDER_ACTUAL_GPU_ACTIVITY_PER_STEP')
            answer['profiled_steps'].append(dict(step=step,raw_trace_SHA256=pin['SHA256'],raw_trace_bytes=pin['bytes'],
                **{k:observation[k] for k in ('hardware_attribution','cpu_backward_marker_us','observed_activity_counts',
                    'kernel_correlation_counts','kernel_union_device_us','linked_kernel_union_device_us','bmm_recomputation','step6_blocks')},capacity=memory))
        stop,_=load(RUN/'prefix-stop.json',16384)
        ingress,_=load(RUN/'profile-ingress.json',16384)
        answer['prefix_stop_verified']=(answer['completed_updates']==6 and stop['request']==REQUEST
            and ingress['request']==REQUEST and ingress['step']==stop['step'] and ingress['epoch']==stop['epoch']
            and stop['last_update']==6 and stop['no_step7_or_checkpoint_entered'] is True
            and stop['original_framework_count_after']==393
            and not any(output.glob('checkpoint-*')))
        answer['device_attribution_qualified']=(answer['prefix_stop_verified'] and len(answer['profiled_steps'])==1
             and all(r['hardware_attribution']=='OBSERVED_UNIQUE_BACKWARD_EXTERNAL_IDS' for r in answer['profiled_steps']))
    except (OSError,ValueError,TypeError,KeyError) as err:
        answer['failure_type']=type(err).__name__
    answer['bmm_qualified_steps']=[r['step'] for r in answer['profiled_steps']
         if r['bmm_recomputation']['attribution_complete'] and answer['device_attribution_qualified']]
    return answer
