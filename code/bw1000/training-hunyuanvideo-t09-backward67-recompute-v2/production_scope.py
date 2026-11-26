"""Untraced T09 ingress with the original CLI and a warm-only native forward witness."""
from hashlib import sha256
import os
from pathlib import Path
import runpy
import stat
import sys
import time

from contract import CACHE, CONFIG, FRAMEWORK, FRAMEWORK_MANIFEST, NATIVE, NATIVE_ALIAS, REQUEST, ROOT, RUN, WORK, load, need, read, save
from flash_compat import ALIAS
from framework_mount import STAGE_ALIAS
from miopen_wall import HOME, LIB, actual_process_import, maps_pin, readonly_failure, verify_mapping
from mount_evidence import mount_rows, read_proc
from profile_gate import (ALIASES, BACKEND_MODULE, RequestedPrefixComplete,
                          admit_imports, check_api, cpu_prerequisite)
from framework_binding import verify_framework
from mount_observation import explicit_readonly, observe, shared_readonly, strict_private_writable


RUN=Path(RUN)
ROOT=Path(ROOT)
HOME_TARGET=Path('/miopen-private-home-v4')
CLI=Path(FRAMEWORK)/'tools/hygon_training_recipe.py'
BACKEND=Path(FRAMEWORK)/'generation/src/gencuda/backends/hunyuanvideo_training_walltime.py'
BACKEND_SHA='2331d0fa942eb3908dc5392df6008355524f013a0fbdb5067202f9a74400ccae'
MODEL_CODE=Path(NATIVE_ALIAS)/'hyvideo/models/transformers/hunyuanvideo_1_5_transformer.py'
CANONICAL_MODEL_CODE=Path(NATIVE)/'hyvideo/models/transformers/hunyuanvideo_1_5_transformer.py'
MODEL_CODE_SHA='d1a702891a92e04c80cdc072bc06cffe6e631fb962dae702d3c9244d347302d0'
TORCH_VERSION=Path('/usr/local/lib/python3.10/dist-packages/torch/version.py')
TORCH_VERSION_SHA='d324ffd50535bca68131043b76605e36d588a85ec41b2aed9d798ece471aa534'
PLAN=7
SHARED={
    'project':ROOT,'runtime':Path('/training-runtime'),'native':Path(NATIVE_ALIAS),'compatibility':Path(ALIAS),
    'framework':Path(FRAMEWORK),'manifest':Path(FRAMEWORK_MANIFEST),'framework_stage_alias':Path(STAGE_ALIAS),
    'source':RUN/'src','hyhal':Path('/opt/hyhal'),'miopen':LIB,
    **ALIASES,
}
PRIVATE={
    'run':RUN,'home':HOME_TARGET,'tmp':Path('/tmp'),'var_tmp':Path('/var/tmp'),
}


def digest(raw):return sha256(raw).hexdigest()


def identify(path):
    info=Path(path).stat()
    return [info.st_dev,info.st_ino]


def clean_asset_mounts(rows,observer=observe):
    """Reject any deeper writable mount within an approved read-only namespace."""
    offenders=[]
    for row in rows:
        point=Path(row['mountpoint'])
        if 'rw' not in row['options']:
            continue
        if any(point==mount or point.is_relative_to(mount) for mount in SHARED.values()) and not (
                point in PRIVATE.values() and point==RUN):
            if not explicit_readonly(observer(point)):
                offenders.append(digest(str(point).encode()))
    return offenders


def layout_witness(host):
    from contract import WEIGHT

    ro={label:observe(path) for label,path in SHARED.items()}
    rw={label:observe(path) for label,path in PRIVATE.items()}
    input_files={label:observe(Path(path)) for label,path in
                 (('weight',WEIGHT),('training_cache',CACHE),('frozen_model_config',CONFIG))}
    nested=clean_asset_mounts(mount_rows(read_proc('/proc/self/mountinfo')))
    expected=host['writable_dev_inode']
    def system_image_readonly(path):
        obs=observe(path)
        return ((obs.get('metadata_status')=='RECORDED' and obs.get('same_before_open_after_dev_inode') is True
                 and obs['three_reads'][2]=='RO' and 'RO' in obs['three_reads'][:2]) or obs.get('present') is False)
    system={'share_db':system_image_readonly(Path('/opt/dtk/share/miopen/db')),
            'alternative_lib_db':system_image_readonly(Path('/opt/dtk/lib/miopen/db'))}
    result=dict(explicit_readonly={d:explicit_readonly(v) for d,v in ro.items()},
        effective_readonly_assets={d:shared_readonly(v) for d,v in input_files.items()},
        exact_private_writes={label:strict_private_writable(v,os.getuid()) and v['dev_inode']==expected[label]
                              for label,v in rw.items()},
        readonly_or_absent_image_system_DB=system,writable_nested_shared_mounts=nested,
        checked_shared_binds=len(ro),checked_weight_cache_config=len(input_files))
    result['passed']=(all(result['explicit_readonly'].values()) and all(result['effective_readonly_assets'].values())
                      and all(result['exact_private_writes'].values()) and all(system.values()) and not nested)
    return result


def inventory(root):
    """Observe metadata under only this scope's three MIOpen private leaves."""
    results={}
    def report_error(error):raise error
    for label,relative in (('db','.config/miopen/user-db'),('lock','.config/miopen/miopen-lockfiles'),('cache','.cache/miopen')):
        folder=root/relative
        need(folder.resolve(strict=True)==folder and folder.is_dir(),'EXACT_PRIVATE_MIOPEN_STATE_SUBTREE')
        nodes=bytes_sum=links=0
        for directory,names,files in os.walk(folder,followlinks=False,onerror=report_error):
            for name in names+files:
                nodes+=1
                need(nodes<=512,'BOUNDED_OWN_MIOPEN_STATE_TREE')
                st=(Path(directory)/name).lstat()
                links+=stat.S_ISLNK(st.st_mode)
                if stat.S_ISREG(st.st_mode):
                    bytes_sum+=st.st_size
        results[label]=dict(entries=nodes,regular_file_bytes=bytes_sum,symlinks=links)
    return results


def source_frame(frame):
    """Find the exact original train_step on the live Python call stack."""
    while frame is not None:
        if frame.f_code.co_name=='train_step' and Path(frame.f_code.co_filename)==ALIASES[BACKEND_MODULE]:
            return frame
        frame=frame.f_back
    return None


def watch_first_forward(torch,ctx,provider):
    retired=False
    def capture(module,args,output):
        nonlocal retired
        code=getattr(getattr(type(module),'forward',None),'__code__',None)
        if code is None or Path(code.co_filename) not in (MODEL_CODE,CANONICAL_MODEL_CODE):
            return
        frame=source_frame(sys._getframe(1))
        if frame is None or not hasattr(module,'img_in'):
            return
        step=frame.f_locals.get('global_step')
        if retired or type(step) is not int or step<1:
            return
        retired=True
        handle.remove()
        if step!=1:
            return
        live=maps_pin()
        verify_mapping(provider['maps_before'],live,provider['bound_map_identity'])
        save(RUN/'production-first-forward.json',dict(ctx,global_step=step,process_PID=os.getpid(),
             original_model_class=type(module).__name__,backend_SHA256=BACKEND_SHA,
             finite_backend_SHA256='dd15ec0ea7e3ad83500784e2d74bbf1d78a6a0c9efa70d7ab621148da46dd130', model_source_SHA256=MODEL_CODE_SHA,native_forward_returned=True,extra_device_sync=False,
             warm_only_hook_removed=True,trace_or_profiler=False,first_forward_map=live))
    handle=torch.nn.modules.module.register_module_forward_hook(capture)
    return handle


def record_partial_error(ctx,error):
    text=str(error)
    data=dict(ctx,type=type(error).__name__,errno=getattr(error,'errno',None),
              message_SHA256=digest(text.encode('utf8','replace')),readonly_filesystem=readonly_failure(error),
              actual_native_forward_witness=(RUN/'production-first-forward.json').is_file(),planned_updates=PLAN)
    save(RUN/'production-first-error.json',data)


def self_untraced():
    lines=Path('/proc/self/status').read_text().splitlines()
    return [line.split()[1] for line in lines if line.startswith('TracerPid:')]==['0']


def run(cli_argv):
    need(cli_argv[0]==str(CLI) and cli_argv[1]=='T09.sft' and '--run' in cli_argv,
         'PRESERVED_ORIGINAL_T09_COMMON_CLI')
    gate,gate_pin=load(RUN/'production-host-private-bind.json')
    step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','')
    ctx=dict(request=REQUEST,epoch=gate['epoch'],step=step,host_bind=gate_pin)
    handle=None
    before=None
    try:
        need(gate['request']==REQUEST and gate['step']==step and gate['deadline']==gate['epoch']+WORK
             and 0<gate['deadline']-time.time()<=WORK and bool(os.environ.get('CUDA_VISIBLE_DEVICES'))
             and sys.dont_write_bytecode and sys.pycache_prefix is None and self_untraced()
             and all(os.environ.get(key)==str(PRIVATE['tmp']) for key in ('TMPDIR','TEMP','TMP')),
             'SAME_GPU_STEP_PRIVATE_WRITE_NO_TRACER_AND_DEADLINE')
        mounts=layout_witness(gate)
        save(RUN/'production-mount-ingress.json',dict(ctx,**mounts))
        need(mounts['passed'],'T09_EXACT_ALL_SHARED_RO_AND_PRIVATE_WRITABLE')
        before=inventory(HOME)
        need(all(value['entries']==0 for value in before.values()),'NEW_PRIVATE_MIOPEN_STATE_MUST_BEGIN_EMPTY')
        torch,provider=actual_process_import(gate,gate['deadline'])
        _,version=read(TORCH_VERSION,65536)
        need(Path(torch.__file__).resolve(strict=True)==TORCH_VERSION.parent/'__init__.py' and
             torch.__version__=='2.10.0' and torch.version.hip=='6.3.26113' and version['SHA256']==TORCH_VERSION_SHA,
             'EXACT_3600_ORIGINAL_TORCH_PROVIDER')
        provider.update(TracerPid=0,native_torch_version='2.10.0',native_hip_version='6.3.26113',
                        torch_version_file_SHA256=TORCH_VERSION_SHA)
        save(RUN/'production-miopen-ingress.json',dict(ctx,**provider))
        for path,pin in ((BACKEND,BACKEND_SHA),(MODEL_CODE,MODEL_CODE_SHA)):
            _,actual=read(path,1<<20)
            need(actual['SHA256']==pin,'T09_EXACT_NATIVE_FORWARD_SOURCE')
        api=check_api(torch)
        imports=admit_imports()
        save(RUN/'profile-ingress.json',dict(ctx,API=api,imports=imports,CPU=cpu_prerequisite()))
        handle=watch_first_forward(torch,ctx,provider)
        sys.argv=list(cli_argv)
        sys.path[0]=str(CLI.parent)
        runpy.run_path(str(CLI),run_name='__main__')
        raise ValueError('NATIVE_DIAGNOSTIC_DID_NOT_STOP_AT_SEVEN')
    except RequestedPrefixComplete:
        unchanged=verify_framework(Path(FRAMEWORK),Path(FRAMEWORK_MANIFEST))
        save(RUN/'prefix-stop.json',dict(ctx,last_update=7,no_step8_or_checkpoint_entered=True,
             original_framework_count_after=unchanged['framework_files']))
    except SystemExit as error:
        record_partial_error(ctx,error)
        if error.code in (None,0):
            raise RuntimeError('NATIVE_DIAGNOSTIC_STOP_SENTINEL_REQUIRED') from error
        raise
    except BaseException as error:
        record_partial_error(ctx,error)
        raise
    finally:
        if handle is not None:
            handle.remove()
        if before is not None:
            state=dict(ctx,before=before,proven_native_filesystem_syscalls=False,historical_warm_state='UNKNOWN')
            try:state['after']=inventory(HOME)
            except (OSError,ValueError) as error:state['after_error']=type(error).__name__
            save(RUN/'production-private-state.json',state)


def read_production_evidence(work,step,epoch):
    state=dict(passed=False,planned_updates=7,first_forward=None,private_syscall_attribution=False,
               historical_warm_state='UNKNOWN',profile_scope=[6,7],formal_performance=False)
    files={label:work/('production-'+label+'.json') for label in
           ('host-private-bind','mount-ingress','miopen-ingress','first-forward','private-state','first-error')}
    data={}
    phase='production-records'
    try:
        for name,path in files.items():
            try:body,pin=load(path,65536)
            except FileNotFoundError:continue
            need(body['request']==REQUEST and body['step']==step and body['epoch']==epoch,
                 'T09_INDEPENDENT_PRODUCTION_WITNESS_SCOPE')
            data[name]=body
            state[name.replace('-','_')+'_pin']=pin
        phase='production-gates'
        host,mount,miopen,forward,private=(data[key] for key in
                ('host-private-bind','mount-ingress','miopen-ingress','first-forward','private-state'))
        need(mount['passed'] and miopen['mapped_exact_same_FD_verified'] and miopen['provider']==host['provider']
             and miopen['provider_dev_inode']==host['provider_dev_inode'] and miopen['home_dev_inode']==host['home_dev_inode']
             and forward['process_PID']==miopen['process_pid'] and forward['global_step']==1
             and forward['native_forward_returned'] and forward['warm_only_hook_removed'] and not forward['extra_device_sync']
             and forward['backend_SHA256']==BACKEND_SHA and forward['finite_backend_SHA256']=='dd15ec0ea7e3ad83500784e2d74bbf1d78a6a0c9efa70d7ab621148da46dd130'
             and forward['model_source_SHA256']==MODEL_CODE_SHA
             and private['proven_native_filesystem_syscalls'] is False and private['historical_warm_state']=='UNKNOWN',
             'EXACT_SAME_PROCESS_UNTRACED_PRODUCTION_CHAIN')
        verify_mapping(miopen['maps_before'],miopen['maps_after'],miopen['bound_map_identity'])
        verify_mapping(miopen['maps_before'],forward['first_forward_map'],miopen['bound_map_identity'])
        need(miopen['TracerPid']==0 and miopen['torch_version_file_SHA256']==TORCH_VERSION_SHA,
             'EXACT_INDEPENDENT_UNTRACED_TORCH_PROVIDER')
        state.update(passed='first-error' not in data,first_forward=True,
                     scoped_private_state_changes=private.get('after'))
    except (OSError,ValueError,KeyError,TypeError) as error:
        state.update(failed_projection_phase=phase,error_type=type(error).__name__)
    return state


if __name__=='__main__':
    run(sys.argv[1:])
