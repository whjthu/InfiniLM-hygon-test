"""Container-only native8 MiniCPM25+5 worker. HOST owns Slurm, admission and post.

No network, data export, new training loop or target invocation during source tests.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time

LABEL='training-minicpm-usgs-exposure25-resume20-target-v2'
REQUEST='fa68d03ea04f4588ac9c21459097d50e'
RUN=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/LABEL
PARENT,NODE,PORT='1103606','f13r1n19',29929
CORE_SHA='dcf5356c2879bb2c61b1de1f0f037d8fd312542775e46e6556c6f6d89dc0ee82'
ASSETS_SHA='16bd66c3df682fcaa3ea6cc0f66739c968b819a37d6511e97a50cd7ab03f83e7'
BASE_MANIFEST_SHA='e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41'
WHOLE,NATIVE,CLEANUP=2400,1980,2080
CHECKPOINT_NAMES={'trainer_state.json','args.json','adapter_model.safetensors','optimizer.pt','scheduler.pt','adapter_config.json'}|{'rng_state_%d.pth'%i for i in range(8)}


def need(ok,message):
    if not ok:raise ValueError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def save(path,value):
    raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    with path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())


def read(path,cap=8*1024**2):
    with path.open('rb') as stream:raw=stream.read(cap+1)
    need(0<len(raw)<=cap,'bounded metadata exceeds limit')
    return json.loads(raw)


def file_pin(path,cap=256*1024**2):
    before=path.lstat()
    need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=cap,'plain bounded file required')
    h=hashlib.sha256();total=0
    with path.open('rb') as stream:
        for raw in iter(lambda:stream.read(1024**2),b''):h.update(raw);total+=len(raw)
    after=path.lstat()
    need(total==before.st_size and all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'file changed during read')
    return dict(bytes=total,SHA256=h.hexdigest())


def checked_record(record,root,cap=8*1024**2):
    path=Path(record['path'])
    need(path.is_absolute() and '..' not in path.parts and path.is_relative_to(root),'metadata path escapes this scope')
    need(file_pin(path,cap)=={k:record[k] for k in ('bytes','SHA256')},'pinned metadata identity differs')
    return read(path,cap)


class Budget:
    def __init__(self,epoch,deadline,native_deadline,wall=time.time,monotonic=time.monotonic):
        self.wall,self.mono=wall,monotonic
        now=wall()
        need(all(type(v) in (int,float) and math.isfinite(v) for v in (epoch,deadline,native_deadline,now)),'nonfinite/boolean epoch')
        need(0<epoch<=now<native_deadline and deadline==epoch+WHOLE and native_deadline==epoch+NATIVE,'original worker epoch differs/exhausted')
        self.epoch,self.started,self.elapsed=epoch,monotonic(),now-epoch
    def left(self,offset=NATIVE,reserve=0):
        value=min(self.epoch+offset-self.wall(),offset-self.elapsed-(self.mono()-self.started))-reserve
        need(math.isfinite(value) and value>0,'shared original worker budget exhausted')
        return value


def check_config(config,now):
    need(config['schema']=='MINICPM_USGS_TARGET_WORKER_CONFIG_V1' and config['request']==REQUEST,'worker context differs')
    need(config['run_root']==str(RUN) and config['core_directory']==str(RUN/'src') and config['framework_directory']==str(RUN/'framework'),'isolated source root differs')
    need(config['parent']==PARENT and config['node']==NODE and config['port']==PORT,'resource recipe differs')
    need(all(config.get(k) is True for k in ('root_GO','training_GO','GPU_GO')),'worker not adopted')
    need(config['core_manifest_SHA256']==CORE_SHA,'frozen MiniCPM core manifest differs')
    Budget(config['epoch'],config['deadline'],config['native_deadline'],wall=lambda:now,monotonic=lambda:0)


def check_snapshots(before,current,cpus):
    need(before['node']==current['node']==NODE and before['boot_id']==current['boot_id'],'snapshot host/boot differs')
    need(before['full_fd_inventory_performed'] is True and before['holder_inventory_basis']=='kernel_KFD_plus_sameUID_FD_complete','complete HOST before ownership required')
    need(current['full_fd_inventory_performed'] is False and current['holder_inventory_basis']=='kernel_KFD_proc_only','container ownership scope differs')
    need(before['kfd_holder_count']==current['kfd_holder_count']==0,'KFD holder observed')
    need(len(current['pci'])==len(set(current['pci']))==8 and current['pci']==before['pci'],'eight original physical PCI differ')
    need(0<=current['monotonic_ns']-before['monotonic_ns']<=45_000_000_000 and cpus>=64,'stale snapshot or CPU allocation differs')


def normalized_expected(plan,run):
    # The core translator intentionally names the deployed base entry; HOST fixes only it.
    command=plan['command']
    old='/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-a68952c-swift-initial-CPU-v1/framework/vlm/runtime/swift_entry.py'
    need(command[0]==old,'unexpected core original common entry')
    command[0]=str(run/'framework/vlm/runtime/swift_entry.py')
    return plan


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def source_and_data(config):
    stage=checked_record(config['asset_stage'],RUN)
    need(stage['state']=='MINICPM_PAIR_TARGET_ASSETS_STAGED' and stage['request']==REQUEST
         and stage['parent']==PARENT and stage['node']==NODE and stage['port']==PORT,'HOST asset stage scope differs')
    need(stage['source_rows']==88 and stage['train_media']==22,'HOST staged data denominator differs')
    # Staging has its own completed CPU scope; it does not claim a GPU epoch.
    need(file_pin(RUN/'src/SOURCE_FILES.json',2*1024**2)['SHA256']==CORE_SHA,'actual core manifest differs')
    core=read(RUN/'src/SOURCE_FILES.json')['files']
    for name,pin in core.items():
        need(Path(name).name==name and file_pin(RUN/'src'/name,2*1024**2)==pin,'core source member differs')
    assets=checked_record(config['asset_binding'],RUN)
    need(config['asset_binding']['path']==str(RUN/'src/ASSETS.json') and config['asset_binding']['SHA256']==ASSETS_SHA,'original asset binding differs')
    native_record=config['native88_independent'];native_path=Path(native_record['path'])
    need(native_path.is_absolute() and '..' not in native_path.parts,'exact native88 independent path required')
    need(file_pin(native_path,2*1024**2)=={k:native_record[k] for k in ('bytes','SHA256')},'native88 independent pin differs')
    with native_path.open('rb') as stream:native_raw=stream.read(2*1024**2+1)
    need(len(native_raw)<=2*1024**2,'native88 record cap')
    sys.path.insert(0,str(RUN/'src'))
    recipe=load_module('minicpm_worker_recipe',RUN/'src/minicpm_domain_recipe.py')
    cpu=recipe.require_native88(native_raw,config['native88_independent']['SHA256'])
    need(cpu['original_initial_named_SHA256']==assets['initial']['named_SHA256'],'native88 initial named pin differs')
    need(cpu['actual_GPU_allocation']==cpu['actual_optimizer_updates']==0 and cpu['model_loaded'] is False,'native88 qualification scope differs')
    view_path=Path(config['view']['path'])
    need(view_path==RUN/'view/training800.jsonl' and file_pin(view_path,2*1024**2)=={k:config['view'][k] for k in ('bytes','SHA256')},'800 actual view pin differs')
    ledger=checked_record(config['ledger'],RUN)
    need(ledger['source_rows']==88 and ledger['source_SHA256']=='0a8560e6046439aa7450a3c1080ff5e537be17a56a8d4d332ff501d302b6bc62'
         and ledger['physical_row_count']==800 and ledger['view_SHA256']==config['view']['SHA256']
         and ledger['original_index_by_physical_row']==[i%88 for i in range(800)] and ledger['dev_or_blind_read'] is False,'exposure ledger differs')
    framework=stage['framework'];need(framework['files']==393 and framework['base_manifest_SHA256']==BASE_MANIFEST_SHA,'common framework baseline differs')
    fmanifest=checked_record(framework['manifest'],RUN,2*1024**2)
    files=fmanifest.get('files',fmanifest)
    need(len(files)==393,'installed candidate framework file count differs')
    for name,pin in files.items():
        rel=Path(name);need(not rel.is_absolute() and '..' not in rel.parts,'framework path escape')
        need(file_pin(RUN/'framework'/rel,2*1024**2)=={k:pin[k] for k in ('bytes','SHA256')},'actual isolated framework member differs')
    plans={}
    for arm in ('continuous','resumed'):
        plan=checked_record(config['plans'][arm],RUN,2*1024**2)
        expected=normalized_expected(recipe.command_plan(RUN/'framework/tools/hygon_vlm_recipe.py',
            (RUN/'src/original-launch.json').read_bytes(),str(RUN),config['view']['SHA256'],arm,PORT),RUN)
        need(plan['command']==expected['command'] and plan['environment_overrides']==expected['environment_overrides'],'HOST native plan differs from original T05 bridge')
        need(plan['command'][0]==str(RUN/'framework/vlm/runtime/swift_entry.py'),'native plan bypasses isolated framework')
        plans[arm]=plan
    return plans


def checkpoint_inventory(run,arm,step,budget):
    root=run/arm/'train'/('checkpoint-%d'%step);pins={}
    for name in sorted(CHECKPOINT_NAMES):budget.left(reserve=1);pins[name]=file_pin(root/name)
    state=read(root/'trainer_state.json')
    need(state['global_step']==step and state['max_steps']==25,'checkpoint native global step differs')
    need(type(state['epoch']) in (int,float) and state['epoch']==step/25,'checkpoint native loader/epoch differs')
    value=dict(arm=arm,global_step=step,member_count=14,files=pins,
               actual_tensor_deserialization_here=False,accumulation_boundary='native witness complete-four-micro-update')
    save(run/(arm+'-checkpoint-%d-files.json'%step),value)
    return value


def rank_metadata(arm,run):
    result=[];expected=100 if arm=='continuous' else 20
    for rank in range(8):
        root=run/arm/'train/minicpm-usgs-witness'/('rank-%02d'%rank)
        row=dict(rank=rank,planned_micro=expected,planned_updates=25 if arm=='continuous' else 5,
                 result_present=False,progress_present=False,completed_micro=None,completed_updates=None,complete=False)
        try:
            path=root/'result.json'
            if path.is_file():
                value=read(path);row.update(result_present=True,result_file=dict(path=str(path),**file_pin(path,8*1024**2)))
                steps=[x['global_step'] for x in value['steps']]
                row.update(completed_micro=len(value['outputs']),completed_updates=len(steps))
                row['complete']=bool(value['rank']==rank and value['world_size']==8 and value['completed_global_step']==25
                    and len(value['inputs'])==len(value['outputs'])==expected
                    and steps==list(range(1,26) if arm=='continuous' else range(21,26))
                    and value['saved_checkpoints']==([10,20,25] if arm=='continuous' else [25]))
                if arm=='resumed':row['complete']=row['complete'] and value['native_resume_window_complete'] is True
            else:
                path=root/'progress.json'
                if path.is_file():
                    progress=read(path);row.update(progress_present=True,progress_file=dict(path=str(path),**file_pin(path,2*1024**2)),
                        completed_micro=progress.get('outputs'),completed_updates=len(progress.get('completed_steps',[])))
        except (ValueError,OSError,KeyError,TypeError) as error:row['metadata_error']=dict(type=type(error).__name__,message=str(error)[:180])
        result.append(row)
    return result


def run_arm(arm,run,plan,config,budget,owned,popen=subprocess.Popen,sleep=time.sleep):
    need(arm in ('continuous','resumed'),'unknown native arm')
    budget.left(reserve=30)
    directory=run/arm;directory.mkdir(mode=0o700,exist_ok=False)
    witness=dict(asset_binding=str(run/'src/ASSETS.json'))
    if arm=='resumed':
        saved=read(run/'continuous-checkpoint-20-files.json')
        need(saved['global_step']==20 and saved['member_count']==14 and set(saved['files'])==CHECKPOINT_NAMES,'exact checkpoint20 fullstate pins required')
        witness.update(checkpoint=str(run/'continuous/train/checkpoint-20'),checkpoint_files=saved['files'])
    cfg=run/(arm+'-witness-config.json');save(cfg,witness)
    env=dict(os.environ,**plan['environment_overrides'])
    need(not any(k in env for k in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','NODE_RANK','NNODES')),'native parent must start unranked')
    env.update(MASTER_ADDR='127.0.0.1',MINICPM_USGS_CONFIG=str(cfg),MINICPM_USGS_CONFIG_SHA256=file_pin(cfg)['SHA256'],
        MINICPM_USGS_EPOCH=str(config['epoch']),MINICPM_USGS_DEADLINE=str(config['native_deadline']),MINICPM_USGS_MAX_SECONDS=str(NATIVE))
    save(directory/'native-intent.json',dict(request=REQUEST,arm=arm,epoch=config['epoch'],deadline=config['native_deadline'],
         planned_updates=25 if arm=='continuous' else 5,planned_micro_per_rank=100 if arm=='continuous' else 20,
         native_plan_SHA256=digest(json.dumps(plan,sort_keys=True).encode())))
    registry={};child=None;rc=None;failure=None;cleaned=None;cleanup_errors=[];started=time.monotonic()
    try:
        budget.left(reserve=2)
        with (directory/'native.log').open('xb') as log:
            child=popen([sys.executable,'-B',*plan['command']],stdin=subprocess.DEVNULL,stdout=log,
                        stderr=subprocess.STDOUT,env=env,start_new_session=True)
            observed=owned.snapshot().get(child.pid)
            need(observed is not None,'native Popen child strong identity unavailable')
            registry[child.pid]={k:observed[k] for k in ('PID','PPID','PGID','startticks','state')}
            save(directory/'child-owner.json',dict(request=REQUEST,host=socket.gethostname(),
                 boot_id=config.get('boot_id'),identity=registry[child.pid]))
            while child.poll() is None:
                owned.refresh(registry,directory);budget.left(reserve=1)
                temporary=directory/'owned-registry.tmp';temporary.write_text(json.dumps(registry),'utf-8');temporary.replace(directory/'owned-registry.json')
                sleep(.25)
            rc=child.returncode
            until=time.monotonic()+min(5,budget.left(reserve=1))
            while rc==0 and owned.refresh(registry,directory) and time.monotonic()<until:sleep(.1)
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error)[:350])
    finally:
        # A second TERM cannot abort strong-owner cleanup or receipt persistence.
        if os.name!='nt':signal.signal(signal.SIGTERM,signal.SIG_IGN)
        try:
            try:available=budget.left(CLEANUP,reserve=4)
            except ValueError:
                available=0
                cleanup_errors.append(dict(stage='cleanup_budget',type='ValueError',message='original cleanup budget exhausted'))
            grace=min(5,available/2)
            cleaned=owned.cleanup(registry,directory,grace=grace)
        except BaseException as error:cleanup_errors.append(dict(stage='owned_cleanup',type=type(error).__name__,message=str(error)[:180]))
        try:
            if child is not None:
                try:available=budget.left(CLEANUP,reserve=1)
                except ValueError:available=0
                child.wait(timeout=min(3,available));rc=child.returncode
        except BaseException as error:cleanup_errors.append(dict(stage='native_child_reap',type=type(error).__name__,message=str(error)[:180]))
        save(directory/'owned-registry-final.json',registry)
    ranks=rank_metadata(arm,run)
    passed=bool(rc==0 and failure is None and not cleanup_errors and cleaned==dict(owned_processes_remaining=0,termination_needed=False)
                and all(r['complete'] for r in ranks))
    result=dict(request=REQUEST,arm=arm,rc=rc,error=failure,cleanup=cleaned,cleanup_errors=cleanup_errors,
                cleanup_known=cleaned is not None and not cleanup_errors,owners=registry,ranks=ranks,passed=passed,
                planned_updates=25 if arm=='continuous' else 5,attempted_child=child is not None,
                planned_micro_per_rank=100 if arm=='continuous' else 20,wall_seconds=time.monotonic()-started,
                accounting_not_performed_here=True,formal_performance=False,Model_GO=False)
    save(directory/'native-outcome.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path);parser.add_argument('--config-sha256',required=True)
    args=parser.parse_args()
    need(file_pin(args.config,2*1024**2)['SHA256']==args.config_sha256,'worker config identity differs')
    config=read(args.config,2*1024**2);check_config(config,time.time())
    budget=Budget(config['epoch'],config['deadline'],config['native_deadline'])
    fields=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
    identity=dict(request=REQUEST,host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        PID=os.getpid(),PGID=os.getpgrp(),startticks=int(fields[19]),epoch=config['epoch'],deadline=config['deadline'])
    save(RUN/'worker-owner.json',identity);config['boot_id']=identity['boot_id']
    def terminate(*_):raise RuntimeError('original native worker received TERM')
    signal.signal(signal.SIGTERM,terminate)
    outcome=dict(request=REQUEST,epoch=config['epoch'],deadline=config['deadline'],native_deadline=config['native_deadline'],
        planned_total_optimizer_updates=30,planned_arms=['continuous','resumed'],arms=[],passed=False,
        HOST_post_and_accounting_pending=True,formal17_qualified=False,Model_GO=False)
    try:
        plans=source_and_data(config)
        need(socket.gethostname().split('.')[0]==NODE and os.environ['SLURM_JOB_ID']==PARENT,'wrong original Slurm node/parent')
        need(re.fullmatch(r'\d+',os.environ['SLURM_STEP_ID']) and int(os.environ['SLURM_CPUS_PER_TASK'])==64,'expected allocated8DCU/64CPU native step')
        device=load_module('minicpm_target_snapshot',Path(__file__).with_name('snapshot.py'))
        owned=load_module('minicpm_target_owned',Path(__file__).with_name('owned_processes.py'))
        before=checked_record(config['before'],RUN);current=device.snapshot(True)
        check_snapshots(before,current,len(os.sched_getaffinity(0)))
        with socket.socket() as port:port.bind(('127.0.0.1',PORT))
        hip=ctypes.CDLL('libamdhip64.so');hip.hipGetDeviceCount.argtypes=[ctypes.POINTER(ctypes.c_int)];hip.hipGetDeviceCount.restype=ctypes.c_int
        hip.hipDeviceGetPCIBusId.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int];hip.hipDeviceGetPCIBusId.restype=ctypes.c_int
        count=ctypes.c_int();need(hip.hipGetDeviceCount(ctypes.byref(count))==0 and count.value==8,'actual HIP count differs')
        pci=[]
        for rank in range(8):
            value=ctypes.create_string_buffer(32);need(hip.hipDeviceGetPCIBusId(value,len(value),rank)==0,'actual HIP PCI unavailable')
            name=value.value.decode().lower()
            if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',name):name='0000:'+name
            need(re.fullmatch(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',name),'actual HIP PCI schema differs');pci.append(name)
        need(len(set(pci))==8 and sorted(pci)==current['pci'],'actual visible HIP devices differ from owned inventory')
        save(RUN/'worker-assigned.json',dict(identity,actual_step=PARENT+'.'+os.environ['SLURM_STEP_ID'],
             visible_PCI_by_index=pci,leased_snapshot=current,actual_CPU_affinity=sorted(os.sched_getaffinity(0))))
        for arm in ('continuous','resumed'):
            signal.signal(signal.SIGTERM,terminate)
            result=run_arm(arm,RUN,plans[arm],config,budget,owned);outcome['arms'].append(result)
            need(result['passed'],'native arm incomplete or failed; original outcome retained')
            for step in ([10,20,25] if arm=='continuous' else [25]):checkpoint_inventory(RUN,arm,step,budget)
            with socket.socket() as port:port.bind(('127.0.0.1',PORT))
        compare=load_module('minicpm_target_compare',RUN/'src/compare_recovery.py').compare
        comparison=compare(RUN);save(RUN/'paired-recovery.json',comparison)
        need(comparison['exact_paired_recovery'] is True,'original exact paired recovery differs')
        outcome['passed']=True
    except BaseException as error:outcome['error']=dict(type=type(error).__name__,message=str(error)[:500])
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        outcome['arms_not_attempted']=[a for a in outcome['planned_arms'] if a not in [r['arm'] for r in outcome['arms']]]
        save(RUN/'worker-outcome.json',outcome)
    print(json.dumps({k:outcome[k] for k in ['request','passed','planned_total_optimizer_updates','arms_not_attempted','HOST_post_and_accounting_pending','formal17_qualified']}),flush=True)
    return 0 if outcome['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
