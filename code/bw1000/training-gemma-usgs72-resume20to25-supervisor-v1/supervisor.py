"""Held eight-HIP-rank Gemma resume20to25 using unchanged audited RSS/tree helpers."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import time

import owned_processes as owned
from held_child import cleanup_held
from rss_monitor import sample_group_rss,RSS_SOFT_BYTES
from contract import D,ROOT,ASSET,CORE,CORE_MF_SHA,STAGE_PIN,FRAMEWORK_SHA,RUNTIME_SHA,REQUEST,NODE,PARENT,INSIDE,CLEANUP,Clock,need,stable_json,save,pin,verify,prior_continuous,CONTINUOUS_GATE_PIN
from compare_recovery import compare

BOOTSTRAP="import sys,os; assert sys.stdin.buffer.readline(4)==b'GO\\n'; os.execv(sys.executable,[sys.executable]+sys.argv[1:])"
STATE_FILES={'adapter_model.safetensors','adapter_config.json','trainer_state.json','args.json',
             'optimizer.pt','scheduler.pt',*(f'rng_state_{i}.pth' for i in range(8))}


def checkpoint_inventory(step,clock):
    folder=ASSET/'resumed/train'/('checkpoint-%d'%step)
    pins={}
    for name in sorted(STATE_FILES):
        path=folder/name;before=path.lstat()
        identity=lambda s:[getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')]
        need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=512*1024**2,'native state file budget/type')
        h=hashlib.sha256();total=0
        with path.open('rb') as f:
            need(identity(os.fstat(f.fileno()))==identity(before),'state open identity changed')
            while True:
                clock.left(INSIDE)
                raw=f.read(min(1024**2,before.st_size-total+1))
                if not raw:break
                total+=len(raw);need(total<=before.st_size,'state grew');h.update(raw)
            need(identity(os.fstat(f.fileno()))==identity(before),'state changed during SHA')
        need(total==before.st_size and identity(path.lstat())==identity(before),'state SHA identity changed')
        pins[name]=dict(bytes=total,SHA256=h.hexdigest(),file_identity=identity(before))
    need(sum(p['bytes'] for p in pins.values())<=512*1024**2,'native checkpoint total budget')
    trainer,_=stable_json(folder/'trainer_state.json');args,_=stable_json(folder/'args.json')
    need(trainer['global_step']==step and trainer['max_steps']==args['max_steps']==25
         and args['gradient_accumulation_steps']==4 and args['per_device_train_batch_size']==1,
         'checkpoint native cursor contract differs')
    result=dict(request_id=REQUEST,step=step,required_members=14,files=pins,
        mathematical_tensor_values_independently_deserialized=False)
    save('checkpoint-%d-files.json'%step,result)
    return result


def native_command(args, unused):
    try:
        index=args.index(b'--output_dir')
        return (args[index+1]==str(ASSET/'resumed/train').encode()
                and b'gemma_domain_resume_witness' in args)
    except (ValueError,IndexError):
        return False


def main():
    verify('GPU')
    intent,intent_pin=stable_json(ROOT/'intent.json')
    clock=Clock(intent['epoch']);registry={};child=None;anchor={};failure=None;phase='admission'
    held={};tree={};cleanup_errors=[];rc=None;rank_pins=[];checkpoints={};comparison=None
    resources=dict(samples=0,sampled_group_peak_RSS_bytes=0,soft_limit_bytes=RSS_SOFT_BYTES,
        soft_limit_exceeded=False,hard_limit=False,
        method='Audited RSS-v2 owned PID/startticks VmRSS; live missing UNKNOWN then persistent rejection; sampled lower bound, shared pages may double count')
    row=owned.snapshot().get(os.getpid())
    need(row is not None,'supervisor strong identity unknown')
    supervisor=dict(PID=os.getpid(),startticks=row['startticks'],uid=os.getuid(),host=socket.gethostname(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),scope_request_id=REQUEST)
    save('supervisor-owner.json',supervisor)
    owned.native_command=native_command
    previous=signal.getsignal(signal.SIGTERM)
    def stop(*_):raise TimeoutError('owned supervisor stop requested')
    signal.signal(signal.SIGTERM,stop)
    try:
        need(socket.gethostname().split('.')[0]==NODE and os.environ.get('SLURM_JOB_ID')==PARENT,
             'wrong eight-card target owner')
        save('gpu-step-entry.json',dict(request_id=REQUEST,epoch=intent['epoch'],
            intent=intent_pin,node=NODE,step=PARENT+'.'+os.environ['SLURM_STEP_ID'],supervisor=supervisor))
        stage,_=stable_json(ASSET/'stage-complete.json',STAGE_PIN)
        need(stage['passed'] is True and stage['training_request_id']==REQUEST
             and stage['source_manifest']['SHA256']==CORE_MF_SHA,'original845 stage missing')
        need(pin(ASSET/'fresh-framework-files.json')['SHA256']==FRAMEWORK_SHA
             and pin(ASSET/'runtime-binding.json')['SHA256']==RUNTIME_SHA,'staged framework/runtime changed')
        need(pin(CORE/'SOURCE_FILES.json')['SHA256']==CORE_MF_SHA,'frozen stage/core source changed')
        before,before_pin=stable_json(ROOT/'before.json')
        need(before['request_id']==REQUEST and before['boot_id']==supervisor['boot_id']
             and 0<=time.monotonic_ns()-before['monotonic_ns']<=45_000_000_000,
             'fresh original HOST admission expired')
        prior_continuous()
        need(not (ASSET/'resumed').exists(),'resumed output already exists; never replay')
        run=dict(request_id=REQUEST,root=str(ASSET),parent=PARENT,node=NODE,epoch=intent['epoch'],
            deadline=intent['epoch']+2400,native_deadline=intent['epoch']+INSIDE,
            source_manifest=stage['source_manifest'],stage_complete=STAGE_PIN,boot_id=before['boot_id'],
            operational_source_manifest=intent['source_manifest'],continuous_independent_terminal=CONTINUOUS_GATE_PIN)
        run_pin=save('training-intent.json',run)
        clock.left(INSIDE,30)
        command=[sys.executable,'-c',BOOTSTRAP,str(CORE/'worker.py'),'resumed',
            '--run',str(ROOT/'training-intent.json'),'--run-sha',run_pin['SHA256'],
            '--admission',str(ROOT/'before.json'),'--admission-sha',before_pin['SHA256']]
        env=dict(os.environ)
        need(not any(n in env for n in ('RANK','LOCAL_RANK','WORLD_SIZE')),'ranked supervisor environment')
        phase='held_start';save('native-attempt.json',dict(request_id=REQUEST,epoch=intent['epoch'],
            planned_updates=5,planned_rank_microbatches=20,planned_total_exposures=160,command=command))
        with (ROOT/'native.stdout').open('xb') as out,(ROOT/'native.stderr').open('xb') as err:
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=out,stderr=err,env=env,start_new_session=True)
            anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True,request_id=REQUEST)
            save('held-child-anchor.json',anchor)
            current=owned.snapshot().get(child.pid)
            need(current is not None and current['PPID']==os.getpid() and current['PGID']==child.pid,'held child identity missing')
            registry[child.pid]={k:current[k] for k in ('PID','PPID','PGID','startticks','state')}
            save('held-child-owner.json',dict(**anchor,startticks=current['startticks'],supervisor=supervisor))
            need(time.monotonic_ns()-before['monotonic_ns']<=45_000_000_000,'admission expired before GO')
            child.stdin.write(b'GO\n');child.stdin.flush();child.stdin.close()
            phase='native_common_swift'
            while child.poll() is None:
                clock.left(INSIDE)
                owned.refresh(registry,ASSET/'resumed')
                save('owned-registry.json',registry,replace=True)
                try:
                    below=sample_group_rss(registry,resources)
                finally:
                    save('resource-samples.json',resources,replace=True)
                need(below,'owned sampled RSS soft stop')
                need(all((ROOT/n).stat().st_size<=16*1024**2 for n in ('native.stdout','native.stderr')),'native log bound exceeded')
                time.sleep(min(.25,clock.left(INSIDE)))
            rc=child.returncode
        need(rc==0,'native common Swift failed')
        phase='rank_receipts'
        for rank in range(8):
            folder=ASSET/'resumed/train/resume-witness'/('rank-%02d'%rank)
            result,actual=stable_json(folder/'result.json',cap=8*1024**2)
            owner,owner_pin=stable_json(folder/'owner.json')
            restored,restored_pin=stable_json(folder/'restored-before-RNG.json')
            skipped,skip_pin=stable_json(folder/'native-data-skip.json')
            need(result['rank']==rank and result['world_size']==8 and result['completed_global_step']==25
                 and len(result['inputs'])==len(result['outputs'])==20
                 and [s['global_step'] for s in result['steps']]==[21,22,23,24,25]
                 and result['saved_checkpoints']==[25] and result['native_resume_window_complete'] is True
                 and len(result['RNG_loads'])==1 and result['RNG_loads'][0]['saved_equal'] is True
                 and result['RNG_loads'][0]['global_step']==20
                 and restored['restored_parameter_optimizer_scheduler_equal'] is True
                 and skipped==dict(rank=rank,before=100,skipped=80,after=20)
                 and owner['rank']==owner['local_rank']==rank,'incomplete original resumed5 receipt')
            rank_pins.append(dict(rank=rank,result=actual,owner=owner_pin,restored_before_RNG=restored_pin,
                                  native_data_skip=skip_pin))
        phase='native_checkpoint_identity'
        checkpoints['25']=checkpoint_inventory(25,clock)
        phase='paired_recovery_comparison'
        comparison=compare(ASSET)
        save('pair-comparison.json',comparison)
    except BaseException as e:
        failure=dict(phase=phase,type=type(e).__name__,reason=str(e)[:512])
    finally:
        # Cleanup branches are independent. A helper failure never hides the other branch.
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        try:
            held=cleanup_held(child,anchor,intent['epoch']+CLEANUP)
        except BaseException as e:cleanup_errors.append(dict(stage='held',type=type(e).__name__))
        try:
            tree=owned.cleanup(registry,ASSET/'resumed',grace=5)
        except BaseException as e:cleanup_errors.append(dict(stage='tree',type=type(e).__name__))
        try:save('resource-samples.json',resources,replace=True)
        except BaseException as e:cleanup_errors.append(dict(stage='resource_save',type=type(e).__name__))
        passed=bool(rc==0 and failure is None and len(rank_pins)==8 and not cleanup_errors
            and held.get('reaped') is True and held.get('termination_needed') is False
            and tree.get('owned_processes_remaining')==0 and tree.get('termination_needed') is False)
        save('native-outcome.json',dict(request_id=REQUEST,epoch=intent['epoch'],passed=passed,
            actual_updates=5 if rc==0 and len(rank_pins)==8 else (0 if child is None else None),
            planned_updates=5,planned_total_exposures=160,original_continuous_updates=25,
            returncode=rc,failure=failure,held_cleanup=held,tree_cleanup=tree,cleanup_errors=cleanup_errors,
            supervisor=supervisor,owners=registry,rank_receipts=rank_pins,resources=resources,checkpoints=checkpoints,
            exact_paired_recovery=comparison.get('exact_paired_recovery') if comparison else None,
            comparison=pin(ROOT/'pair-comparison.json') if comparison else None,
            original_A100_initial_read=False,Model_GO=False,formal17_GO=False))
        signal.signal(signal.SIGTERM,previous)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
