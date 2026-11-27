"""Held eight-HIP-rank Gemma continuous25 using unchanged audited RSS/tree helpers."""
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

import rank_ipc
import owned_processes as owned
from held_child import cleanup_held
from short_lifecycle import Lifecycle, receipt_domains, retire_natural
from rendezvous import DIAGNOSTIC, WAIT
from short_mount import SHORT, SOURCE as SHORT_SOURCE, verify_inside
from rss_monitor import sample_group_rss,RSS_SOFT_BYTES
from contract import D,ROOT,ASSET,CORE,CORE_MF_SHA,STAGE_PIN,FRAMEWORK_SHA,RUNTIME_SHA,REQUEST,NODE,PARENT,INSIDE,CLEANUP,Clock,need,stable_json,save,pin,verify

BOOTSTRAP="import sys,os; assert sys.stdin.buffer.readline(4)==b'GO\\n'; os.execv(sys.executable,[sys.executable]+sys.argv[1:])"
STATE_FILES={'adapter_model.safetensors','adapter_config.json','trainer_state.json','args.json',
             'optimizer.pt','scheduler.pt',*(f'rng_state_{i}.pth' for i in range(8))}


def checkpoint_inventory(step,clock):
    folder=ROOT/'continuous/train'/('checkpoint-%d'%step)
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
        return (args[index+1]==str(ROOT/'continuous/train').encode()
                and b'gemma_usgs_r39_two_updates_witness' in args)
    except (ValueError,IndexError):
        return False


def main():
    verify('GPU')
    intent,intent_pin=stable_json(ROOT/'intent.json')
    clock=Clock(intent['epoch']);registry={};child=None;anchor={};failure=None;phase='admission'
    held={};tree={};cleanup_errors=[];rc=None;rank_pins=[];checkpoints={};short_initial=False;rank_progress={}
    ipc=rank_ipc.Owner(SHORT,ROOT,REQUEST)
    rank_owners={}
    ipc_verified={}
    lifecycle=Lifecycle(SHORT,os.getuid(),archive_path=ROOT/'short-private-tmp-events.jsonl')
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
        need(stage['passed'] is True and stage['training_request_id']=='247fa3682a114542ac3640fbc726aaae'
             and stage['source_manifest']['SHA256']==CORE_MF_SHA,'original845 stage missing')
        need(pin(ASSET/'fresh-framework-files.json')['SHA256']==FRAMEWORK_SHA
             and pin(ASSET/'runtime-binding.json')['SHA256']==RUNTIME_SHA,'staged framework/runtime changed')
        need(pin(CORE/'SOURCE_FILES.json')['SHA256']==CORE_MF_SHA,'frozen stage/core source changed')
        before,before_pin=stable_json(ROOT/'before.json')
        need(before['request_id']==REQUEST and before['boot_id']==supervisor['boot_id']
             and 0<=time.monotonic_ns()-before['monotonic_ns']<=45_000_000_000,
             'fresh original HOST admission expired')
        need(not (ROOT/'continuous').exists(),'continuous output already exists; never replay')
        run=dict(request_id=REQUEST,root=str(ROOT),parent=PARENT,node=NODE,epoch=intent['epoch'],
            deadline=intent['epoch']+2400,native_deadline=intent['epoch']+INSIDE,
            source_manifest=stage['source_manifest'],stage_complete=STAGE_PIN,boot_id=before['boot_id'],
            operational_source_manifest=intent['source_manifest'])
        run_pin=save('training-intent.json',run)
        clock.left(INSIDE,30)
        command=[sys.executable,'-c',BOOTSTRAP,str(D/'native_worker.py'),
            '--run',str(ROOT/'training-intent.json'),'--run-sha',run_pin['SHA256'],
            '--admission',str(ROOT/'before.json'),'--admission-sha',before_pin['SHA256']]
        env=dict(os.environ)
        need(not any(n in env for n in ('RANK','LOCAL_RANK','WORLD_SIZE')),'ranked supervisor environment')
        verify_inside(check_temp_environment=True)
        short_initial=True
        env['GEMMA_IPC_ASSIGNMENTS_SHA256']=ipc.reserve()
        lifecycle.sample('BEFORE_NATIVE')
        phase='held_start';save('native-attempt.json',dict(request_id=REQUEST,epoch=intent['epoch'],
            planned_updates=2,planned_rank_microbatches=8,planned_total_exposures=800,command=command))
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
                lifecycle.sample('RUNNING')
                owned.refresh(registry,ROOT/'continuous')
                save('owned-registry.json',registry,replace=True)
                try:
                    below=sample_group_rss(registry,resources)
                finally:
                    save('resource-samples.json',resources,replace=True)
                need(below,'owned sampled RSS soft stop')
                need(all((ROOT/n).stat().st_size<=16*1024**2 for n in ('native.stdout','native.stderr')),'native log bound exceeded')
                time.sleep(min(.25,clock.left(INSIDE)))
            rc=child.returncode
        lifecycle.sample('NATIVE_RETURNED')
        need(rc==0,'native common Swift failed')
        phase='rank_receipts'
        for rank in range(8):
            folder=ROOT/'continuous/train/continuous-witness'/('rank-%02d'%rank)
            result,actual=stable_json(folder/'result.json',cap=8*1024**2)
            need(result["changing_steps_with_nonzero_rate"] and any(step["adapter_changed"]
                 and any(rate > 0 for rate in step["learning_rates"]) for step in result["steps"]),
                 "R120_POSITIVE_LR_REAL_ADAPTER_PARAMETER_CHANGE_REQUIRED")
            initial,init_pin=stable_json(folder/'initialization.json')
            owner,owner_pin=stable_json(folder/'owner.json')
            rank_owners[rank]=owner
            sampler,sampler_pin=stable_json(folder/'native-sampler.json')
            vote,vote_pin=stable_json(folder/'sampler-votes'/DIAGNOSTIC,cap=32768)
            verdict,verdict_pin=stable_json(folder/'sampler-votes'/WAIT,cap=32768)
            need(vote['rank']==verdict['rank']==rank and vote['passed'] is True
                 and verdict['all_rank_votes_valid'] is verdict['all_rank_barrier_passed'] is True,
                 'R46 exact Swift shard vote and barrier required')
            need(result['rank']==rank and result['world_size']==8 and result['completed_global_step']==2
                 and len(result['inputs'])==len(result['outputs'])==8 and len(result['steps'])==2
                 and result['saved_checkpoints']==[] and result['parameter_update_passed'] is True
                 and sampler['rank']==rank and sampler['RNG_unchanged'] is True
                 and initial['reference_initial_read'] is False and initial['RNG_unchanged'] is True
                 and owner['rank']==owner['local_rank']==rank,'incomplete original rank25 receipt')
            rank_pins.append(dict(rank=rank,result=actual,initialization=init_pin,owner=owner_pin,sampler=sampler_pin,
                                  sampler_vote=vote_pin,sampler_unanimous=verdict_pin,
                                  fresh_named_SHA256=initial['named']['SHA256']))
        need(len({r['fresh_named_SHA256'] for r in rank_pins})==1,'actual initial8 values differ')
        ipc_verified=ipc.verified_ranks(rank_owners)
        phase='native_two_update_adapter_metadata'
        adapter=ROOT/'continuous/train/r39-final-adapter'
        need({name.name for name in adapter.iterdir()}=={'adapter_model.safetensors','adapter_config.json'},
             'R39 exact final adapter was not emitted')
        checkpoints['2']={path.name:pin(path) for path in adapter.iterdir()}
    except BaseException as e:
        failure=dict(phase=phase,type=type(e).__name__,reason=str(e)[:512])
    finally:
        # Cleanup branches are independent. A helper failure never hides the other branch.
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        try:
            held=cleanup_held(child,anchor,intent['epoch']+CLEANUP)
        except BaseException as e:cleanup_errors.append(dict(stage='held',type=type(e).__name__))
        try:
            tree=owned.cleanup(registry,ROOT/'continuous',grace=5)
        except BaseException as e:cleanup_errors.append(dict(stage='tree',type=type(e).__name__))
        for rank in range(8):
            folder=ROOT/'continuous/train/continuous-witness'/('rank-%02d'%rank)
            try:
                progress,record=stable_json(folder/'progress.json',cap=4096)
                need(progress['rank']==rank and progress['planned_updates']==2 and progress['planned_microbatches']==8
                     and all(type(progress[k]) is int for k in ('inputs','outputs','recorded_updates'))
                     and 0<=progress['recorded_updates']<=2 and 0<=progress['outputs']<=progress['inputs']<=8,
                     'R46_BOUNDED_PROGRESS')
                rank_progress[str(rank)]={'pin':record,**{k:progress[k] for k in ('inputs','outputs','recorded_updates')}}
            except (OSError,ValueError,TypeError,KeyError) as exc:
                rank_progress[str(rank)]={'status':type(exc).__name__}
        retirement={}
        registered={}
        lifecycle.sample('PRE_RETIREMENT')
        try:
            verify_inside(check_temp_environment=False,require_marker_only=False)
            creation,creation_pin=stable_json(ROOT/'short-private-tmp-creation.json',cap=4096)
            ipc.retire(reaped=bool(held.get('reaped') is True and tree.get('owned_processes_remaining')==0),report=registered)
            retire_natural(SHORT_SOURCE,SHORT,creation,REQUEST,initially_marker_only=short_initial,report=retirement,
                           children_reaped=bool(held.get('reaped') is True and tree.get('owned_processes_remaining')==0))
            verify_inside(check_temp_environment=False)
        except BaseException as e:
            code=str(e) if type(e) is ValueError else ''
            if not (0<len(code)<=96 and all(x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789' for x in code)): code=None
            retirement['refusal']=dict(type=type(e).__name__,code=code,errno=getattr(e,'errno',None))
            cleanup_errors.append(dict(stage='short_private_tmp',**retirement['refusal']))
        try:
            ipc.close()
        except BaseException as e:
            cleanup_errors.append(dict(stage='registered_ipc_fds',type=type(e).__name__))
        try:
            lifecycle.sample('POST_RETIREMENT')
            save('short-private-tmp-cleanup.json',dict(request_id=REQUEST,retirement=retirement,registered=registered))
            save('short-private-tmp-lifecycle.json',dict(request_id=REQUEST,**lifecycle.evidence()))
            need(not lifecycle.errors and not lifecycle.truncated_events,'R46_SHORT_LIFECYCLE_WITNESS_INCOMPLETE')
        except BaseException as e:
            cleanup_errors.append(dict(stage='short_private_tmp_receipt',type=type(e).__name__))
        try:save('resource-samples.json',resources,replace=True)
        except BaseException as e:cleanup_errors.append(dict(stage='resource_save',type=type(e).__name__))
        domains=receipt_domains(exact_rank_results=rank_pins,returncode=rc,failure=failure,
            cleanup_errors=cleanup_errors,held=held,tree=tree,retirement=retirement)
        passed=domains["passed"]
        progress_set=[item["recorded_updates"] for item in rank_progress.values() if "recorded_updates" in item]
        witnessed_updates=2 if domains["native_two_updates_witnessed"] else (
            min(progress_set) if len(progress_set)==8 else (0 if child is None else None))
        save("native-outcome.json",dict(request_id=REQUEST,epoch=intent["epoch"],**domains,
            actual_updates=witnessed_updates,planned_updates=2,nominal_scheduler_steps=25,planned_total_exposures=800,
            returncode=rc,failure=failure,held_cleanup=held,tree_cleanup=tree,cleanup_errors=cleanup_errors,
            supervisor=supervisor,owners=registry,rank_receipts=rank_pins,rank_progress=rank_progress,
            resources=resources,checkpoints=checkpoints,registered_IPC=registered,verified_rank_IPC=ipc_verified,
            original_A100_initial_read=False,Model_GO=False,formal17_GO=False))
        signal.signal(signal.SIGTERM,previous)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
