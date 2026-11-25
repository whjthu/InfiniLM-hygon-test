"""Single assigned DCU, exact frozen infer child held with audited tree/RSS cleanup."""
import os
import math
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import owned_processes as owned
import rss_monitor as rss
from held_child import cleanup_held
from runtime_pci import runtime_pci
from assigned_kfd import assigned_kfd
from contract import *

rss.RSS_SOFT_BYTES=32*1024**3 # exact RSS-v2 functions, explicit sampled soft profile, no hard-RSS claim
BOOTSTRAP="import sys,os; assert sys.stdin.buffer.readline(4)==b'GO\\n'; os.execv(sys.executable,[sys.executable]+sys.argv[1:])"

def native_command(args,unused):
    try:
        i=args.index(b'--output')
        return args[i+1]==str(ASSET/'generated').encode() and str(CORE/'infer_blind36.py').encode() in args
    except (ValueError,IndexError):return False

def validate_device(before,current,selected,device,now_ns,step,selector):
    need(before['request_id']==REQUEST and before['step']==step and current['pci']==before['pci'] and
         len(set(before['pci']))==8 and current['boot_id']==before['boot_id'] and
         before['kfd_holder_count']==0 and before['full_fd_inventory_performed'] is True and
         0<=now_ns-before['monotonic_ns']<=45_000_000_000 and
         re.fullmatch(r'0|[1-9][0-9]*',selector) is not None and selected in before['pci'] and
         device['pci']==selected and device['gpu_id']>0,'fresh real eight-physical / actual assigned-one evidence required')

def work_deadlines(epoch,submitted_at,slurm_seconds,now):
    need(all(type(x) in (int,float) and math.isfinite(x) for x in (epoch,submitted_at,slurm_seconds,now)) and
         0<epoch<=submitted_at<=now and 60<=slurm_seconds<=GPU,'actual original Slurm deadline inputs required')
    end=submitted_at+slurm_seconds
    native=min(epoch+INSIDE-10,end-55)
    cleanup=min(epoch+CLEANUP,end-5)
    need(now+30<native<native+5<cleanup<=epoch+WHOLE,'bounded native/cleanup reserve exhausted')
    return native,cleanup

def main():
    source=verify('GPU');intent,ipin=stable_json(ROOT/'intent.json');clock=Clock(intent['epoch'])
    step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','')
    need(numbered([step]) and socket.gethostname().split('.')[0]==NODE,'wrong actual inference lease')
    mine=owned.snapshot().get(os.getpid());need(mine is not None,'keeper strong identity UNKNOWN')
    owner=dict(PID=os.getpid(),startticks=mine['startticks'],host=socket.gethostname(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),scope_request_id=REQUEST)
    save('supervisor-owner.json',owner)
    save('gpu-step-entry.json',dict(request_id=REQUEST,epoch=intent['epoch'],step=step,intent=ipin,supervisor=owner))
    child=None;anchor={};registry={};held={};tree={};errors=[];failure=None;rc=None;phase='device_admission';final=None
    resources=dict(samples=0,sampled_group_peak_RSS_bytes=0,soft_limit_bytes=rss.RSS_SOFT_BYTES,
        soft_limit_exceeded=False,hard_limit=False,method='Qualified RSS-v2 owned PID/startticks, 32GiB sampled soft stop; unknown preserved, not0')
    previous=signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('keeper stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original660 keeper work')))
    signal.setitimer(signal.ITIMER_REAL,clock.left(INSIDE,3))
    cleanup_deadline=intent['epoch']+CLEANUP
    try:
        before,bpin=stable_json(ROOT/'gpu-host-before.json')
        # Read-only DRM+kernel evidence before our own HIP SDK can open KFD.
        import importlib.util
        spec=importlib.util.spec_from_file_location('effect_container_snapshot',D/'snapshot.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        current=dict(pci=module.drm_inventory(),boot_id=owner['boot_id'])
        need(len(module.kernel_holders())==0,'container kernel KFD had an existing holder before own SDK')
        selector=os.environ.get('CUDA_VISIBLE_DEVICES','')
        need(re.fullmatch(r'0|[1-9][0-9]*',selector) is not None,'one actual Slurm selector required, no parent8 fallback')
        selected=runtime_pci() # exact qualified helper: hipGetDeviceCount must equal1
        device=assigned_kfd(selected) # exact qualified helper; unallocated denied properties remain explicit
        validate_device(before,current,selected,device,time.monotonic_ns(),step,selector)
        with socket.socket() as probe:probe.bind(('127.0.0.1',PORT))
        phase_intent,_=stable_json(ROOT/'gpu-step-intent.json')
        native_deadline,cleanup_deadline=work_deadlines(intent['epoch'],phase_intent['submitted_at'],phase_intent['slurm_seconds'],time.time())
        # The frozen worker gets five seconds to persist its timeout/failure seal
        # before this keeper signals it. Both deadlines derive from the same epoch.
        signal.setitimer(signal.ITIMER_REAL,native_deadline+5-time.time())
        admission=dict(request=REQUEST,passed=True,epoch=intent['epoch'],deadline=native_deadline,
            source_manifest=core_source(),operational_source_manifest=source,host=owner['host'],boot_id=owner['boot_id'],
            monotonic_ns=time.monotonic_ns(),physical_GPU_count=8,physical_PCI=before['pci'],actual_visible_devices=1,
            allocated_DCU=1,selected_PCI=selected,assigned_device=device,actual_Slurm_selector=selector,step=step,
            actual_GPU_step_host_snapshot=bpin,fresh_exact_GPU_PCI_KFD_port_scope_satisfied=True,
            policy='HOST full8/FD/KFD clean before own SDK; actual one HIP PCI + accessible assigned KFD only',
            unassigned_KFD_properties_complete=False,allFDclean_claimed=False,exclusive_GPU_claimed=False)
        apin=save('admission.json',admission)
        binding,binding_pin=stable_json(ASSET/'effect-binding.json')
        need(binding['source_manifest']==core_source(),'actual ready binding core source differs')
        need(not (ASSET/'generated').exists(),'original generation already attempted; never replay')
        argv=[sys.executable,'-c',BOOTSTRAP,str(CORE/'infer_blind36.py'),'--binding',str(ASSET/'effect-binding.json'),
            '--binding-bytes',str(binding_pin['bytes']),'--binding-sha',binding_pin['SHA256'],
            '--admission',str(ROOT/'admission.json'),'--admission-bytes',str(apin['bytes']),'--admission-sha',apin['SHA256'],
            '--output',str(ASSET/'generated'),'--epoch',str(intent['epoch']),'--deadline',str(native_deadline)]
        save('native-attempt.json',dict(request_id=REQUEST,epoch=intent['epoch'],deadline=native_deadline,
            whole_deadline=intent['epoch']+WHOLE,cleanup_deadline=cleanup_deadline,argv=argv,planned_each=36,planned_total=108,
            admission=apin,ready_binding=binding_pin))
        owned.native_command=native_command;phase='held_inference'
        with (ROOT/'native.stdout').open('xb') as out,(ROOT/'native.stderr').open('xb') as err:
            child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=out,stderr=err,env=dict(os.environ),start_new_session=True)
            anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True,scope_request_id=REQUEST)
            actual=owned.snapshot().get(child.pid)
            need(actual is not None and actual['PPID']==os.getpid() and actual['PGID']==child.pid,'held child identity UNKNOWN')
            registry[child.pid]={k:actual[k] for k in ('PID','PPID','PGID','startticks','state')}
            save('held-child-owner.json',dict(**anchor,startticks=actual['startticks'],supervisor=owner))
            need(0<=time.monotonic_ns()-admission['monotonic_ns']<=45_000_000_000,'fresh admission expired before GO')
            child.stdin.write(b'GO\n');child.stdin.flush();child.stdin.close()
            while child.poll() is None:
                clock.left();need(time.time()<native_deadline+5,'original native work and failure-seal grace deadline')
                owned.refresh(registry,ASSET/'generated');save('owned-registry.json',registry,replace=True)
                try:below=rss.sample_group_rss(registry,resources)
                finally:save('resource-samples.json',resources,replace=True)
                need(below,'sampled32GiB soft stop')
                need(all((ROOT/n).stat().st_size<=16*1024**2 for n in ('native.stdout','native.stderr')),'native log cap')
                time.sleep(min(.25,max(.001,native_deadline-time.time())))
            rc=child.returncode
        phase='generation_finalized'
        if (ASSET/'generated/generation-finalized.json').exists():
            final,finalpin=stable_json(ASSET/'generated/generation-finalized.json')
            need(final['request']==REQUEST and final['epoch']==intent['epoch'] and final['deadline']==native_deadline and
                 final['source_manifest']==core_source(),'actual generation context differs')
        need(rc==0 and final is not None and final['passed'] is True and final['all108sealed_before_gold'] is True,
             'inference actual failed/partial, preserve108 denominator')
        need(resources['samples']>0,'actual sampled RSS absent')
    except BaseException as e:failure=dict(phase=phase,type=type(e).__name__,private_reason=str(e)[:512])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        try:held=cleanup_held(child,anchor,cleanup_deadline)
        except BaseException as e:errors.append(dict(phase='held_cleanup',type=type(e).__name__))
        try:tree=owned.cleanup(registry,ASSET/'generated',grace=3)
        except BaseException as e:errors.append(dict(phase='tree_cleanup',type=type(e).__name__))
        try:save('resource-samples.json',resources,replace=True)
        except BaseException as e:errors.append(dict(phase='resources_save',type=type(e).__name__))
        passed=bool(rc==0 and failure is None and not errors and held.get('reaped') is True and
            held.get('termination_needed') is False and tree.get('owned_processes_remaining')==0 and tree.get('termination_needed') is False)
        save('native-outcome.json',dict(request_id=REQUEST,epoch=intent['epoch'],whole_deadline=intent['epoch']+WHOLE,
            passed=passed,returncode=rc,failure=failure,cleanup_errors=errors,supervisor=owner,owners=registry,
            held_cleanup=held,tree_cleanup=tree,resources=resources,planned_each=36,planned_total=108,
            actual_arm_counts=final.get('arms') if final else None,
            generation_finalized=pin(ASSET/'generated/generation-finalized.json') if final else None,
            actual_updates=0,Model_GO=False,formal17_GO=False))
        signal.signal(signal.SIGTERM,previous)
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
