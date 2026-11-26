"""Independent compute-side terminal and read-only Slurm accounting audit."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from contract import *

def original_absent(owner,host,boot,reader=None):
    need(owner['host']==host and owner['boot_id']==boot and owner['PID']>0 and owner['startticks']>0,
         'original owner current host/boot identity UNKNOWN')
    def read_proc(pid):
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()
        return dict(startticks=int(fields[19]),state=fields[0])
    try:now=(reader or read_proc)(owner['PID'])
    except (FileNotFoundError,ProcessLookupError):return 'ORIGINAL_IDENTITY_ABSENT'
    need(now['startticks']!=owner['startticks'] or now['state'] in ('Z','X'),'original keeper still live')
    return 'ORIGINAL_IDENTITY_ABSENT_PID_REUSED' if now['startticks']!=owner['startticks'] else 'ORIGINAL_TERMINAL_ZOMBIE'

def consume(epoch,source_sha):
    source=verify('READ');need(source['SHA256']==source_sha,'original outer source differs')
    need(socket.gethostname().split('.')[0]==NODE,'reader must use exact same compute mount')
    intent,ipin=stable_json(ROOT/'intent.json');out,opin=stable_json(ROOT/'outcome.json')
    need(intent['request_id']==out['request_id']==REQUEST and intent['epoch']==out['epoch']==epoch and
         intent['dispatch_request_id']==DISPATCH_REQUEST and intent['source_manifest']==out['source_manifest']==source and
         intent['deadline']==out['deadline']==epoch+WHOLE and intent['publication']==out['publication'] and
         out['core_source_manifest']==core_source(),'original context/outer/core provenance differs')
    result=dict(state='GEMMA_CONFIRMATORY96_SUPERVISOR_INDEPENDENT_TERMINAL_V4',request=REQUEST,
        dispatch_request=DISPATCH_REQUEST,epoch=epoch,source_manifest=source,core_source_manifest=core_source(),
        outcome=opin,passed=False,planned_each=96,planned_total=288,actual_updates=0,blind_gold_read=False,dev_inputs_read=False,
        Model_GO=False,formal17=False)
    try:
        native,npin=stable_json(ROOT/'native-outcome.json',out['native_outcome'])
        fresh=parse_accounting(subprocess.check_output(accounting_argv(),text=True,timeout=5))
        actual_steps=[]
        phase_rows={}
        phase_pins={}
        accounting_pins={}
        for phase in ('pre','gpu','post'):
            entry=out['results'][phase];raw,accounting_pins[phase]=stable_json(ROOT/(phase+'-accounting.json'))
            row=entry['terminal']['row']
            need(raw==entry and (entry['passed'] is True or phase=='gpu') and entry['accounting_columns']==COLUMNS and
                 entry['terminal']['terminal_visible'] is True and numbered(row) and row[1]==LABEL+'-'+phase and
                 ((phase=='gpu' and row[2] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY')
                   and (entry['passed'] is True or out['engineering_completed'] is False))
                  or (phase!='gpu' and row[2:4]==['COMPLETED','0:0']))
                 and row[5]==NODE and resources_match(phase,row[6]) and
                 (entry['returncode']==0 or phase=='gpu') and entry['held']['reaped'] is True and entry['held']['termination_needed'] is False,
                 'actual original C0/allocated count/held accounting missing')
            step_intent,phase_pins[phase]=stable_json(ROOT/(phase+'-step-intent.json'))
            need(step_intent['request_id']==REQUEST and step_intent['epoch']==epoch
                 and step_intent['owner_intent']==ipin and step_intent['phase']==phase
                 and step_intent['original_label']==LABEL+'-'+phase
                 and step_intent['GPU_cards']==(1 if phase=='gpu' else 0) and accounting_window(row,step_intent)
                 and exact_label(fresh,phase)==row,'original step intent or independent nine-column accounting differs')
            phase_rows[phase]=row
            actual_steps.append(row[0])
        need(len(set(actual_steps))==3 and phase_rows['pre'][8]<=phase_rows['gpu'][7]
             and phase_rows['gpu'][8]<=phase_rows['post'][7],'distinct ordered original three steps required')
        before,bpin=stable_json(ROOT/'before.json',out['before']);after,apin=stable_json(ROOT/'after.json',out['after'])
        pre_admitted,pre_admission_pin=stable_json(ROOT/'pre-admission.json')
        need(pre_admitted['request_id']==REQUEST and pre_admitted['epoch']==epoch
             and pre_admitted['row']==phase_rows['pre'] and pre_admitted['accounting_columns']==COLUMNS
             and pre_admitted['before']==bpin and pre_admitted['owner_intent']==ipin
             and before['owner_intent']==after['owner_intent']==ipin
             and before['phase_intent']==phase_pins['pre'] and after['phase_intent']==phase_pins['post']
             and pre_admitted['phase_intent']==phase_pins['pre']
             and pre_admitted['stored_pre_accounting']==accounting_pins['pre'],
             'independent physical snapshots and pre admission must bind this scope')
        gpu_host,gpin=stable_json(ROOT/'gpu-host-before.json');entry,_=stable_json(ROOT/'gpu-step-entry.json')
        admission,adpin=stable_json(ROOT/'admission.json');owner,ownerpin=stable_json(ROOT/'supervisor-owner.json')
        need(before['request_id']==after['request_id']==entry['request_id']==REQUEST and
             before['step']==actual_steps[0] and gpu_host['step']==entry['step']==admission['step']==actual_steps[1] and
             after['step']==actual_steps[2] and entry['intent']==ipin and
             before['pci']==gpu_host['pci']==after['pci']==admission['physical_PCI'] and len(set(before['pci']))==8,
             'actual fresh eight-physical PCI/step context differs')
        need(gpu_host["actual_step_resources"]==dict(step=actual_steps[1],node=NODE,one_DCU=True,cpu=8,mem_MiB=64000)
             and gpu_host["host_cwd"]=="/","actual GPU controller and compute launch context required")
        host=socket.gethostname();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        need(before['boot_id']==gpu_host['boot_id']==after['boot_id']==admission['boot_id']==owner['boot_id']==boot and
             native['supervisor']==entry['supervisor']==owner and native['request_id']==REQUEST and native['epoch']==epoch,
             'actual same-host/boot native context differs')
        need(all(x['kfd_holder_count']==0 and x['full_fd_inventory_performed'] is True for x in (before,gpu_host,after)) and
             after['owned_processes_remaining']==[] and after['master_port_available'] is True and after['master_port']==PORT,
             'independent actual HOST FD/KFD/port cleanup incomplete')
        need(admission['request']==REQUEST and admission['source_manifest']==core_source() and
             admission['operational_source_manifest']==source and admission['actual_visible_devices']==admission['allocated_DCU']==1 and
             admission['epoch']==epoch and epoch<admission['deadline']<=epoch+INSIDE-10 and
             0<=admission['monotonic_ns']-gpu_host['monotonic_ns']<=45_000_000_000 and admission['host']==host and
             admission['physical_GPU_count']==8 and admission['selected_PCI'] in before['pci'] and
             admission['assigned_device']['pci']==admission['selected_PCI'] and admission['assigned_device']['gpu_id']>0 and
             admission['actual_GPU_step_host_snapshot']==gpin and admission['fresh_exact_GPU_PCI_KFD_port_scope_satisfied'] is True,
             'actual assigned-one admission differs')
        if native['passed'] is not True:
            from failed_seal import consume_failure
            partial=consume_failure(native,source,epoch,actual_steps)
            result.update(failed_prediction_audit=partial,post_scope_satisfied=True,actual_steps=actual_steps)
            save('independent-read-result.json',result)
            return result
        need(phase_rows['gpu'][2:4]==['COMPLETED','0:0'], 'successful prediction requires exact GPU C0')
        need(native['passed'] is True and native['returncode']==0 and native['cleanup_errors']==[] and
             native['held_cleanup']['reaped'] is True and native['held_cleanup']['termination_needed'] is False and
             native['tree_cleanup']['owned_processes_remaining']==0 and native['tree_cleanup']['termination_needed'] is False,
             'actual native held/tree incomplete')
        state=original_absent(owner,host,boot)
        resources=native['resources']
        need(resources['samples']>0 and resources['hard_limit'] is False and resources['soft_limit_exceeded'] is False,
             'RSS evidence missing or exceeded')
        binding,binding_pin=stable_json(ASSET/'effect-binding.json')
        ready,_=stable_json(ROOT/'ready-binding.json')
        attempt,_=stable_json(ROOT/'native-attempt.json')
        need(ready['core_source']==core_source() and ready['original_binding']==attempt['ready_binding']==binding_pin
             and attempt['request_id']==REQUEST and attempt['epoch']==epoch and binding['request']==REQUEST,
             'exact same owner result and native input binding required')
        from asset_witness import INDEPENDENT_SHA, OWNER_SHA
        sys.path.insert(0,str(CORE))
        from blind_inputs import CURRENT_INPUT_PIN as INPUT_PIN
        need(binding['input_manifest']==INPUT_PIN and binding['independent_asset_witness']['independent_SHA256']==INDEPENDENT_SHA
             and binding['independent_asset_witness']['owner_objects']['outcome.json']['SHA256']==OWNER_SHA,
             'independent read lacks exact original .3538 evidence')
        # This normalized record is derived only from the actual above original metadata.
        # Frozen core consumer then independently checks raw108/files/EOS once, no gold.
        normalized=dict(request=REQUEST,source_manifest=core_source(),operational_source_manifest=source,
            epoch=epoch,deadline=admission['deadline'],whole_deadline=epoch+WHOLE,
            accounting_C0=True,owned_cleanup_passed=True,post_scope_satisfied=True,
            strong_same_host_boot_owner_terminal=True,original_keeper_state=state,blind_gold_read=False,
            generation_finalized=native['generation_finalized'],actual_steps=actual_steps,binding=binding_pin,
            allocated_DCU=1,physical_node_GPU_count=8,actual_visible_devices=1,selected_PCI=admission['selected_PCI'],
            originals=dict(outcome=opin,native_outcome=npin,pre=bpin,post=apin,admission=adpin,owner=ownerpin),
            RSS_sampled_peak_bytes=resources['sampled_group_peak_RSS_bytes'],
            RSS_unknown_coverage_samples=resources.get('RSS_unknown_coverage_samples'),RSS_hard_limit=False,
            allFDclean_claimed=False,exclusive_GPU_claimed=False)
        op_path=ASSET/'operational-independent-terminal.json';op_pin=save_at(op_path,normalized)
        sys.path.insert(0,str(CORE))
        spec=importlib.util.spec_from_file_location('frozen_core_original288_consumer',CORE/'read_terminal.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        inner=module.consume(op_path,op_pin,ASSET/'independent-terminal.json')
        need(inner['passed'] is inner['independent_passed'] is True,'original288 prediction/EOS seal independent fail')
        result.update(passed=True,independent_passed=True,all288sealed_before_gold=True,
            original288_terminal=pin(ASSET/'independent-terminal.json'),normalized_operational=op_pin,
            actual_steps=actual_steps,accounting_columns=COLUMNS,exact_accounting=phase_rows,pre_admission=pre_admission_pin,
            completed=sum(a['completed'] for a in inner['arms'].values()),
            missing=sum(a['missing'] for a in inner['arms'].values()),
            errors=sum(a['error'] for a in inner['arms'].values()),actual_allocated_DCU=1,physical_node_GPU_count=8,
            original_keeper_state=state,RSS_sampled_peak_bytes=resources['sampled_group_peak_RSS_bytes'],
            RSS_unknown_coverage_samples=resources.get('RSS_unknown_coverage_samples'),allFDclean_claimed=False)
    except BaseException as e:
        result['failure']=dict(type=type(e).__name__,private_reason=str(e)[:512])
        result['actual_arm_counts']=native.get('actual_arm_counts') if 'native' in locals() else None
    need(len(json.dumps(result).encode())<=16*1024,'bounded original terminal summary')
    save('independent-read-result.json',result);return result

def save_at(path,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(bytes=len(raw),SHA256=__import__('hashlib').sha256(raw).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=float,required=True);p.add_argument('--source-manifest-sha',required=True)
    a=p.parse_args();Clock(a.epoch).left(WHOLE,5)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original independent read45s')))
    signal.setitimer(signal.ITIMER_REAL,min(45,Clock(a.epoch).left(WHOLE,3)))
    r=consume(a.epoch,a.source_manifest_sha)
    print(json.dumps(r,sort_keys=True));raise SystemExit(0 if r['passed'] else 1)
