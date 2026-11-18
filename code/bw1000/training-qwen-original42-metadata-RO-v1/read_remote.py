"""Read only the original42 small receipts; never replay original code."""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import socket
import stat
import subprocess
import time


def main(binding,context):
    assert binding['root_GO'] is True and binding['RO_GO'] is True
    assert binding['GPU_GO'] is False and binding['replay_GO'] is False
    epoch=context['epoch'];deadline=epoch+45
    assert 0<=time.time()-epoch<40
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
    allowed=sorted(os.sched_getaffinity(0)-{0,1});assert len(allowed)>=2;os.sched_setaffinity(0,allowed[:2])
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    H=lambda b:hashlib.sha256(b).hexdigest()
    identify=lambda s:[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    root=Path(binding['namespace']);assert not root.is_symlink()
    result=dict(context=context,original_request=binding['original_request'],namespace_exists=root.exists(),
                observer_host=socket.gethostname(),observer_boot_SHA256=H(Path('/proc/sys/kernel/random/boot_id').read_bytes()),
                records={},explicit_file_bytes=0,accounting_columns=binding['accounting_columns'],accounting={},
                original_state='UNKNOWN',process_state='UNKNOWN',replayed=False,raw_profile_files_read=0,
                native_files_read=0,new_Slurm_steps=0,GPU_calls=0,release_permitted=False)
    values={}
    for name,cap in binding['files'].items():
        record=dict(state='UNKNOWN');result['records'][name]=record
        try:
            path=root/name;s=path.lstat()
            assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid()
            if s.st_size>cap:
                record.update(state='OVERSIZE_NOT_READ',stat_bytes=s.st_size,read_cap=cap);continue
            assert result['explicit_file_bytes']+s.st_size<=208*1024
            with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb',buffering=0) as stream:
                before=os.fstat(stream.fileno());raw=stream.read(s.st_size);after=os.fstat(stream.fileno())
            result['explicit_file_bytes']+=len(raw)
            assert len(raw)==s.st_size and identify(s)==identify(before)==identify(after)==identify(path.stat())
            values[name]=json.loads(raw);record.update(state='PRESENT_STABLE',bytes=len(raw),SHA256=H(raw))
        except FileNotFoundError:record['state']='MISSING_NOT_NONEXECUTION_PROOF'
        except Exception as exc:record.update(state='READ_FAILED',type=type(exc).__name__)
    intent=values.get('intent.json')
    bound=False
    if intent:
        bound=(intent.get('request_id')==binding['original_request'] and intent.get('source_manifest_SHA256')==binding['original_source_MF_SHA256']
               and re.fullmatch('[0-9a-f]{40}',str(intent.get('claim_commit',''))) is not None
               and intent.get('deadline')==intent.get('epoch',0)+7800
               and intent.get('resource')==dict(parent='1087656',node='a01r4n16',GPU_cards=8,CPU=64,memory_MiB=224000))
        result['intent_bound']=bound
        result['records']['intent.json']['value']={k:intent.get(k) for k in ('request_id','claim_commit','epoch','deadline','source_manifest_SHA256','installer_host','installer_boot_SHA256','resource')}
    for name in ('keeper-owner.json','controller-ready.json'):
        if name in values:
            value=values[name]
            result['records'][name]['value']={k:value.get(k) for k in ('PID','PGID','startticks','host','boot_id_SHA256','epoch','deadline','claim_commit','CPU_controller_alive','GPU_launched') if k in value}
    owner=values.get('keeper-owner.json',{})
    if bound and owner:
        same_host=owner.get('host')==result['observer_host'];same_boot=owner.get('boot_id_SHA256')==result['observer_boot_SHA256']
        result.update(same_original_host=same_host,same_original_boot=same_boot)
        if same_host and same_boot and owner.get('epoch')==intent['epoch'] and owner.get('claim_commit')==intent['claim_commit']:
            try:
                pid=owner['PID'];assert type(pid)is int and pid>0
                proc=Path('/proc')/str(pid);assert proc.stat().st_uid==os.getuid()
                ticks=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])
                result['process_state']='ORIGINAL_IDENTITY_LIVE' if ticks==owner['startticks'] else 'ORIGINAL_IDENTITY_ABSENT_PID_REUSED'
            except FileNotFoundError:result['process_state']='ORIGINAL_IDENTITY_ABSENT_SAME_HOST_BOOT'
            except Exception as exc:result['process_error_type']=type(exc).__name__
    outcome=values.get('outcome.json')
    if outcome:
        # Do not export embedded native/profiler/version/provider values.
        whitelisted={k:outcome.get(k) for k in ('claim_commit','epoch','deadline','GPU_step_requested','native_intent_present','keeper_passed','independent_model_qualified','formal_performance')}
        whitelisted['error_type']=(outcome.get('error') or {}).get('type')
        whitelisted['runs']=[{k:r.get(k) for k in ('name','GPU_cards','returncode','forced_cleanup','accounting_columns','terminal','passed')} for r in outcome.get('runs',[])]
        for key in ('before','after'):
            row=outcome.get(key)
            whitelisted[key]=None if not row else dict(node=row.get('node'),step=row.get('step'),
                PCI_count=len(row['pci']) if isinstance(row.get('pci'),list) else None,
                kfd_holder_count=row.get('kfd_holder_count'),
                owned_remaining_count=len(row['owned_processes_remaining']) if isinstance(row.get('owned_processes_remaining'),list) else None,
                field_coverage='PRESENT_LISTS' if isinstance(row.get('pci'),list) and isinstance(row.get('owned_processes_remaining'),list) else 'UNKNOWN',
                master_port_available=row.get('master_port_available'))
        result['records']['outcome.json']['value']=whitelisted
        result['outcome_context_bound']=bool(bound and outcome.get('claim_commit')==intent['claim_commit'] and outcome.get('epoch')==intent['epoch'])
    labels=binding['labels'];columns=','.join(binding['accounting_columns'])
    try:
        timeout=min(12,max(.001,deadline-time.time()-3))
        query=subprocess.run(['sacct','-n','-P','-j','1087656','--format='+columns],capture_output=True,timeout=timeout)
        result['sacct_returncode']=query.returncode
        assert len(query.stdout)<=512*1024 and query.returncode==0
        rows=[line.split('|') for line in query.stdout.decode().splitlines()]
        result['accounting']={label:[r for r in rows if len(r)==len(binding['accounting_columns']) and r[1]==label] for label in labels}
        if any(len(v)>1 for v in result['accounting'].values()):result['original_state']='UNKNOWN_DUPLICATE_EXACT_LABEL'
        elif bound and result.get('outcome_context_bound') and result['accounting'][labels[1]] and result['accounting'][labels[2]]:
            terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
            if all(v[0][2].split()[0] in terminal for v in result['accounting'].values() if v):
                result['original_state']='ORIGINAL_TERMINAL_METADATA_PRESENT_FULL_AUDIT_STILL_REQUIRED'
    except Exception as exc:result['accounting_error_type']=type(exc).__name__
    raw=json.dumps(result,separators=(',',':'),allow_nan=False).encode();assert len(raw)<=32768
    os.write(1,raw+b'\n')


if __name__=='__main__':
    main(BINDING,CONTEXT)
