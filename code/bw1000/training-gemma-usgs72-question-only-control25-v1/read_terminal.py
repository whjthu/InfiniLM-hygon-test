"""Read exactly this original terminal; no model, new device query or replay."""
import json
import math
from pathlib import Path
import signal
import socket
import time

from contract import *


def audit_rank(rank,expected,epoch):
    folder=ASSET/'continuous/train/continuous-witness'/('rank-%02d'%rank)
    result,rpin=stable_json(folder/'result.json',expected['result'],cap=8*1024**2)
    initial,ipin=stable_json(folder/'initialization.json',expected['initialization'])
    owner,opin=stable_json(folder/'owner.json',expected['owner'])
    need(result['rank']==rank and result['world_size']==8 and result['completed_global_step']==25
         and len(result['inputs'])==len(result['outputs'])==100 and len(result['steps'])==25
         and result['saved_checkpoints']==[10,20,25],'rank full denominator failed')
    for i,(inp,out) in enumerate(zip(result['inputs'],result['outputs']),1):
        need(inp['ordinal']==out['ordinal']==i and inp['global_step']==(i-1)//4
             and inp['full_input']['finite'] is True and out['loss']['finite'] is True
             and type(out['loss_scalar']) in (int,float) and math.isfinite(out['loss_scalar']),
             'input/finite loss/accumulation record incomplete')
    for step,rec in enumerate(result['steps'],1):
        need(rec['global_step']==step and rec['optimizer_slot_count']==476
             and type(rec['adapter_changed']) is bool and type(rec['zero_learning_rate']) is bool,
             'actual native optimizer counter/parameter record incomplete')
    need(any(r['adapter_changed'] for r in result['steps']),'full25 never changed parameters')
    n=initial['named']
    need(initial['mode']=='TARGET_CREATED_DOMAIN_INITIAL_REUSE' and initial['original_A100_initial_read'] is False
         and initial['RNG_equal_domain_reference'] is True and initial['parameters_equal_domain_reference'] is True and initial['target_created'] is True
         and initial['original_A100_value_equivalence'] is False
         and n['finite'] is True and n['tensors']==476 and n['elements']==29802496,
         'actual fresh initial trainables/RNG observation incomplete')
    need(owner['rank']==owner['local_rank']==rank and owner['PID']>0 and owner['startticks']>0
         and owner['epoch']==epoch and owner['deadline']==epoch+INSIDE,
         'rank original owner identity incomplete')
    need(n['SHA256']=='e6b624e3342df4d84751be5b65c2722137ff17109f2dfb18db8c5fd4c916b98c',
         'control initial values differ from actual domain endpoint')
    tokens=[row.get('tokens') for row in result['inputs']]
    need(all(isinstance(t,dict) and type(t.get('shifted_supervised_tokens')) is int and
             t['shifted_supervised_tokens']>0 and type(t.get('padded_input_tokens')) is int and
             t['padded_input_tokens']>0 for t in tokens),'actual native token exposure metadata absent')
    return dict(rank=rank,inputs=100,updates=25,result=rpin,initialization=ipin,owner=opin,
                fresh_named_SHA256=n['SHA256'],parameter_changing_updates=sum(r['adapter_changed'] for r in result['steps']),
                actual_padded_input_tokens=sum(t['padded_input_tokens'] for t in tokens),
                actual_shifted_supervised_tokens=sum(t['shifted_supervised_tokens'] for t in tokens))


def consume(publication,epoch,source_manifest_sha):
    need(len(publication)==40 and set(publication)<=set('0123456789abcdef'),'publication format')
    source=verify('READ');need(source['SHA256']==source_manifest_sha,'reader published operative source differs')
    need(socket.gethostname().split('.')[0]==NODE,
         'terminal reader must run zero-GRES on original compute mount, not compare login HOST st_dev')
    intent,ipin=stable_json(ROOT/'intent.json')
    need(intent['publication']==publication and intent['epoch']==epoch and intent['request_id']==REQUEST
         and intent['dispatch_request_id']==DISPATCH_REQUEST and intent['source_manifest']==source,
         'actual original execution context differs')
    out,opin=stable_json(ROOT/'outcome.json')
    native,npin=(stable_json(ROOT/'native-outcome.json') if (ROOT/'native-outcome.json').exists() else (None,None))
    summary=dict(state='GEMMA_QUESTION_ONLY_CONTROL25_INDEPENDENT_TERMINAL_V1',request_id=REQUEST,dispatch_request_id=DISPATCH_REQUEST,
        publication=publication,epoch=epoch,source_manifest=source,outcome=opin,native_outcome=npin,
        passed=False,independent_passed=False,planned_updates=25,actual_updates=out.get('actual_updates'),
        planned_exposures=800,unique_train_QA=72,train_media=0,resume_invoked=False,
        Model_GO=False,formal17_GO=False,effect_GO=False,original_A100_initial_read=False)
    try:
        need(native is not None,'native outcome absent; preserve original pre/native failure, no replay')
        need(out['request_id']==native['request_id']==REQUEST and out['epoch']==native['epoch']==epoch
             and out['publication']==publication and out['source_manifest']==source
             and out['native_outcome']==npin,'original output linkage differs')
        handles=[]
        for phase in ('pre','gpu','post'):
            p=out['results'][phase]
            row=p['terminal']['row']
            need(p['passed'] is True and p['terminal']['terminal_visible'] is True
                 and p['accounting_columns']==COLUMNS and numbered(row)
                 and row[1]==LABEL+'-'+phase and row[2:4]==['COMPLETED','0:0']
                 and row[5]==NODE and allocated(row[6])==(8 if phase=='gpu' else 0)
                 and p['returncode']==0 and p['held']['reaped'] is True
                 and p['held']['termination_needed'] is False,'actual phase/accounting/held incomplete')
            saved,_=stable_json(ROOT/(phase+'-accounting.json'))
            need(saved==p,'original phase receipt differs from controller summary')
            handles.append(row[0])
        need(len(set(handles))==3,'pre/GPU/post must be distinct original steps')
        before,bpin=stable_json(ROOT/'before.json',out['before'])
        after,apin=stable_json(ROOT/'after.json',out['after'])
        entry,epin=stable_json(ROOT/'gpu-step-entry.json')
        need(before['step']==handles[0] and entry['step']==handles[1] and after['step']==handles[2]
             and entry['intent']==ipin and before['request_id']==entry['request_id']==after['request_id']==REQUEST,
             'original device/entry context incomplete')
        need(before['pci']==after['pci'] and len(set(before['pci']))==8
             and before['boot_id']==after['boot_id']==native['supervisor']['boot_id']
             and before['kfd_holder_count']==after['kfd_holder_count']==0
             and before['full_fd_inventory_performed'] is after['full_fd_inventory_performed'] is True
             and before['master_port']==after['master_port']==PORT
             and before['master_port_available'] is after['master_port_available'] is True
             and after['owned_processes_remaining']==[],'actual independent8PCI/KFD/port/post incomplete')
        need(native['passed'] is True and native['returncode']==0 and native['actual_updates']==25
             and not native['cleanup_errors'] and native['held_cleanup']['reaped'] is True
             and native['held_cleanup']['termination_needed'] is False
             and native['tree_cleanup']['owned_processes_remaining']==0
             and native['tree_cleanup']['termination_needed'] is False,'native25 or independent cleanup incomplete')
        ranks=[audit_rank(rank,native['rank_receipts'][rank],epoch) for rank in range(8)]
        need(len({r['fresh_named_SHA256'] for r in ranks})==1,'actual rank initial cohort differs')
        cps={}
        for step in ('10','20','25'):
            record,cpa=stable_json(ROOT/('checkpoint-'+step+'-files.json'))
            need(record==native['checkpoints'][step] and record['step']==int(step)
                 and len(record['files'])==14,'native required14 checkpoint inventory incomplete')
            for name,expected in record['files'].items():
                s=(ASSET/'continuous/train'/('checkpoint-'+step)/name).lstat()
                need([getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')]
                     ==expected['file_identity'],'saved native checkpoint identity changed since recorded whole SHA')
            cps[step]=cpa
        resources=native['resources']
        need(resources['samples']>0 and resources['hard_limit'] is False
             and resources['soft_limit_exceeded'] is False,'sampled RSS resource evidence missing')
        summary.update(passed=True,independent_passed=True,actual_updates=25,arm='continuous',ranks_completed=8,
            microbatches_per_rank=100,actual_exposures=800,rank_receipts=ranks,
            fresh_initial_named_SHA256=ranks[0]['fresh_named_SHA256'],accounting_C0=True,
            owned_cleanup_passed=True,actual_steps=handles,checkpoint_manifests=cps,
            checkpoint20_files={n:{k:v[k] for k in ('bytes','SHA256')} for n,v in native['checkpoints']['20']['files'].items()},
            RSS_sampled_group_peak_bytes=resources['sampled_group_peak_RSS_bytes'],
            RSS_unknown_coverage_samples=resources.get('RSS_unknown_coverage_samples'),
            RSS_method=resources['method'],RSS_hard_limit=False,
            mathematical_saved_state_deserialization_independent=False,
            current_GPU_owner_liveness='NOT_REQUERIED; actual same-host independent post strongly bound',
            endpoint_host=socket.gethostname(),initialization_mode='TARGET_CREATED_DOMAIN_INITIAL_REUSE',
            actual_padded_input_tokens=sum(r['actual_padded_input_tokens'] for r in ranks),
            actual_shifted_supervised_tokens=sum(r['actual_shifted_supervised_tokens'] for r in ranks),
            equal_token_exposure_to_domain_claimed=False,domain_actual_supervised_token_count='UNKNOWN_NOT_IN_ORIGINAL_COUNTER',
            caption_format='original unmodified; domain FINAL format differs')
    except BaseException as error:
        summary.update(failure=dict(type=type(error).__name__,reason=str(error)[:512]),
                       native_failure=native.get('failure') if native else None,controller_failure=out.get('error'))
    need(len(json.dumps(summary).encode())<=16*1024,'bounded terminal summary')
    save('independent-terminal.json',summary)
    if summary['passed']:
        target=ASSET/'control-independent-terminal.json'
        with target.open('x',encoding='utf8') as f:json.dump(summary,f,sort_keys=True,allow_nan=False)
    return summary


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--epoch',type=float,required=True)
    p.add_argument('--source-manifest-sha',required=True);a=p.parse_args()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('terminal readonly45s exhausted')))
    signal.setitimer(signal.ITIMER_REAL,45)
    r=consume(a.publication,a.epoch,a.source_manifest_sha)
    print(json.dumps(r,sort_keys=True,allow_nan=False))
    raise SystemExit(0 if r['passed'] else 1)
