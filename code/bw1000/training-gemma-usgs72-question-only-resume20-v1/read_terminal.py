"""Read exactly this original terminal; no model, new device query or replay."""
import json
import math
from pathlib import Path
import signal
import socket
import time

from contract import *
from compare_recovery import compare


def audit_rank(rank,expected):
    folder=ASSET/'resumed/train/resume-witness'/('rank-%02d'%rank)
    result,rpin=stable_json(folder/'result.json',expected['result'],cap=8*1024**2)
    owner,opin=stable_json(folder/'owner.json',expected['owner'])
    restored,restored_pin=stable_json(folder/'restored-before-RNG.json',expected['restored_before_RNG'])
    skip,skip_pin=stable_json(folder/'native-data-skip.json',expected['native_data_skip'])
    need(result['rank']==rank and result['world_size']==8 and result['completed_global_step']==25
         and len(result['inputs'])==len(result['outputs'])==20
         and [r['global_step'] for r in result['steps']]==[21,22,23,24,25]
         and result['saved_checkpoints']==[25] and result['native_resume_window_complete'] is True,
         'resumed full5 denominator failed')
    for i,(inp,out) in enumerate(zip(result['inputs'],result['outputs']),1):
        need(inp['ordinal']==out['ordinal']==i and inp['global_step']==20+(i-1)//4
             and inp['full_input']['finite'] is True and out['loss']['finite'] is True
             and type(out['loss_scalar']) in (int,float) and math.isfinite(out['loss_scalar']),
             'resumed input/finite loss/accumulation record incomplete')
    need(len(result['RNG_loads'])==1 and result['RNG_loads'][0]['saved_equal'] is True
         and result['RNG_loads'][0]['global_step']==20
         and restored['restored_parameter_optimizer_scheduler_equal'] is True
         and skip==dict(rank=rank,before=100,skipped=80,after=20),
         'native checkpoint20 full state/RNG/cursor restore incomplete')
    for row in result['steps']:
        need(set(row['state'])=={'adapter','optimizer','scheduler'}
             and all(v['finite'] is True for v in row['state'].values()),'resumed update state incomplete/nonfinite')
    need(any(row['state']['adapter']!=restored['actual']['adapter'] for row in result['steps']),
         'resumed window never changed parameters')
    need(owner['rank']==owner['local_rank']==rank and owner['PID']>0 and owner['startticks']>0,
         'resumed rank original owner identity incomplete')
    return dict(rank=rank,inputs=20,updates=5,result=rpin,owner=opin,restored_before_RNG=restored_pin,native_data_skip=skip_pin)


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
    summary=dict(state='GEMMA_RESUME20TO25_INDEPENDENT_TERMINAL_V1',request_id=REQUEST,dispatch_request_id=DISPATCH_REQUEST,
        publication=publication,epoch=epoch,source_manifest=source,outcome=opin,native_outcome=npin,
        passed=False,independent_passed=False,planned_updates=5,original_continuous_updates=25,actual_updates=out.get('actual_updates'),
        planned_exposures=160,unique_train_QA=72,train_media=12,resume_invoked=True,release_permitted=False,exact_paired_recovery=None,
        Model_GO=False,formal17_GO=False,effect_GO=False,original_A100_initial_read=False)
    try:
        need(native is not None,'resume outcome absent; preserve original pre/native failure, no replay')
        need(out['request_id']==native['request_id']==REQUEST and out['epoch']==epoch
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
        need(native['passed'] is True and native['returncode']==0 and native['actual_updates']==5
             and not native['cleanup_errors'] and native['held_cleanup']['reaped'] is True
             and native['held_cleanup']['termination_needed'] is False
             and native['tree_cleanup']['owned_processes_remaining']==0
             and native['tree_cleanup']['termination_needed'] is False,'native5 or independent cleanup incomplete')
        ranks=[audit_rank(rank,native['rank_receipts'][rank]) for rank in range(8)]
        cps={}
        for step in ('25',):
            record,cpa=stable_json(ROOT/('checkpoint-'+step+'-files.json'))
            need(record==native['checkpoints'][step] and record['step']==int(step)
                 and len(record['files'])==14,'native required14 checkpoint inventory incomplete')
            for name,expected in record['files'].items():
                s=(ASSET/'resumed/train'/('checkpoint-'+step)/name).lstat()
                need([getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')]
                     ==expected['file_identity'],'saved native checkpoint identity changed since recorded whole SHA')
            cps[step]=cpa
        resources=native['resources']
        need(resources['samples']>0 and resources['hard_limit'] is False
             and resources['soft_limit_exceeded'] is False,'sampled RSS resource evidence missing')
        prior_continuous()
        original_comparison,cpin=stable_json(ROOT/'pair-comparison.json',out['pair_comparison'])
        independent=compare(ASSET)
        need(independent==original_comparison,'independent full160/40 comparison differs from original metadata')
        save('independent-pair-comparison.json',independent)
        exact=independent['exact_paired_recovery'] is True
        summary.update(passed=exact,independent_passed=True,engineering_completed=True,
            release_permitted=True,exact_paired_recovery=exact,actual_updates=5,arm='resumed',ranks_completed=8,
            microbatches_per_rank=20,actual_exposures=160,rank_receipts=ranks,
            accounting_C0=True,owned_cleanup_passed=True,actual_steps=handles,checkpoint_manifests=cps,
            original_continuous_gate=CONTINUOUS_GATE_PIN,planned_tail_microbatch_pairs=160,
            compared_tail_microbatch_pairs=independent['compared_tail_microbatch_pairs'],
            planned_rank_update_pairs=40,compared_rank_update_pairs=independent['compared_rank_update_pairs'],
            maximum_absolute_loss_gap=independent['maximum_absolute_loss_gap'],
            difference_count=len(independent['differences']),differences_first32=independent['differences'][:32],
            paired_comparison=cpin,all8_restore_cursor='100-to-skip80-to20',
            RSS_sampled_group_peak_bytes=resources['sampled_group_peak_RSS_bytes'],
            RSS_unknown_coverage_samples=resources.get('RSS_unknown_coverage_samples',0),
            RSS_method=resources['method'],RSS_hard_limit=False,
            final_saved25_typed_values_independently_deserialized=False,
            current_GPU_owner_liveness='NOT_REQUERIED; actual same-host independent post strongly bound',
            endpoint_host=socket.gethostname())
    except BaseException as error:
        summary.update(failure=dict(type=type(error).__name__,reason=str(error)[:512]),
                       native_failure=native.get('failure') if native else None,controller_failure=out.get('error'))
    need(len(json.dumps(summary).encode())<=16*1024,'bounded terminal summary')
    save('independent-terminal.json',summary)
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
