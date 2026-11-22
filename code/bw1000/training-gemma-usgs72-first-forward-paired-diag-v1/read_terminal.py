"""Independent metadata-only consumer on the original compute mount after terminal."""
import argparse
import json
import signal
import socket

from contract import *
from pair_compare import compare_pair


def consume(publication,epoch,mf_sha):
    source=verify('READ')
    need(ARM is None and socket.gethostname().split('.')[0]==NODE,'reader requires original zero-GRES compute mount')
    need(source['SHA256']==mf_sha,'exact published reader MF differs')
    intent,ipin=stable_json(PAIR_ROOT/'intent.json')
    need(intent['publication']==publication and intent['epoch']==epoch and intent['dispatch_request_id']==DISPATCH_REQUEST
         and intent['source_manifest']==source,'original diagnostic context differs')
    out,opin=stable_json(PAIR_ROOT/'outcome.json')
    summary=dict(state='GEMMA_FIRST_FORWARD_PAIRED_DIAGNOSTIC_TERMINAL_V1',dispatch_request_id=DISPATCH_REQUEST,
        original_training_request=REQUEST,publication=publication,epoch=epoch,source_manifest=source,outcome=opin,
        engineering_completed=False,release_permitted=False,diagnostic_passed=False,actual_updates=out.get('actual_updates'),
        observed_completed_updates_lower_bound=out.get('observed_completed_updates_lower_bound'),
        planned_native_processes=2,planned_updates_each=5,planned_new_updates=10,planned_input_loss_pairs=160,
        planned_state_pairs=40,planned_first4_RNG_pairs=64,original860_recovery_requalified=False,
        original_continuous25_replayed=False,Model_GO=False,formal17=False,effect_qualified=False)
    try:
        prior_continuous()
        native,npin=stable_json(PAIR_ROOT/'native-outcome.json',out['native_outcome'])
        need(native['passed'] is True and native['actual_updates']==10,'two native5 processes incomplete')
        handles=[]
        for phase in ('pre','gpu','post'):
            rec=out['results'][phase];row=rec['terminal']['row']
            need(rec['passed'] is True and rec['accounting_columns']==COLUMNS and numbered(row)
                 and row[1]==LABEL+'-'+phase and row[2:4]==['COMPLETED','0:0'] and row[5]==NODE
                 and allocated(row[6])==(8 if phase=='gpu' else 0),'actual three-step accounting incomplete')
            saved,_=stable_json(PAIR_ROOT/(phase+'-accounting.json'));need(saved==rec,'original receipt changed')
            handles.append(row[0])
        need(len(set(handles))==3,'actual phase handles not distinct')
        before,_=stable_json(PAIR_ROOT/'before.json',out['before'])
        after,_=stable_json(PAIR_ROOT/'after.json',out['after'])
        need(before['pci']==after['pci'] and len(set(before['pci']))==8
             and before['boot_id']==after['boot_id']==native['supervisor']['boot_id']
             and after['owned_processes_remaining']==[] and after['held_reaped'] is True
             and before['kfd_holder_count']==after['kfd_holder_count']==0
             and before['full_fd_inventory_performed'] is after['full_fd_inventory_performed'] is True
             and all(before['paired_ports_available'].values()) and before['paired_ports_available']==after['paired_ports_available'],
             'physical8/post/ports/owners incomplete')
        profiles={}
        for arm in ('untraced','traced'):
            folder=PAIR_ROOT/arm
            arm_native,_=stable_json(folder/'native-outcome.json',native['arms'][arm]['native_outcome'])
            lease,lpin=stable_json(folder/'lease-admission.json')
            need(lease['pci']==before['pci'] and lease['boot_id']==before['boot_id'] and lease['step']==handles[1]
                 and lease['master_port']==PORTS[arm] and lease['kfd_holder_count']==0
                 and lease['full_fd_inventory_performed'] is True,'each arm fresh full HOST admission differs')
            need(arm_native['passed'] is True and arm_native['actual_updates']==5
                 and arm_native['held_cleanup']['reaped'] is True and not arm_native['held_cleanup']['termination_needed']
                 and arm_native['tree_cleanup']['owned_processes_remaining']==0
                 and arm_native['tree_cleanup']['termination_needed'] is False,'native arm5/cleanup incomplete')
            for rank,pins in enumerate(arm_native['rank_receipts']):
                witness=folder/'train/resume-witness'/('rank-%02d'%rank)
                for name,key in [('result.json','result'),('owner.json','owner'),
                                 ('restored-before-RNG.json','restored_before_RNG'),('native-data-skip.json','native_data_skip')]:
                    stable_json(witness/name,pins[key],cap=8*1024**2)
                hooks,_=stable_json(folder/'diagnostic-hooks'/('rank-%02d.json'%rank))
                expected=arm=='traced' and rank in (0,1)
                need(hooks['common_complete'] is True and hooks['full_trace_enabled'] is expected
                     and (not expected or hooks['full_trace_complete'] is True),'actual selected trace/control schema differs')
                if expected:
                    trace=folder/'light-trace'/('rank-%02d'%rank)
                    report,_=stable_json(trace/'result-private.json')
                    need(report['complete'] is True and report['entries_attempted']==report['exits_completed']==4,
                         'four trace entry/exit denominator incomplete')
                    for record in report['records']:stable_json(trace/record['name'],{k:record[k] for k in ('bytes','SHA256')})
            cp,_=stable_json(folder/'checkpoint-25-files.json')
            need(cp==arm_native['checkpoints']['25'] and len(cp['files'])==14,'new native25 required14 inventory differs')
            for name,p in cp['files'].items():
                s=(folder/'train/checkpoint-25'/name).lstat()
                need([getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')]==p['file_identity'],
                     'new saved state identity changed since wholeSHA')
            profiles[arm]=dict(RSS_sampled_peak=arm_native['resources']['sampled_group_peak_RSS_bytes'],
                RSS_unknown_coverage=arm_native['resources'].get('RSS_unknown_coverage_samples',0),lease_admission=lpin,
                actual_updates=5,actual_native_forwards=160,full_light_trace=arm=='traced')
        producer,cpin=stable_json(PAIR_ROOT/'pair-comparison.json',out['pair_comparison'])
        independent=compare_pair(PAIR_ROOT);need(independent==producer,'independent pair comparison differs')
        save('independent-pair-comparison.json',independent)
        summary.update(engineering_completed=True,independent_passed=True,release_permitted=True,
            diagnostic_passed=independent['observed_noninterference_exact'],actual_updates=10,actual_native_processes=2,
            profiles=profiles,actual_steps=handles,GPU_steps=1,GPU_cards=8,
            same_independent_input_rng_qualified=independent['same_independent_input_rng_qualified'],
            compared_input_loss_pairs=independent['compared_input_loss_pairs'],
            compared_state_pairs=independent['compared_state_pairs'],compared_first4_RNG_pairs=independent['compared_first4_RNG_pairs'],
            difference_count=len(independent['differences']),differences_first32=independent['differences'][:32],
            maximum_absolute_loss_gap=independent['maximum_absolute_loss_gap'],pair_comparison=cpin,
            saved_state_tensors_independently_deserialized=False)
    except BaseException as error:
        summary['failure']=dict(type=type(error).__name__,reason=str(error)[:512])
    need(len(json.dumps(summary).encode())<=16*1024,'bounded independent summary')
    save('independent-terminal.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--epoch',required=True,type=float)
    p.add_argument('--source-manifest-sha',required=True);a=p.parse_args()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('read45s exhausted')))
    signal.setitimer(signal.ITIMER_REAL,45)
    r=consume(a.publication,a.epoch,a.source_manifest_sha);print(json.dumps(r,sort_keys=True,allow_nan=False))
    raise SystemExit(0 if r['diagnostic_passed'] else 1)
