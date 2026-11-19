"""Independent new-process metadata audit and once fresh exact new assembly step accounting."""
import argparse,json,time,re
from pathlib import Path
from common import P,need,sha,canonical
from dependency_gate import stable_json,validate_snapshot
D=Path(__file__).resolve().parent
KEYS=('publication','request_id','epoch','deadline','source_manifest_SHA256','dependency_snapshot_SHA256')

def audit(run,context):
    from verify_package import verify
    v=verify(D);need(v['READ_GO'] is True and v['MF_SHA256']==context['source_manifest_SHA256'],'ROOT_ADOPTED_EXACT_NEW_CONSUMER')
    r=dict(independent_metadata_verified=False,planned_files=17,planned_read3=3,planned_copy14=14,verified_files=0,asset_body_reads=0,old_qualification_tests=0,original_scope_reconciliations=0,Slurm_submissions=0,fresh_accounting=None,pins={},errors=[],GPU_GO=False,Model_GO=False,training_GO=False,all26_qualified=False,replay_allowed=False)
    def get(name):
        x,p=stable_json(run/name);need(all(x[k]==v for k,v in context.items()),'NEW_ORIGINAL_CONTEXT_'+name);r['pins'][name]=p;return x
    host=None
    try:
        local=get('outcome.json')
        if not local['exec_request_attempted']:
            need(local['execution_state']=='NOT_STARTED_BLOCKED_PREREQUISITE','NO_NEW_EXECUTION_OR_READY_FROM_FAILED_PREREQUISITE');r['blocked_before_new_SSH']=True;raise ValueError('FIXED_ORIGINAL3PASS_MISSING_FAILED_UNKNOWN_NO_EXECUTION_QUALIFICATION')
        host=get('host-terminal.json');worker=host.get('worker')
        if worker and re.fullmatch(re.escape(P['parent'])+r'\.\d+',worker.get('step','')):
            from fresh_accounting import query_once,COLS
            r['fresh_accounting']=query_once(run,context,dict(target_parent=P['parent'],label=P['label']),worker['step'])
        snapshot,pin=stable_json(run/'upstream-snapshot.json',32768);r['pins']['upstream-snapshot.json']=pin;validate_snapshot(snapshot);need(sha(canonical(snapshot))==context['dependency_snapshot_SHA256'],'EXACT_FIXED_ORIGINAL_PASS_SNAPSHOT')
        need(local['passed'] is host['passed'] is worker['passed'] is True,'NO_FAILED_PRODUCER_PROMOTION');need(local['transport']['returncode']==0 and local['transport']['held_SSH_reaped'] is True and local['transport']['forced_local_SSH_kill'] is local['transport']['timed_out'] is False,'NORMAL_NEW_HELD_SSH')
        need(worker['deadline']==context['epoch']+900 and worker['finished_epoch']<=context['epoch']+800 and host['finished_epoch']<=context['epoch']+865 and worker['work_deadline']<=context['epoch']+760,'ONE900_BUDGET_NO_EXTENSION')
        need(worker['owner']['UID']==154459 and worker['owner']['host']==P['node'] and worker['allocated_CPU']==4 and worker['worker_CPU']==2 and len(worker['CPU_affinity'])==2 and not set(worker['CPU_affinity'])&{0,1} and worker['AS_limit_bytes']==4294967296 and worker['nice']>=10,'ONE_ACTUAL_COMPUTE_OBSERVER_ALLOC4_WORKER2')
        need(worker['dependency_snapshot_SHA256']==context['dependency_snapshot_SHA256'] and worker['planned_files']==17 and worker['completed_read3']==3 and worker['completed_copy14']==14,'FULL17_FAILURE_DENOMINATOR')
        c=worker['counters'];need(c['read3_bytes']==P['read3_bytes'] and c['copy14_source_read_bytes']==P['copy14_bytes'] and c['total_read_bytes']==c['read_hash_bytes']==17072712250 and c['total_write_bytes']==c['output_hash_bytes']==4922323586,'ALL_EXPLICIT_ASSET_IO_BYTES')
        for rows,expected,kind in [(worker['verified_read3'],P['read3'],'read3'),(worker['verified_copy14'],P['copy14'],'copy14')]:
            need(len(rows)==len(expected),'EXACT_'+kind+'_COUNT')
            for row,pin in zip(rows,expected):
                need(all(row[k]==pin[k] for k in ['name','bytes','SHA256']) and row['observer']==worker['owner'],'EACH_ORIGINAL_WHOLE_SHA_SAME_HOST_BOOT_OWNER')
                if kind=='read3':need(row['stable_FD_path'] is row['whole_SHA_qualified'] is True and row['source_stat'][2]==row['bytes'],'NEW3_FULL_SINGLE_OBSERVER_READ')
                else:need(row['source_stable_FD_path'] is row['target_stable_FD_path'] is row['file_fsync'] is row['pre_rename_directory_fsync'] is row['directory_fsync'] is row['new_exclusive_inode'] is True and row['source_stat'][2]==row['target_stat'][2]==row['bytes'] and row['source_stat'][:2]!=row['target_stat'][:2],'NEW14_COPY_FILE_DIR_FSYNC_RENAME_STABLE_FD')
        need(worker['buffer_max_bytes']==1048576 and all(worker[k]==0 for k in ['old_partial_accesses','initial8_accesses','reference_accesses','hardlinks','tensor_decodes','GPU_calls','output_body_rereads']),'NO26_SCAN_OTHER_ASSET_OR_MODEL_ACCESS')
        a=host['accounting'];need(a[0]==worker['step'] and a[1]==P['label'] and a[2:4]==['COMPLETED','0:0'] and a[5]==P['node'] and set(a[6].split(','))=={'cpu=4','mem=4G','node=1'},'ACTUAL_NEW_CPU_C0_RESOURCES')
        need(host['held_srun']['reaped'] is True and host['held_srun']['termination_needed'] is False and host['held_srun']['returncode']==0,'HELD_NEW_SRUN_NORMAL_REAP')
        fresh=r['fresh_accounting'];need(fresh and fresh['passed'] is True and fresh['actual_ordinary_SSH_attempts']==fresh['actual_sacct_calls']==1 and fresh['record']['actual_row']==a and fresh['record']['requested_label']==P['label'] and fresh['record']['columns']==host['accounting_columns']==COLS,'INDEPENDENT_ONCE_FRESH_EXACT_NEW_STEP_SACCT_EQUALS_PRODUCER')
        need(time.time()<=context['epoch']+P['budget']['consumer_closure'],'INDEPENDENT_CLOSURE_WITHIN_ORIGINAL900')
        for name in ['fresh-accounting-attempt.json','fresh-accounting-result.json']:
            _,pin=stable_json(run/name);r['pins'][name]=pin
        r.update(independent_metadata_verified=True,verified_files=17,verified_read_bytes=17072712250,verified_write_bytes=4922323586,observer=worker['owner'],proof_boundary='Only17 model files in one compute observer: three fresh wholeSHA/read+14 copy/hash/fsync/stableFD, source paths exact, original fixed3PASS gate, newstep freshactualaccounting; no input8/26 bundle/model/native/training or performance qualification')
    except (OSError,ValueError,KeyError,TypeError,IndexError) as e:r['errors'].append(dict(type=type(e).__name__,reason=str(e)[:180]))
    return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--context',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();intent,_=stable_json(a.context);context={k:intent[k] for k in KEYS};result=audit(a.run,context)
    with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ['independent_metadata_verified','planned_files','verified_files','errors']}));raise SystemExit(0 if result['independent_metadata_verified'] else 1)
