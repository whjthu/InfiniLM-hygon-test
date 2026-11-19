"""Metadata-only newprocess consumer; never loads model/tensors or re-encodes old data."""
import argparse,json,os,signal,time
from pathlib import Path
from common import D,P,need,sha,obj,pinned,verify_source,save,canonical

def read(a):
    end=min(a.epoch+890,a.reader_origin+35);need(a.epoch<=a.reader_origin<=time.time()<end,'SAME_ORIGINAL900_READER');signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,end-time.time());need(set(P['CPU_affinity'])<=os.sched_getaffinity(0),'READER_FIXED_CPU');os.sched_setaffinity(0,P['CPU_affinity']);verify_source();need(P['root_GO'] is P['READ_GO'] is True and sha((D/'SOURCE_FILES.json').read_bytes())==a.source_manifest_sha256 and sha((D/'CLAIM.json').read_bytes())==a.claim_sha256,'EXACT_ADOPTED_READER_SOURCE');need(a.scope==D.parent==Path(P['namespace']),'EXACT_NEW_NAMESPACE');pins={};total=0
    def take(name):
        nonlocal total
        p=a.scope/name;s=p.stat();need(0<s.st_size<=2*1024**2,'SMALL_RECEIPT');raw=pinned(p,dict(bytes=s.st_size,SHA256=sha(p.read_bytes())));total+=len(raw);need(total<=16*1024**2,'METADATA_AGGREGATE');pins[name]=dict(bytes=len(raw),SHA256=sha(raw));return json.loads(raw)
    intent=take('intent.json');need(all(intent[k]==getattr(a,k) for k in ['request_id','publication','epoch']) and intent['deadline']==a.epoch+900 and intent['source_manifest_SHA256']==a.source_manifest_sha256,'ORIGINAL_CONTEXT')
    outcome=take('outcome.json');need(all(outcome[k]==intent[k] for k in ['request_id','publication','epoch','deadline','source_manifest_SHA256','claim_SHA256']),'ORIGINAL_PRODUCER_CONTEXT');owner=outcome['owner'];need(owner['host']==os.uname().nodename and owner['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and owner['uid']==os.getuid(),'SAME_HOST_BOOT_UID')
    try:
        fields=(Path('/proc')/str(owner['PID'])/'stat').read_text().rsplit(') ',1)[1].split();need(int(fields[19])!=owner['startticks'] or fields[0]=='Z','ORIGINAL_OWNER_STILL_LIVE')
    except (FileNotFoundError,ProcessLookupError):pass
    r=dict(intent,metadata_verified=True,training_numerical_qualified=False,producer_failure_retained=outcome['passed'] is not True,planned_updates=32,planned_microbatches=128,completed_updates=outcome['completed_updates'],completed_microbatches=outcome['completed_microbatches'],new_native_calls=0,raw_tensor_or_weight_reads=0,old_RSS_FAILURE_preserved=True,Model_GO=False,Hygon_GPU_GO=False,formal17=False)
    if outcome['passed'] is not True:return dict(r,original_metadata_pins=pins,actual_metadata_bytes=total)
    need(outcome['completed_updates']==32 and outcome['completed_microbatches']==128 and outcome['failure'] is None and outcome['post']['scope_admission_satisfied'] is True,'ALL_PHASES_AND_POST_REQUIRED')
    post=take('post.json');need(post==outcome['post'] and post['active_compute_processes']==0,'FRESH_POST_NO_MODEL')
    clean=take('cleanup.json');need(clean==outcome['cleanup'] and len(clean['phases'])==3,'ALL3_CLEANUP')
    for x in clean['phases']:
        need(x['held']['reaped'] is True and x['held']['termination_needed'] is False and x['tree']['owned_processes_remaining']==0 and x['tree']['termination_needed'] is False,'NORMAL_OWNED_PHASE_TERMINAL')
    need(clean['tail']['held']['reaped'] is True and clean['tail']['tree']['owned_processes_remaining']==0 and not any(x.endswith('_error') for x in clean['tail']),'TAIL_CLEAN')
    for item in P['phases']:
        name=item['name'];v=take(name+'/receipt.json');progress=take(name+'/progress.json');micro=take(name+'/microbatches.json');cursor=take(name+'/cursor.json')
        need(all(v[k]==intent[k] for k in ['request_id','publication','epoch','deadline']) and v['phase']==name and v['passed'] is True and v['updates']==item['updates'] and v['microbatches']==item['microbatches'],'ACTUAL_PHASE_DENOMINATOR')
        need(progress['completed_microbatches']==item['microbatches'] and progress['completed_updates']==item['updates'] and not progress['failed'] and len(micro['inputs'])==len(micro['forwards'])==len(micro['losses'])==item['microbatches'] and micro['steps']==list(range(item['start']+1,item['end']+1)),'ALL128_INPUTS_LOSS_STEPS')
        need(cursor['loader_length']==72 and cursor['updates_per_epoch']==18 and cursor['trained_epochs']==item['start']//18 and cursor['skipped_microbatches']==item['start']%18*4,'NATIVE_ACTUAL72_CURSOR')
        if name=='continuous':
            need(v['saved_steps']==[10,20,25] and v['capture_files']==17,'ALL_INITIAL_FIRST8_CAPTURE');cp=take(name+'/capture-pins.json');need(len(cp['files'])==17 and 0<cp['bytes']<=P['capture_cap_bytes'],'CAPTURE_FULL_DENOMINATOR')
            for step in [10,20,25]:
                st=take(name+'/checkpoint-%d-files.json'%step);need(len(st['files'])==7 and 'rng_state.pth' in st['files'],'ALL_SINGLE_RANK_STATE')
        elif name=='replay2':
            comparisons=take(name+'/numerical-comparisons.json');need(v['numerical_exact'] is True and len(comparisons)==v['numerical_comparisons']==16 and all(x['exact'] is True for x in comparisons),'ALL16_NUMERICAL_COMPARISONS')
        else:
            st=take(name+'/checkpoint-25-files.json');rr=take(name+'/recovery.json');need(len(st['files'])==7 and v['recovery_exact'] is True and rr['saved_RNG_exact'] is True and all(rr[n]['exact'] is True for n in ['adapter_model.safetensors','optimizer.pt','scheduler.pt']),'FULL_ADAPTER_OPTIM_SCHED_RNG_RECOVERY')
    return dict(r,training_numerical_qualified=True,original_metadata_pins=pins,actual_metadata_bytes=total,tensor_math_independently_recomputed=False)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['scope','output']:p.add_argument('--'+n,type=Path,required=True)
    for n in ['request-id','publication','source-manifest-sha256','claim-sha256']:p.add_argument('--'+n,required=True)
    for n in ['epoch','reader-origin']:p.add_argument('--'+n,type=float,required=True)
    a=p.parse_args();need(not a.output.exists(),'READER_ONCE')
    try:r=read(a)
    except BaseException as e:r=dict(metadata_verified=False,training_numerical_qualified=False,error_type=type(e).__name__,error_SHA256=sha(str(e).encode()),planned_updates=32,planned_microbatches=128,Model_GO=False,Hygon_GPU_GO=False,formal17=False)
    save(a.output,r);print(json.dumps(r));raise SystemExit(0 if r['metadata_verified'] else 1)
