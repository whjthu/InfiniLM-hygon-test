"""Corrected48frame audit only; original177 FAILED and all old reports remain immutable."""
import argparse,hashlib,json,math,os,re,resource,signal,socket,stat,statistics,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def strict(raw):
    def pairs(items):
        out={}
        for k,v in items:need(k not in out,'DUPLICATE_JSON');out[k]=v
        return out
    def finite(s):
        v=float(s);need(math.isfinite(v),'NONFINITE_JSON');return v
    return json.loads(raw,object_pairs_hook=pairs,parse_float=finite,parse_constant=lambda _:need(False,'NONFINITE_JSON'))
def identity(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def safe_error(e,allowed):
    s=str(e);return dict(type=type(e).__name__,known_constant=s if s in allowed else None,message_SHA256=H(s[:4096].encode()))
def read_owner(path,pin,guard):
    guard();s=path.lstat();need(stat.S_ISREG(s.st_mode) and 0<s.st_size==pin['bytes']<=128*1024**2,'OWNER_REGULAR_SIZE')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        before=os.fstat(f.fileno());need(identity(before)==identity(s),'OWNER_OPEN_IDENTITY')
        parts=[];h=hashlib.sha256();n=0
        while True:
            guard();b=f.read(1024**2)
            if not b:break
            n+=len(b);need(n<=pin['bytes'],'OWNER_GROWTH');h.update(b);parts.append(b)
        after=os.fstat(f.fileno())
    need(identity(before)==identity(after)==identity(path.lstat()) and n==pin['bytes'] and h.hexdigest()==pin['SHA256'],'OWNER_PIN_OR_STABILITY')
    raw=b''.join(parts);del parts
    return strict(raw)
def summary(values):
    need(values and all(math.isfinite(v) and v>=0 for v in values),'AGGREGATE_VALUES')
    return dict(count=len(values),minimum_s=min(values),median_s=statistics.median(values),maximum_s=max(values),sum_s=math.fsum(values))
def main(a):
    start=time.monotonic();left=a.epoch+180-time.time();need(math.isfinite(a.epoch) and 0<left<=180,'EPOCH')
    def guard():
        if min(a.epoch+175-time.time(),left-(time.monotonic()-start)-5)<=0:raise TimeoutError('WORK_DEADLINE')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('WORK_DEADLINE')));signal.setitimer(signal.ITIMER_REAL,max(.001,left-5))
    need(len(os.sched_getaffinity(0))==2 and not {0,1}&set(os.sched_getaffinity(0)) and resource.getrlimit(resource.RLIMIT_AS)==(3*1024**3,3*1024**3),'CPU_AS_BOUNDARY')
    raw=(D/'SOURCE_FILES.json').read_bytes();need(H(raw)==a.source_sha,'OWN_MANIFEST')
    for name,pin in strict(raw).items():
        guard();b=(D/name).read_bytes();need(len(b)==pin['bytes'] and H(b)==pin['SHA256'],'OWN_SOURCE')
    binding=strict((D/'BINDING.json').read_bytes());original=strict((D/'ORIGINAL_PACKAGE_FILES.json').read_bytes())
    for dst,src in binding['copied_original_map'].items():
        b=(D/dst).read_bytes();need(H(b)==binding['copied_source_SHA256'][dst] and (src=='PACKAGE_FILES.json' or original[src]==H(b)),'ORIGINAL_SOURCE_PIN')
    pins=strict((D/'ORIGINAL_OWNER_PINS.json').read_bytes());need(pins['prior_report_SHA256']=='3a11469233946da5695dc6d9cfb6383480bb6230968464626dd502b2603a3401' and pins['prior_observed_frames']==48 and pins['prior_schema_valid_frames']==0,'PRIOR_FAILURE_AUDIT_PIN')
    names=['owner-%02d.json'%i for i in range(16)];need(set(pins['owner_files'])==set(names) and sum(p['bytes'] for p in pins['owner_files'].values())<=2*1024**3,'EXACT16_PINNED_OWNERS')
    inp=strict((D/'INPUT_BINDING.json').read_bytes());original_input=inp['private_plan']['requests'][0]
    allowed=set(strict((D/'SAFE_ERRORS.json').read_bytes()))
    from ffn_evidence import check_frame
    from attention_evidence import attention_windows
    report=dict(schema='glm177-corrected-attention48-saved-frame-audit-v1',request_id=a.request,reader_publication=a.publication,original_step=binding['step'],original_run_id=binding['run_id'],original_request=binding['original_request'],original_publication=binding['original_publication'],prior_report_SHA256=pins['prior_report_SHA256'],prior_observed_frames=48,prior_schema_valid_frames=0,prior_rejection_and_original_failure_retained=True,full_run_qualified=False,Model_qualified=False,GPU_GO=False,planned_owner_files=16,planned_frames=48,files={},owners=[],errors=[],new_model_or_native_calls=0,new_GPU_or_driver_queries=0,quality_or_32screen_regraded=False,generic_projection_includes_FFN_gate_up=True,phase_attribution_requires_original_mark_indices=True,old35reader_reexecuted=False,clock_scope='Original host/same-stream enclosing windows overlap/include enqueue/wait. No exclusiveGPUbusy, speedup, or fullModelPASS.',original_33planned_and32screen_denominators_preserved=True)
    aggregates={};observed=valid=0
    try:
        for rank,name in enumerate(names):
            guard();pin=pins['owner_files'][name];rec=dict(rank=rank,observed_frames=0,schema_valid_frames=0,frames=[]);report['owners'].append(rec)
            try:
                q=read_owner(Path(binding['original_root'])/name,pin,guard);report['files'][name]=dict(pin,stable_full_SHA=True)
                owner=pins['owner_identity'][str(rank)]
                need(q['global_rank']==rank and q['node_index']==rank//8 and q['device']==rank%8 and q['node']==binding['nodes'][rank//8] and q['actual_GPU']==binding['step'] and q['gpu_epoch']==binding['epoch'] and q['PID']==owner['PID'] and q['startticks']==owner['startticks'],'ORIGINAL_OWNER_IDENTITY')
                need(q['Native_SHA256']==binding['Native_SHA256'] and q['Native_arm']=='socket' and q['GPU_manifest_SHA256']==pins['original_RUN_MANIFEST_SHA256'] and q['source_manifest_SHA256']==H((D/'ORIGINAL_SOURCE_MANIFEST.json').read_bytes()) and q['private_input_SHA256']==inp['private_plan']['SHA256'],'ORIGINAL_NATIVE_SOURCE_INPUT')
                diagnostic=q['requests'][0]
                need(all(diagnostic[k]==original_input[k] for k in ('ordinal','case_id','suite','prompt_tokens','prompt_IDs_SHA256','body_SHA256')) and diagnostic['diagnostic'] is True and diagnostic['screen_index'] is None and diagnostic['observer_diagnostic'] is True,'ORIGINAL_DIAGNOSTIC_INPUT')
                need(q['observer_diagnostic']==diagnostic['observer_report'] and q['attention_detail_optin'] is True and q['observer_pool_events']==512 and q['observer_source_event_bound']==440,'ORIGINAL_OBSERVER_BINDING')
                obs=diagnostic['observer_report'];frames=obs['samples']
                need(obs['global_tp_rank']==rank and obs['event_cap_per_frame']==obs['events_allocated']==512 and obs['frame_cap']==3 and obs['total_frames_used']==len(frames)==3 and obs['coverage']==[0,0,0,3] and obs['pending_next_decode'] is False,'EXACT3_SAVED_FRAMES')
                rec['observed_frames']=3;observed+=3
                for index,frame in enumerate(frames):
                    item=dict(frame=index,passed=False);rec['frames'].append(item)
                    try:
                        check_frame(frame,diagnostic,rank,index);layers=attention_windows(frame,rank)
                        history=[];windows=[]
                        for layer in layers:
                            hs=[t for t in layer['tuples'] if t['site']=='projection' and t['M']>1]
                            need(len(hs)==1,'EXACT_ONE_HISTORY_TUPLE_PER_LAYER')
                            history.append(dict(layer=layer['layer'],M=hs[0]['M'],N=hs[0]['N'],K=hs[0]['K'],row_offset=hs[0]['row_offset'],column_offset=hs[0]['column_offset']))
                            for w in layer['windows']:
                                windows.append((layer['layer'],w))
                        # Aggregate only complete frames passing both selected layers.
                        for layer,w in windows:
                            key='%d:%s:%s:%d'%(layer,w['caller_phase'],w['name'],w['occurrence']);b=aggregates.setdefault(key,dict(host=[],stream=[]));b['host'].append(w['host_envelope_seconds'])
                            if w['stream_envelope_seconds'] is not None:b['stream'].append(w['stream_envelope_seconds'])
                        item.update(passed=True,past=frame['row_past_count'][0][1],sequence=frame['sequence'],event_count=frame['event_count'],history_projection=history,window_count=len(windows),tuple_count=sum(len(l['tuples']) for l in layers),dispatch_counts={branch:sum(d['observed_branch']==branch for l in layers for d in l['dispatches']) for branch in ('cached_r809','fallback_Fp8BlockLinear')})
                        valid+=1;rec['schema_valid_frames']+=1
                    except (ValueError,KeyError,TypeError,IndexError,AttributeError,OverflowError) as e:item['rejection']=safe_error(e,allowed)
                del q
            except (ValueError,KeyError,TypeError,IndexError,AttributeError,OSError,MemoryError) as e:
                rec['rejection']=safe_error(e,allowed);report['files'].setdefault(name,dict(pin,status=type(e).__name__,stable_full_SHA=False))
    except (TimeoutError,MemoryError,ValueError) as e:report['errors'].append(safe_error(e,allowed))
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    report.update(observed_frames=observed,corrected_schema_valid_frames=valid,corrected48_schema_passed=valid==48 and len(report['files'])==16 and not report['errors'],envelope_costs={key:dict(host=summary(v['host']),stream=summary(v['stream']) if v['stream'] else None,exclusive_GPU_time=False) for key,v in aggregates.items()},schema_state='SAVED48_CORRECTED_DIAGNOSTIC_SCHEMA_PASS_NOT_MODEL_QUALIFICATION' if valid==48 else 'UNKNOWN_INCOMPLETE_OR_REJECTED',epoch=a.epoch,deadline=a.epoch+180,elapsed_s=time.monotonic()-start,CPU_affinity=sorted(os.sched_getaffinity(0)),RLIMIT_AS_bytes=3*1024**3,AS_is_not_RSS=True,peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,reader_owner=dict(host=socket.gethostname(),PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    body=(json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode();need(len(body)<=128*1024 and time.time()<a.epoch+180,'FINAL_REPORT_BOUNDARY')
    with a.output.open('xb') as f:f.write(body)
    print(json.dumps(dict(request_id=a.request,report_bytes=len(body),report_SHA256=H(body),report=report),separators=(',',':'),allow_nan=False),flush=True)
    return 0 if report['corrected48_schema_passed'] else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=float,required=True);p.add_argument('--request',required=True);p.add_argument('--publication',required=True);p.add_argument('--source-sha',required=True);p.add_argument('--output',type=Path,required=True);raise SystemExit(main(p.parse_args()))
