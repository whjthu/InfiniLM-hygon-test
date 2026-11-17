"""Bounded original177 failure evidence; stdlib only, no Model qualification."""
import argparse,ast,gc,hashlib,json,math,os,re,resource,signal,socket,stat,sys,time
from pathlib import Path

D=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()
FLOOR=8*1024**3
PRIMARY=['RUN_MANIFEST.json']+['worker-%d.json'%i for i in range(2)]+['owner-%02d.json'%i for i in range(16)]+['owner-%02d.log'%i for i in range(16)]
PHASES=['created','loaded']+['case%d'%i for i in range(33)]+['quiescent','closed']

def need(ok,code):
    if not ok:raise ValueError(code)

def strict(raw):
    def pairs(items):
        out={}
        for k,v in items:need(k not in out,'JSON_DUPLICATE_KEY');out[k]=v
        return out
    def finite(s):
        v=float(s);need(math.isfinite(v),'JSON_NONFINITE');return v
    return json.loads(raw,object_pairs_hook=pairs,parse_float=finite,parse_constant=lambda _:need(False,'JSON_NONFINITE'))

def identity(st):return(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
def ishash(v):return type(v)is str and re.fullmatch('[0-9a-f]{64}',v)is not None
def real(v):return type(v)in(int,float) and math.isfinite(v) and v>=0

def safe_exception(text,trace,allowed):
    text=str(text)[:4096];trace=str(trace)[-8192:]
    typ=None;message=None
    match=re.match(r'^([A-Za-z_][A-Za-z_0-9]{0,63})\((.*)\)$',text,re.S)
    if match:
        typ=match[1]
        try:
            value=ast.literal_eval(match[2])
            if type(value)is str and value in allowed:message=value
        except (ValueError,SyntaxError):pass
    if typ is None:
        match=re.search(r'(?:^|\n)([A-Za-z_][A-Za-z_0-9]{0,63}): ([^\n]*)',text)
        if match:
            typ=match[1];message=match[2] if match[2] in allowed else None
    frames=[]
    for name,line,function in re.findall(r'File "([^"\n]+)", line (\d+), in ([^\n]+)',trace)[-16:]:
        basename=Path(name).name
        frames.append(dict(file=basename if re.fullmatch(r'[A-Za-z_0-9.-]{1,128}',basename) else None,line=int(line),function=function if re.fullmatch('[A-Za-z_0-9.<>]{1,80}',function) else None))
    return dict(type=typ,known_constant_message=message,text_prefix_SHA256=H(text.encode()),trace_suffix_SHA256=H(trace.encode()),frames=frames)

class BoundedRead:
    def __init__(self,root,guard):self.root=root;self.guard=guard;self.records={};self.owner_bytes=0
    def get(self,name,cap,log=False):
        need(name in PRIMARY,'FOREIGN_PATH');self.guard();path=self.root/name
        rec=dict(status='UNREAD');self.records[name]=rec
        try:
            before=path.lstat();need(stat.S_ISREG(before.st_mode),'NOT_REGULAR')
            rec['bytes']=before.st_size;need(before.st_size<=cap,'FILE_CAP')
            if name.startswith('owner-') and name.endswith('.json'):
                need(self.owner_bytes+before.st_size<=2*1024**3,'TOTAL_OWNER_CAP');self.owner_bytes+=before.st_size
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            with os.fdopen(fd,'rb') as f:
                start=os.fstat(f.fileno());need(identity(start)==identity(before),'OPEN_IDENTITY_CHANGED')
                digest=hashlib.sha256();parts=[];head=b'';tail=b'';size=0
                while True:
                    self.guard();chunk=f.read(1024**2)
                    if not chunk:break
                    size+=len(chunk);need(size<=cap,'GROWTH_CAP');digest.update(chunk)
                    if log:
                        if len(head)<65536:head+=(chunk[:65536-len(head)])
                        tail=(tail+chunk)[-65536:]
                    else:parts.append(chunk)
                after=os.fstat(f.fileno())
            need(identity(start)==identity(after)==identity(path.lstat()) and size==start.st_size,'FILE_CHANGED')
            rec.update(status='STABLE_FULL_SHA',SHA256=digest.hexdigest(),stable=True)
            if log:
                sample=head+(b'\n<OMITTED_MIDDLE>\n'+tail if size>len(head) else b'')
                return sample.decode('utf8','replace')
            raw=b''.join(parts);del parts
            return strict(raw)
        except (OSError,ValueError,TypeError,UnicodeError) as e:
            rec['status']=str(e) if type(e)is ValueError and str(e) in {'NOT_REGULAR','FILE_CAP','TOTAL_OWNER_CAP','OPEN_IDENTITY_CHANGED','GROWTH_CAP','FILE_CHANGED','JSON_DUPLICATE_KEY','JSON_NONFINITE'} else type(e).__name__
            return None

def validate_slot(row,index,pins,eos,special):
    pin=pins[0 if index==0 else index-1]
    need(type(row)is dict and all(row.get(k)==pin[k] for k in ('ordinal','case_id','suite','prompt_tokens','prompt_IDs_SHA256','body_SHA256')),'SLOT_INPUT_BINDING')
    need(row.get('diagnostic') is (index==0) and row.get('screen_index')==(None if index==0 else index-1),'SLOT_ROLE')
    need(row.get('completed') is True and row.get('reset_before') is True and row.get('reset_after') is True,'SLOT_INCOMPLETE')
    ids=row['generated_token_ids'];need(type(ids)is list and 0<len(ids)<=4096 and all(type(t)is int and 0<=t<154880 for t in ids),'TOKEN_RANGE')
    digest=H(json.dumps(ids,separators=(',',':')).encode());natural=ids[-1] in eos
    need(row['generated_token_ids_SHA256']==digest and not any(t in eos for t in ids[:-1]) and (natural or len(ids)==4096),'TOKEN_HASH_OR_EOS')
    need(row['planned_max_new_tokens']==4096 and row['natural_EOS'] is natural and row['cap_hit'] is (not natural),'EOS_CAP_BOUNDARY')
    ticks=row['sampled_ns'];need(len(ticks)==len(ids) and all(type(t)is int and t>0 for t in ticks) and all(a<=b for a,b in zip(ticks,ticks[1:])),'SAMPLED_CLOCK')
    pf=row['prefill_fragment_plan'];expected=[(o,min(512,pin['prompt_tokens']-o)) for o in range(0,pin['prompt_tokens'],512)]
    need([(x['offset'],x['tokens']) for x in pf]==expected and all(ishash(x['IDs_SHA256']) for x in pf) and row['runtime_prefill_chunk']==512 and row['original_request_prefill_window']==4096,'PREFILL_BOUNDARY')
    pre=row['prefill_CAPI_s'];dec=row['decode_CAPI_s']
    need(len(pre)==len(pf) and len(dec)==len(ids)-1 and all(real(v) for v in pre+dec) and row['Native_forward_end']-row['Native_forward_begin']==len(pre)+len(dec),'FORWARD_COUNT')
    need(real(row['whole_backend_s']) and row['whole_backend_s']>0 and (row['backend_decode_tok_s'] is None or real(row['backend_decode_tok_s'])),'DURATION')
    enter=row['request_enter_ns'];need(type(enter)is int and enter>0 and enter<=ticks[0],'REQUEST_ENTER_CLOCK')
    visible=[t for token,t in zip(ids,ticks) if token not in eos and token not in special]
    rate=(len(visible)-1)*1e9/(visible[-1]-visible[0]) if len(visible)>1 and visible[-1]>visible[0] else None
    ttft=(visible[0]-enter)/1e9 if visible else None
    def same(actual,expected):return actual is None if expected is None else type(actual)in(int,float) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12)
    need(type(row['visible_nonEOS_nonspecial'])is int and row['visible_nonEOS_nonspecial']==len(visible)
        and same(row['backend_decode_tok_s'],rate) and same(row['backend_sampled_TTFT_s'],ttft),'RECOMPUTED_VISIBLE_CLOCK_DIFFERS')
    return dict(slot=index,valid_saved_completion=True,generated=len(ids),generated_SHA256=digest,natural_EOS=natural,cap=not natural,reported_whole_backend_s=row['whole_backend_s'],backend_decode_tok_s=rate,backend_sampled_TTFT_s=ttft,visible_nonEOS_nonspecial=len(visible),sampled_clock_independently_recomputed=True)

def profile_summary(row):
    out={}
    for name,expected in (('pipeline_prefill',len(row['prefill_CAPI_s'])),('pipeline_decode',len(row['decode_CAPI_s']))):
        b=row[name]
        if expected==0:need(b=={},'EMPTY_PROFILE');out[name]=dict(count=0);continue
        need(b['count']==expected and all(len(b[k])==6 and all(real(x) for x in b[k]) for k in ('sum_s','min_s','max_s')),'PROFILE_SIX_SCALARS')
        out[name]=dict(count=expected,sum_s=b['sum_s'])
    return out

def main(a):
    start=time.monotonic();initial=a.epoch+180-time.time();need(math.isfinite(a.epoch) and 0<initial<=180,'READER_EPOCH')
    def guard():
        if min(a.epoch+175-time.time(),initial-(time.monotonic()-start)-5)<=0:raise TimeoutError('READER_WORK_DEADLINE')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('READER_WORK_DEADLINE')));signal.setitimer(signal.ITIMER_REAL,max(.001,initial-5))
    need(len(os.sched_getaffinity(0))==2 and not {0,1}&set(os.sched_getaffinity(0)),'AFFINITY')
    need(resource.getrlimit(resource.RLIMIT_AS)==(3*1024**3,3*1024**3),'ADDRESS_SPACE_LIMIT')
    own=strict((D/'SOURCE_FILES.json').read_bytes());need(H((D/'SOURCE_FILES.json').read_bytes())==a.source_sha,'OWN_SOURCE_MANIFEST')
    for name,pin in own.items():
        guard();raw=(D/name).read_bytes();need(len(raw)==pin['bytes'] and H(raw)==pin['SHA256'],'OWN_SOURCE_BYTES')
    binding=strict((D/'BINDING.json').read_bytes());original=strict((D/'ORIGINAL_PACKAGE_FILES.json').read_bytes())
    need(len(original)==46,'ORIGINAL47_MAP')
    for dst,src in binding['copied_original_map'].items():
        raw=(D/dst).read_bytes();need(H(raw)==binding['copied_source_SHA256'][dst],'ORIGINAL_COPY_PIN')
        if src!='PACKAGE_FILES.json':need(original[src]==H(raw),'ORIGINAL_PUBLISHED_MAP_MEMBER')
    task=strict((D/'ORIGINAL_TASK.json').read_bytes());terminal=strict((D/'ORIGINAL_TERMINAL.json').read_bytes())
    need(H((D/'ORIGINAL_TASK.json').read_bytes())==binding['original_task_SHA256'] and H((D/'ORIGINAL_TERMINAL.json').read_bytes())==binding['original_terminal_SHA256'],'ORIGINAL_EVIDENCE_PIN')
    need(task['run_id']==terminal['run_id']==binding['run_id'] and task['original_request']==terminal['original_request']==binding['original_request'] and terminal['source_publication']==binding['original_publication'] and terminal['launcher_rc']==1 and terminal['independent_post_clean'] is True,'FAILED_ORIGINAL_CONTEXT')
    inp=strict((D/'INPUT_BINDING.json').read_bytes());pins=inp['private_plan']['requests'];need(len(pins)==32,'SCREEN32_PIN');eos=set(inp['eos_token_ids']);special=set(inp['tokenizer']['special_ids'])
    cpu_binding=strict((D/'CPU_BINDING.json').read_bytes())
    need(task['package_SHA256']==H((D/'ORIGINAL_PACKAGE_FILES.json').read_bytes()) and task['claim_SHA256']==H((D/'ORIGINAL_CLAIM.json').read_bytes()) and task['epoch']==binding['epoch'] and terminal['actual_accounting']['gate'][0:4:2]==[binding['step'],'FAILED'],'TASK_STEP_SOURCE_BINDING')
    allowed=set(strict((D/'SAFE_ERRORS.json').read_bytes()))
    from ffn_evidence import audit_pair,check_frame
    from attention_evidence import attention_windows
    rr=BoundedRead(Path(binding['original_root']),guard)
    report=dict(schema='glm177-partial-original-failure-audit-v1',request_id=a.request,reader_publication=a.publication,original_run_id=binding['run_id'],original_request=binding['original_request'],original_step=binding['step'],original_publication=binding['original_publication'],failed_original_retained=True,full_run_qualified=False,Model_qualified=False,GPU_GO=False,planned_logical_requests=33,planned_unique_screens=32,planned_ranks=16,planned_rank_records=528,planned_attention_frames=48,workers=[],owners=[],slots=[dict(slot=i,valid_saved_completion=False,state='MISSING') for i in range(33)],files=rr.records,complete_reader_replaced=False,errors=[],score_scope='Saved rank0 score flags only; exact35scope excludes gold-plan reread/regrading.',clock_scope='Host/same-stream envelopes overlap/include enqueue/waits; not exclusive GPUbusy or speed gain.')
    workers={};worker_identity={};seen=[0]*33;hashes=[None]*33;mismatched=[False]*33;frame_valid=0;frame_observed=0;envelopes={};manifest_sha=None
    try:
        manifest=rr.get('RUN_MANIFEST.json',65536);need(manifest is not None,'RUN_MANIFEST_UNAVAILABLE');manifest_sha=rr.records['RUN_MANIFEST.json']['SHA256']
        claim=strict((D/'ORIGINAL_CLAIM.json').read_bytes())
        expected=dict(schema='glm474-attention-detail-worker-v1',run_id=binding['run_id'],arm='socket',epoch=binding['epoch'],parent='1103606',nodes=binding['nodes'],output='/refactor/results/'+Path(binding['original_root']).name,Native_SHA256=binding['Native_SHA256'],GPU_GO=True,Model_execution_authorized=True,Model_qualified=False)
        need(set(manifest)==set(expected)|{'source_manifest_SHA256','claim_SHA256','transport_key_SHA256'} and all(type(manifest.get(k))is type(v) and manifest[k]==v for k,v in expected.items()) and manifest['claim_SHA256']==H((D/'ORIGINAL_CLAIM.json').read_bytes()) and manifest['source_manifest_SHA256']==H((D/'ORIGINAL_SOURCE_MANIFEST.json').read_bytes())==claim['source_manifest_SHA256'] and ishash(manifest['transport_key_SHA256']),'RUN_MANIFEST_BINDING')
        report['run_manifest_valid']=True
        for node in range(2):
            w=rr.get('worker-%d.json'%node,2*1024**2)
            info=dict(node_index=node,identity_valid=False)
            try:
                need(w is not None and w['node_index']==node and w['node']==binding['nodes'][node] and w['actual_GPU']==binding['step'] and w['gpu_epoch']==binding['epoch'],'WORKER_NODE_STEP')
                ident=w['identity'];need(ident['run_manifest_SHA256']==manifest_sha and ident['source_manifest_SHA256']==manifest['source_manifest_SHA256'] and ident['Native_SHA256']==binding['Native_SHA256'] and ident['epoch']==binding['epoch'] and ident['step']==binding['step'] and ident['arm']=='socket' and ident['label']==Path(binding['original_root']).name,'WORKER_IDENTITY')
                rows=w['owners'];need(len(rows)==8 and [r['rank'] for r in rows]==list(range(node*8,node*8+8)) and all(r['global_rank']==r['rank'] and type(r['PID'])is int and r['PID']>0 and type(r['startticks'])is str and r['startticks'].isdigit() for r in rows),'WORKER_OWNER8')
                rounds=w['rounds'];actual=[r['phase'] for r in rounds];need(actual==PHASES[:len(actual)] and len(actual)<=37 and all(ishash(r['digest']) for r in rounds),'WORKER_PHASE_SEQUENCE')
                worker_identity[node]=True;workers[node]=w
                info.update(identity_valid=True,completed=w.get('completed'),passed=w.get('passed'),forced_TERM=bool(w.get('forced_TERM')),forced_KILL=bool(w.get('forced_KILL')),phase_count=len(actual),all37_phases=len(actual)==37,phase_digests=[r['digest'] for r in rounds],owner_PID_startticks=[dict(rank=r['rank'],PID=r['PID'],startticks=r['startticks']) for r in rows],exception=safe_exception(w.get('error',''),w.get('traceback',''),allowed))
            except (KeyError,ValueError,TypeError,IndexError,AttributeError,OverflowError) as e:info['rejection']=str(e)[:96] if isinstance(e,ValueError) else type(e).__name__
            report['workers'].append(info)
        report['worker_phase_digests_consistent']=len(workers)==2 and [r['digest'] for r in workers[0]['rounds']]==[r['digest'] for r in workers[1]['rounds']]
        for rank in range(16):
            guard();q=rr.get('owner-%02d.json'%rank,128*1024**2);info=dict(rank=rank,identity_valid=False,valid_completion_count=0,invalid_slots=[],observed_frames=0,schema_valid_frames=0,frame_rejections=[])
            try:
                node=rank//8;need(q is not None and worker_identity.get(node),'OWNER_OR_WORKER_MISSING');known=workers[node]['owners'][rank%8]
                need(q['global_rank']==rank and q['node_index']==node and q['node']==binding['nodes'][node] and q['device']==rank%8 and q['actual_GPU']==binding['step'] and q['gpu_epoch']==binding['epoch'] and q['PID']==known['PID'] and q['startticks']==known['startticks'],'OWNER_PID_RANK_STEP')
                need(q['Native_arm']=='socket' and q['Native_SHA256']==binding['Native_SHA256'] and q['GPU_manifest_SHA256']==manifest_sha and q['source_manifest_SHA256']==manifest['source_manifest_SHA256'] and q['private_input_SHA256']==inp['private_plan']['SHA256'],'OWNER_SOURCE_NATIVE')
                need(q['profile_getter_binding']['SHA256']==binding['Native_SHA256'] and q['profile_getter_binding']['symbol']=='modern_glm_pipeline_last_profile' and q['profile_getter_binding']['count']==6
                    and q['CAPI_bindings'] and all(v['SHA256']==binding['Native_SHA256'] for v in q['CAPI_bindings'].values())
                    and all(q['actual_provider_maps'][path]['SHA256']==sha for path,sha in cpu_binding['actual_project_DSO_SHA256'].items()),'OWNER_RECORDED_PROVIDER_BINDING')
                info.update(identity_valid=True,recorded_provider_binding_valid=True,completed=q.get('completed'),passed=q.get('passed'),normal_close=q.get('normal_close'),Runtime_weak_expired_CABI_assertion=q.get('Runtime_weak_expired_CABI_assertion'),Native_forward_attempts=q.get('Native_forward_attempts'),Native_forward_completed=q.get('Native_forward_completed'),resets=q.get('resets'),failure_context_live=q.get('failure_context_live'),exception=safe_exception(q.get('error',''),q.get('traceback',''),allowed))
                mem=q.get('driver_memory',{});bounds=q.get('memory_boundaries',[])
                memory_valid=type(mem.get('samples'))is int and mem['samples']>0 and len(bounds)>0 and type(mem.get('min_observed'))is int and mem['min_observed']>=FLOOR and not mem.get('errors') and all(type(x.get('free'))is int and x['free']>=FLOOR and x.get('floor')==FLOOR for x in bounds)
                info['memory']=dict(sampled_floor_bytes=FLOOR,min_observed=mem.get('min_observed'),samples=mem.get('samples'),boundary_count=len(bounds),minimum_boundary_free=min((x['free'] for x in bounds if type(x.get('free'))is int),default=None),sampled_and_boundary_floor_satisfied=memory_valid,hard_peak_guarantee=False)
                rows=q['requests'];need(type(rows)is list and len(rows)<=33,'OWNER_REQUEST_DENOMINATOR');info['reported_request_count']=len(rows)
                for index in range(33):
                    if rank==0 and index<len(rows) and type(rows[index])is dict:
                        item=rows[index];rawscore=item.get('score',{})
                        report['slots'][index].update(state='OBSERVED_NOT_YET_VALIDATED',reported_completed=item.get('completed') if type(item.get('completed'))is bool else None,reported_cap=item.get('cap_hit') if type(item.get('cap_hit'))is bool else None,reported_correct=rawscore.get('correct') if type(rawscore)is dict and type(rawscore.get('correct'))is bool else None)
                    try:
                        need(index<len(rows),'MISSING_REQUEST');r=validate_slot(rows[index],index,pins,eos,special)
                        if hashes[index] is None:hashes[index]=r['generated_SHA256']
                        elif hashes[index]!=r['generated_SHA256']:mismatched[index]=True
                        seen[index]+=1;info['valid_completion_count']+=1
                        if rank==0:
                            score=rows[index].get('score',{});fields=('correct','answer_matches_gold','parse_failure','natural_EOS','cap_hit','unexpected_tool_intent')
                            score_ok=type(score)is dict and all(type(score.get(k))is bool for k in fields) and score['natural_EOS']==r['natural_EOS'] and score['cap_hit']==r['cap'] and (not score['correct'] or score['answer_matches_gold'] and r['natural_EOS'] and not r['cap'] and not score['unexpected_tool_intent'])
                            r['reported_score']={k:score[k] for k in fields} if score_ok else None
                            try:r['profile']=profile_summary(rows[index])
                            except (KeyError,ValueError,TypeError) as e:r['profile_rejection']=str(e)[:96] if type(e)is ValueError else type(e).__name__
                            r.update({k:v for k,v in report['slots'][index].items() if k.startswith('reported_')});report['slots'][index]=r
                    except (KeyError,ValueError,TypeError,IndexError,AttributeError,OverflowError) as e:info['invalid_slots'].append(index)
                if rows:
                    diagnostic=rows[0];frames=diagnostic.get('observer_report',{}).get('samples',[])
                    need(type(frames)is list and len(frames)<=3,'FRAME_COUNT');info['observed_frames']=len(frames);frame_observed+=len(frames)
                    try:
                        need(len(rows)>=2,'PAIR_MISSING');pair=audit_pair(rows[0],rows[1],rank,eos);info['original_audit_pair_passed']=True
                    except (KeyError,ValueError,TypeError,IndexError,AttributeError,OverflowError) as e:info.update(original_audit_pair_passed=False,pair_rejection=safe_exception(repr(e),'',allowed))
                    for index,frame in enumerate(frames):
                        try:
                            need(q.get('attention_detail_optin') is True and q.get('observer_pool_events')==512 and q.get('observer_source_event_bound')==440 and diagnostic['observer_report']['global_tp_rank']==rank,'FRAME_OWNER_BINDING')
                            check_frame(frame,diagnostic,rank,index);layers=attention_windows(frame);info['schema_valid_frames']+=1;frame_valid+=1
                            for layer in layers:
                                for win in layer['windows']:
                                    key='%d:%s'%(layer['layer'],win['name']);b=envelopes.setdefault(key,dict(n=0,host_sum=0.,stream_sum=0.,stream_n=0));need(math.isfinite(b['host_sum']+win['host_envelope_seconds']),'AGGREGATE_NONFINITE');b['n']+=1;b['host_sum']+=win['host_envelope_seconds']
                                    if win['stream_envelope_seconds'] is not None:need(math.isfinite(b['stream_sum']+win['stream_envelope_seconds']),'AGGREGATE_NONFINITE');b['stream_sum']+=win['stream_envelope_seconds'];b['stream_n']+=1
                        except (KeyError,ValueError,TypeError,IndexError,AttributeError,OverflowError) as e:info['frame_rejections'].append(dict(frame=index,error=safe_exception(repr(e),'',allowed)))
            except (KeyError,ValueError,TypeError,IndexError,AttributeError,OverflowError) as e:info['rejection']=str(e)[:96] if type(e)is ValueError else type(e).__name__
            report['owners'].append(info);del q;gc.collect()
        for rank in range(16):
            text=rr.get('owner-%02d.log'%rank,4*1024**2,log=True)
            if text is not None:
                lines=text.splitlines();errors=[s for s in lines if re.match(r'^[A-Za-z_][A-Za-z_0-9]*(?:Error|Exception)(?:\:|$)',s)]
                report['owners'][rank]['log_exception']=safe_exception(errors[-1] if errors else '',text,allowed)
    except (ValueError,KeyError,OSError,TypeError,TimeoutError,MemoryError,AttributeError,OverflowError) as error:
        report['errors'].append(dict(type=type(error).__name__,safe_constant=str(error) if str(error) in {'READER_WORK_DEADLINE','RUN_MANIFEST_UNAVAILABLE','RUN_MANIFEST_BINDING'} else None))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
    for index,row in enumerate(report['slots']):
        row.update(valid_rank_completions=seen[index],cross_rank_hash_disagreement=mismatched[index],all16_saved_completion_consistent=seen[index]==16 and not mismatched[index])
    screen=report['slots'][1:];complete=[r for r in screen if r['all16_saved_completion_consistent']]
    scored=[r for r in complete if r.get('reported_score') is not None]
    report.update(files=rr.records,files_read=len(rr.records),missing_primary_paths=[n for n in PRIMARY if n not in rr.records or rr.records[n]['status']!='STABLE_FULL_SHA'],screen_summary=dict(planned=32,all16_consistent_completed=len(complete),missing_invalid_or_discordant=32-len(complete),caps=sum(r.get('cap') is True for r in complete),reported_score_known=len(scored),reported_correct=sum(r['reported_score']['correct'] for r in scored),reported_wrong_or_invalid=sum(not r['reported_score']['correct'] for r in scored),score_unknown=32-len(scored),rank0_reported_caps_all_slots=sum(r.get('reported_cap') is True for r in screen),rank0_reported_wrong_all_slots=sum(r.get('reported_correct') is False for r in screen),rank0_reported_score_missing=sum(r.get('reported_correct') is None for r in screen)),attention_observed_frames=frame_observed,attention_schema_valid_frames=frame_valid,attention_schema_full48=frame_valid==48,valid_attention_detail=False,attention_sample_state='RETAINED_DIAGNOSTIC_SCHEMA_FULL48_NOT_RUN_QUALIFIED' if frame_valid==48 else 'UNKNOWN_INSUFFICIENT_OR_REJECTED_FRAMES',attention_envelope_aggregates=envelopes,actual_affinity=sorted(os.sched_getaffinity(0)),address_space_limit_bytes=3*1024**3,address_space_is_not_RSS=True,peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_s=time.monotonic()-start,epoch=a.epoch,deadline=a.epoch+180,owner=dict(host=socket.gethostname(),PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()),new_GPU_Model_calls=0,partial_audit_completed=len(rr.records)==35 and not report['errors'])
    raw=(json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode();need(len(raw)<=128*1024,'REPORT_CAP')
    need(time.time()<a.epoch+180,'FINAL_ACCOUNTING_DEADLINE')
    with a.output.open('xb') as f:f.write(raw)
    print(json.dumps(dict(request_id=a.request,report_bytes=len(raw),report_SHA256=H(raw),report=report),separators=(',',':'),allow_nan=False),flush=True)
    return 0 if report['partial_audit_completed'] else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=float,required=True);p.add_argument('--request',required=True);p.add_argument('--publication',required=True);p.add_argument('--source-sha',required=True);p.add_argument('--output',type=Path,required=True);raise SystemExit(main(p.parse_args()))
