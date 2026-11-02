"""Fixed-input numerical diagnosis. No HTTP, free-running AR, or new Native math."""
import copy,hashlib,json,os,pathlib,time
from hc_model_contract import HCModel

D=pathlib.Path(__file__).resolve().parent
R=pathlib.Path('/refactor')
MODES=(0,1,1,0)
PROMPT=8489
TEACHER=83
DECISIONS=84
MAX_FORWARDS=368
NATIVE_SHA='976f97b51e9f4b11dfb2439d2c97f2a1e7788e4de05ee8a8a0138f33e38b6ed1'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
ids_sha=lambda ids:hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()

def load_private():
    ref=json.loads((D/'TEACHER_REFERENCE.json').read_bytes())
    p=pathlib.Path(ref['path']);assert sha(p)==ref['SHA256']
    payload=json.loads(p.read_bytes());teacher=payload['teacher_token_ids']
    assert len(teacher)==ref['teacher_count']==TEACHER and ids_sha(teacher)==ref['teacher_IDs_SHA256']
    assert all(type(x)is int and 0<=x<129280 and x!=1 for x in teacher)
    assert [payload['correct_reference_output83'],payload['incorrect_reference_output83']]==ref['reference_output83']==[30722,28302]
    fixture=pathlib.Path(payload['fixture_path']);assert sha(fixture)==payload['fixture_SHA256']
    f=json.loads(fixture.read_bytes());assert f['actual_prompt_tokens']==PROMPT and f['actual_prompt_token_ids_SHA256']==payload['prompt_token_ids_SHA256']
    return payload,f

def prepare_prompt(prepared,payload,fixture):
    # Same pure upstream tool normalization and same real prepared encoder as
    # the original HTTP route. No HTTP request is submitted and no queue entry
    # is created. Exact original prompt/ID hashes are required before forward.
    from infinilm.server.agent_protocol import prepare_agent_request
    raw=copy.deepcopy(fixture['body'])
    data,schemas,choice=prepare_agent_request(raw,'deepseek_v4')
    data['_agent_schemas']=schemas;data['_agent_choice']=choice
    assert data.get('add_generation_prompt',True) is True
    prompt,ids=prepared['engine'].encode_chat(data['messages'],data,True,data.get('chat_template_kwargs')or{})
    ids=list(ids)
    assert len(ids)==PROMPT and all(type(x)is int and 0<=x<129280 for x in ids)
    assert ids_sha(ids)==payload['prompt_token_ids_SHA256']
    assert hashlib.sha256(prompt.encode()).hexdigest()==payload['prompt_text_SHA256']
    return ids

def logits_summary(np,logits,sampled,index,teacher,vocab):
    assert getattr(logits,'dtype',None)==np.dtype('uint16') and logits.shape==(1,vocab), 'Actual Native BF16 uint16 logits ABI differs'
    # Bit expansion of returned BF16, not a new F32 Native calculation.
    scores=(logits[0].astype(np.uint32)<<16).view(np.float32)
    if not np.isfinite(scores).all():raise RuntimeError('Nonfinite actual Native BF16 logits')
    assert len(sampled)==1 and 0<=int(sampled[0])<vocab
    native=int(sampled[0]);order=np.lexsort((np.arange(vocab),-scores));top=order[:8]
    independent=int(top[0]);maximum=float(scores[independent]);native_score=float(scores[native])
    assert native_score==maximum, 'Native sampled ID is not in independent BF16-logit argmax tie set'
    selected={'native_argmax':native,'independent_argmax':independent,'correct_original83':30722,'incorrect_original83':28302}
    if index<TEACHER:selected['teacher']=teacher[index]
    return {'output_index':index,'logits_storage':'Native BF16 encoded uint16; exact bit-expanded F32 diagnostic representation',
            'finite':True,'top8':[{'id':int(k),'score':float(scores[k])}for k in top],
            'top1_margin':float(scores[top[0]])-float(scores[top[1]]),
            'maximum_tie_count':int(np.count_nonzero(scores==maximum)),
            'selected_scores':{k:{'id':v,'score':float(scores[v])}for k,v in selected.items()},
            'logits_BF16_bytes_SHA256':hashlib.sha256(logits.tobytes()).hexdigest()}

def owner_threads(ns):
 try:
  owners=ns['verify_owner_affinity'](ns['lib'])['actual_owner_report']['owners'];rows=[];unknown=[]
  for owner in owners:
   mask={int(x)for x in owner['verified_mask'].split(',')};matches=[]
   for p in pathlib.Path('/proc/self/task').iterdir():
    try:
     if set(os.sched_getaffinity(int(p.name)))==mask:matches.append(int(p.name))
    except ProcessLookupError:pass
   if len(matches)==1:rows.append({'device':owner['device'],'tid':matches[0],'basis':'unique actual verified owner CPU mask'})
   else:unknown.append({'device':owner['device'],'matching_tasks':len(matches)})
  return {'status':'OBSERVED'if len(rows)==8 else'UNKNOWN','owners':rows,'ambiguous':unknown,'not_Native_per_owner_callback':True}
 except Exception as exc:return {'status':'UNKNOWN','error':repr(exc),'owners':[]}

def serve(ns):
    from native_binding_r206b import binding
    b,c,n=binding();assert b['library_SHA256']==c['library_SHA256']==NATIVE_SHA and len(n['link']['full']['objects'])==49
    payload,fixture=load_private();prepared=ns['__latest_prepared_r277__']
    assert prepared['identity']['upstream_commit']=='99f5c90a40f9b2546bd0202cba3397ed1d55c5a0' and prepared['identity']['eos_token_ids']==[1]
    ids=prepare_prompt(prepared,payload,fixture);teacher=list(payload['teacher_token_ids'])
    np=ns['np'];lib=ns['lib'];check=ns['check'];ctx=ns['ctx'];hc=HCModel(lib,ns['cfg']['num_hidden_layers'])
    # Existing observers stay disabled throughout; no arm(1), read or packet.
    check(lib.deep_owner_arm_r8900(0));check(lib.deep_attention_inner_arm_r9193(0))
    tag=ns['tag'];root=R/'results'/('Deep-HC-teacher-forced-logits-r397-'+tag);root.mkdir(exist_ok=False)
    result=pathlib.Path(str(root)+'.json')
    report={'completed':False,'passed':False,'diagnostic_only':True,'quality_PASS':False,'performance_PASS':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'legacy_tag':tag,
            'Native_SHA256':NATIVE_SHA,'source_manifest_SHA256':os.environ['DEEP_SOURCE_MANIFEST_SHA'],
            'prepared_entry_identity':prepared['identity'],'bootstrap':ns['__bootstrap_proof_r206b__'],
            'provider_preweight':ns['__provider_proof_r206b__'],'actual_preload_create':ns['__create_proof_r206b__'],
            'teacher_IDs_SHA256':ids_sha(teacher),'teacher_count':83,'prompt_IDs_SHA256':ids_sha(ids),'prompt_count':8489,
            'HC_mode_schedule':list(MODES),'planned_sequences':4,'planned_forwards':368,'actual_forwards':0,'actual_forward_attempts':0,'actual_resets':0,'actual_reset_attempts':0,
            'context':32768,'prefill_chunk':1024,'evaluated_output_indices':[0,83],'observer_enabled':False,
            'HTTP_requests':0,'natural_EOS_claim':False,'teacher_forced_stop':True,'sequences':[],
            'normal_Native_shutdown':False,'normal_server_shutdown':'NOT_STARTED_NO_HTTP',
            'numerical_scope':'Returned Native BF16 logits, bit-expanded F32 summaries only; no full logits published'}
    ns['__owner_exit_observation_r206b__']=owner_threads(ns)
    def save():
        tmp=result.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(result)
    def deadline():
        if time.time()>=float(os.environ['DEEP_R277_GPUSTART_UNIX'])+1680:raise TimeoutError('Original residency absolute diagnostic deadline')
    save()
    try:
        for sequence,mode in enumerate(MODES):
            deadline();row={'sequence':sequence,'HC_mode':mode,'reset_ok':False,'completed':False,'prefill_calls':0,'decode_calls':0,'decisions':[],'HC_forward_coverage':[]};report['sequences'].append(row);save()
            # Exactly one planned full real reset before each sequence.
            report['actual_reset_attempts']+=1;check(lib.modern_native_engine_reset(ctx));report['actual_resets']+=1;row['reset_ok']=True
            row['HC_counters_before']=hc.select(mode)
            row['GPU_free_before_all8']=[list(ns['torch'].cuda.mem_get_info(d))for d in range(8)]
            assert min(v[0]for v in row['GPU_free_before_all8'])>=16*1024**3
            past=0
            def forward(part,kind,index=None):
                nonlocal past
                deadline();assert report['actual_forward_attempts']<MAX_FORWARDS
                before=hc.snapshot();began=time.monotonic()
                report['actual_forward_attempts']+=1
                code,logits,sampled=ns['forward'](part,list(range(past,past+len(part))),[0],[past],[past+len(part)],[0,len(part)])
                check(code);report['actual_forwards']+=1;row[kind+'_calls']+=1
                row['HC_forward_coverage'].append(hc.verify_forward(before,hc.snapshot(),len(part),mode))
                # All intermediate prefill logits also require the actual ABI
                # and finite values, but only the last prefill is output0.
                assert logits.dtype==np.dtype('uint16') and logits.shape==(1,ns['cfg']['vocab_size'])
                if not np.isfinite((logits[0].astype(np.uint32)<<16).view(np.float32)).all():raise RuntimeError('Nonfinite Native logits')
                if index is not None:
                    item=logits_summary(np,logits,sampled,index,teacher,ns['cfg']['vocab_size'])
                    item.update(input_past=past,input_count=len(part),input_IDs_SHA256=ids_sha(part),forward_wall_s=time.monotonic()-began)
                    row['decisions'].append(item)
                past+=len(part);deadline();save()
            for offset in range(0,PROMPT,1024):
                part=ids[offset:offset+1024];forward(part,'prefill',0 if offset+len(part)==PROMPT else None)
            assert past==PROMPT and len(row['decisions'])==1
            for output_index in range(1,84):
                # Feed the same teacher token regardless of Native argmax.
                forward([teacher[output_index-1]],'decode',output_index)
            assert past==8572 and row['prefill_calls']==9 and row['decode_calls']==83
            assert [v['output_index']for v in row['decisions']]==list(range(84))
            row.update(completed=True,HC_counters_after=hc.snapshot(),last_total=past);save()
        assert report['actual_forwards']==368 and report['actual_resets']==4 and len(report['sequences'])==4
        report.update(completed=True,passed=True);save()
    except BaseException as exc:
        report.update(completed=True,passed=False,error=repr(exc),primary_error=repr(exc));save();raise
    finally:
        # Default OFF on exit, including exceptional exit. The unmodified
        # legacy loader finally still destroys the one real Native engine.
        report.update(HC_clear_rc=None,HC_after_clear=None,observer_disarm_rc=[None,None],cleanup_errors=[])
        operations=[('HC_clear',lambda:int(hc.set(0))),('HC_readback',lambda:int(hc.get())),('owner_disarm',lambda:int(lib.deep_owner_arm_r8900(0))),('inner_disarm',lambda:int(lib.deep_attention_inner_arm_r9193(0)))]
        for name,operation in operations:
            try:
                value=operation()
                if name=='HC_clear':report['HC_clear_rc']=value
                elif name=='HC_readback':report['HC_after_clear']=value
                else:report['observer_disarm_rc'][0 if name=='owner_disarm'else 1]=value
                if value!=0:report['cleanup_errors'].append({'operation':name,'returncode':value})
            except BaseException as cleanup_exc:report['cleanup_errors'].append({'operation':name,'error':repr(cleanup_exc)})
        if report['cleanup_errors']:
            report.update(completed=True,passed=False)
            if not report.get('primary_error'):report['error']='Diagnostic mode cleanup failed'
        save()
        if report['cleanup_errors']and not report.get('primary_error'):raise RuntimeError('Diagnostic mode cleanup failed')
