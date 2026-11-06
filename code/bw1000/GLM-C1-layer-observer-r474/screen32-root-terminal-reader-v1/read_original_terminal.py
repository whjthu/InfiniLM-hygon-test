"""Original906 terminal reader. Stdlib only; never loads Native or executes jobs.
Requires actual terminal accounting and original plan bytes. CPU fixtures qualify
only validation logic; actual906 original artifacts have not been read here.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,re,statistics
H=lambda b:hashlib.sha256(b).hexdigest()
JH=lambda x:H(json.dumps(x,separators=(',',':')).encode())
LABEL='GLM-global-TP16-C1-observer-screen32-r474-v1'
MF='a94ef2b3e69c38087e3e434e0761a7aa1792e65c85f2fcbc1cb44776149ff560'
GMF='98505c1b0d5a08dc5cde88ddc9e59e4389a044a05b524e56790fc8ba2a27c305'
NATIVE='955e6a14060c636196a5afb1603f6b7405a63215dd5736f11c6f2f39adb7b18f'
GPU='910371.906';KEEPER='910407.1121';EPOCH=1793949098.9437428
FLOOR=8589934592;TOTAL=68702699520;NODES=['a02r3n15','a02r3n16']
ORDINALS=[1,2,6,12,22,28,34,35,43,55,60,69,71,79,83,103,128,129,146,147,150,162,168,179,187,189,200,214,221,234,238,251]
PROFILE_NAMES=['incoming_host_s','layers_host_s','outgoing_host_s','incoming_stream_s','layers_stream_s','outgoing_stream_s']
SELECTED={1,2,3,4,8,16,32,64,128,256,512,1024,2048,4096}
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}

def accounting(data):
 """Explicit columns prevent treating JobName or ExitCode as State."""
 cols=data['accounting_columns'];assert cols in [
 ['JobIDRaw','State','ExitCode','ElapsedRaw'],
 ['JobIDRaw','JobName','State','ExitCode','ElapsedRaw'],
 ['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList']]
 rows={}
 for line in data['accounting']:
  v=line.split('|')if isinstance(line,str)else line;assert len(v)==len(cols)
  row=dict(zip(cols,v));key=row['JobIDRaw'];assert key not in rows;row['ElapsedRaw']=int(row['ElapsedRaw']);assert row['ElapsedRaw']>=0
  row['normalized_state']=row['State'].split()[0].rstrip('+');rows[key]=row
 return rows

def successful(row):return row['normalized_state']=='COMPLETED'and row['ExitCode']=='0:0'

def collector_state(w):
 observations=w.get('terminal_accounting_observations',[]);assert len(observations)<=12
 for row in observations:
  assert row is None or(len(row)==5 and row[0]==GPU and row[1]==LABEL)
 if w['passed']:
  assert w['GPU']==GPU and w['srun_rc']==0 and w['terminal_visibility_complete']is True
  assert w['GPU_accounting'][0]==GPU and w['GPU_accounting'][2:4]==['COMPLETED','0:0']and observations and observations[-1]==w['GPU_accounting']
 return dict(original_passed=w['passed'],terminal_visibility_complete=w.get('terminal_visibility_complete'),original_observed_states=[r[2]if r else'MISSING'for r in observations],original_final_row=w.get('GPU_accounting'))

def bucket(b,count,prompt,fragments,prefill):
 if count==0:assert b=={},'zero decode bucket must stay empty';return
 assert b['count']==count and len(b['selected'])<=24
 for v in [b['sum_s'],b['min_s'],b['max_s']]:assert len(v)==6 and all(math.isfinite(x)and x>=0 for x in v)
 for i in range(6):
  lo,hi,total=b['min_s'][i],b['max_s'][i],b['sum_s'][i];tol=1e-7+abs(total)*1e-10
  assert lo<=hi and lo*count-tol<=total<=hi*count+tol
 expected=[]
 for i in range(1,count+1):
  tokens=fragments[i-1]['tokens']if prefill else 1
  if tokens>1 or i in SELECTED:expected.append(i)
 expected=expected[:24];assert [v['index']for v in b['selected']]==expected
 for v in b['selected']:
  i=v['index'];assert v['tokens']==(fragments[i-1]['tokens']if prefill else 1)
  assert v['past']==(fragments[i-1]['offset']if prefill else prompt+i-1)
  assert len(v['seconds'])==6 and all(math.isfinite(x)and b['min_s'][j]-1e-9<=x<=b['max_s'][j]+1e-9 for j,x in enumerate(v['seconds']))

def clocks(ids,ticks,eos,special,begin_ns):
 assert type(begin_ns)is int and begin_ns>0 and len(ids)==len(ticks)>0
 assert all(type(x)is int and x>=begin_ns for x in ticks)and all(b>a for a,b in zip(ticks,ticks[1:]))
 visible=[t for token,t in zip(ids,ticks)if token not in set(eos)|set(special)]
 itl=sorted((b-a)/1e9 for a,b in zip(visible,visible[1:]))
 pct=lambda f:itl[math.ceil(f*len(itl))-1]if itl else None
 return dict(visible=len(visible),rate=(len(visible)-1)*1e9/(visible[-1]-visible[0])if len(visible)>1 else None,TTFT_s=(visible[0]-begin_ns)/1e9 if visible else None,ITL_P95_s=pct(.95),ITL_P99_s=pct(.99))

def equal_optional(a,b):
 assert(a is None)==(b is None)
 if a is not None:assert math.isfinite(a)and math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-10)

def frame_report(q,rank,prompt,fragments,generated):
 expected=min(3,max(0,generated-1));assert q['global_tp_rank']==rank
 assert q['enabled']is True and q['frame_cap']==3 and q['event_cap_per_frame']==512 and q['events_allocated']==512
 assert q['pending_next_decode']is False and q['total_frames_used']==len(q['samples'])==expected
 assert q['coverage']==[0,0,0,expected]and q['required']==[0,0,0,3]and q['missing_is_UNKNOWN']is True
 assert q['sequence']==fragments+generated-1
 result=[]
 for i,f in enumerate(q['samples']):
  assert f['tokens']==1 and f['stage']==0 and f['rank']==rank%8 # legacy local device; outer field above is global TP rank
  assert f['row_past_count']==[[0,prompt+i,1]]and f['sequence']==fragments+i+1
  marks=f['marks'];assert len(marks)==f['event_count']and 157<=len(marks)<=250
  assert marks[0]['name']=='CAPI.owner.begin'
  hs=[m['owner_host_s']for m in marks];gs=[m['gpu_from_first_s']for m in marks]
  assert all(math.isfinite(v)and v>=0 for v in hs+gs)and abs(gs[0])<=2e-6
  assert all(b>=a for a,b in zip(hs,hs[1:]))and all(b+2e-6>=a for a,b in zip(gs,gs[1:]))
  begin={};end={};kinds={}
  for j,m in enumerate(marks):
   a=re.fullmatch(r'layer[.](\d+)[.](full|shared)[.]begin',m['name']);b=re.fullmatch(r'layer[.](\d+)[.]end',m['name'])
   if a:
    layer=int(a[1]);assert layer not in begin;begin[layer]=j;kinds[layer]=a[2]
   if b:
    layer=int(b[1]);assert layer not in end;end[layer]=j
  assert set(begin)==set(end)==set(range(78))
  assert [begin[x]for x in range(78)]==sorted(begin.values())
  for l in range(78):assert begin[l]<end[l]and(l==77 or end[l]<begin[l+1])
  assert f['full_detail_layer']==next(l for l in range(78)if kinds[l]=='full')
  assert f['shared_detail_layer']==next(l for l in range(78)if kinds[l]=='shared')
  spans=[dict(layer=l,kind=kinds[l],host_enclosure_s=hs[end[l]]-hs[begin[l]],stream_enclosure_s=max(0.,gs[end[l]]-gs[begin[l]]))for l in range(78)]
  adjacent=[dict(before=marks[j-1]['name'],after=marks[j]['name'],host_delta_s=hs[j]-hs[j-1],stream_delta_s=max(0.,gs[j]-gs[j-1]))for j in range(1,len(marks))]
  result.append(dict(sequence=f['sequence'],past=prompt+i,events=len(marks),legacy_local_device_rank=f['rank'],global_tp_rank=rank,layer_enclosures=spans,adjacent_mark_windows=adjacent,window_scope='Overlapping host/event enclosures on original stream, with enqueue idle; not exclusive GPU busy or FLOPs.'))
 return dict(expected_frames=expected,observed_frames=len(result),coverage_complete=expected==3,coverage='COMPLETE_THREE'if expected==3 else'UNKNOWN_SHORT_NATURAL_DIAGNOSTIC',frames=result)

def schedule():return [dict(case_index=0,diagnostic=True,screen_index=None)]+[dict(case_index=i,diagnostic=False,screen_index=i)for i in range(32)]
def phases():return ['created','loaded']+['case'+str(i)for i in range(33)]+['quiescent','closed']

def request(r,p,frozen,arm,index,rank,common,inp,forward_begin):
 prompt=p['prompt_token_ids'];ids=r['generated_token_ids'];ticks=r['sampled_ns'];fr=r['prefill_fragment_plan'];assert r['completed']and r['reset_before']and r['reset_after']
 assert r['ordinal']==p['ordinal']==frozen['ordinal']and r['case_id']==p['case_id']==frozen['case_id']and r['suite']==p['suite']==frozen['suite']
 assert r['prompt_tokens']==len(prompt)==frozen['prompt_tokens']and r['prompt_IDs_SHA256']==JH(prompt)==frozen['prompt_IDs_SHA256']and r['body_SHA256']==frozen['body_SHA256']
 assert r['planned_max_new_tokens']==4096 and r['runtime_prefill_chunk']==512 and r['original_request_prefill_window']==4096
 assert p['max_new_tokens']==4096 and p['prefill_window']==4096 and p['body']['temperature']==0.0 and len(prompt)+4096<=32768
 assert r['diagnostic']==r['observer_diagnostic']==arm['diagnostic']and r['screen_index']==arm['screen_index']
 assert r['arm']==('diagnostic'if arm['diagnostic']else'screen')and r['pair']==-1 and r['warmup']is False and r['owner_cores']==4
 assert r['owner_CPUmask']==common[(rank%8)*8:(rank%8)*8+4]and r['pipeline_profile_names']==PROFILE_NAMES
 expected=[dict(offset=o,tokens=len(prompt[o:o+512]),IDs_SHA256=JH(prompt[o:o+512]))for o in range(0,len(prompt),512)];assert fr==expected
 assert 0<len(ids)<=4096 and all(type(v)is int and 0<=v<154880 for v in ids)and r['generated_token_ids_SHA256']==JH(ids)
 natural=ids[-1]in inp['eos_token_ids'];assert not any(v in inp['eos_token_ids']for v in ids[:-1]);assert r['natural_EOS']==natural and r['cap_hit']==(not natural and len(ids)==4096)
 assert natural or r['cap_hit'];count=len(fr)+len(ids)-1
 assert r['Native_forward_begin']==forward_begin and r['Native_forward_end']==forward_begin+count and r['past_after']==len(prompt)+len(ids)-1
 assert len(r['prefill_CAPI_s'])==len(fr)and len(r['decode_CAPI_s'])==len(ids)-1
 assert all(math.isfinite(x)and x>=0 for x in r['prefill_CAPI_s']+r['decode_CAPI_s'])
 bucket(r['pipeline_prefill'],len(fr),len(prompt),fr,True);bucket(r['pipeline_decode'],len(ids)-1,len(prompt),fr,False)
 timing=clocks(ids,ticks,inp['eos_token_ids'],inp['tokenizer']['special_ids'],r['request_enter_ns']);assert timing['visible']==r['visible_nonEOS_nonspecial'];equal_optional(r['backend_decode_tok_s'],timing['rate']);equal_optional(r['backend_sampled_TTFT_s'],timing['TTFT_s'])
 assert math.isfinite(r['whole_backend_s'])and r['whole_backend_s']>=0 and(timing['TTFT_s']is None or timing['TTFT_s']<=r['whole_backend_s']+1e-6)
 if arm['diagnostic']:
  frames=frame_report(r['observer_report'],rank,len(prompt),len(fr),len(ids));assert r['observer_mark_calls_before']==0 and r['observer_mark_calls_after']==frames['observed_frames']
 else:frames=None;assert r['observer_mark_calls_before']==r['observer_mark_calls_after']<=3
 row=dict(index=index,ordinal=r['ordinal'],suite=r['suite'],diagnostic=arm['diagnostic'],screen_index=arm['screen_index'],prompt_tokens=len(prompt),prefill_fragments=[f['tokens']for f in fr],generated_including_EOS=len(ids),visible_tokens=timing['visible'],natural_EOS=natural,cap_hit=r['cap_hit'],backend_decode_tok_s=timing['rate'],backend_TTFT_s=timing['TTFT_s'],backend_ITL_P95_s=timing['ITL_P95_s'],backend_ITL_P99_s=timing['ITL_P99_s'],request_enter_ns=r['request_enter_ns'],whole_backend_s=r['whole_backend_s'],prefill_CAPI_sum_s=sum(r['prefill_CAPI_s']),decode_CAPI_sum_s=sum(r['decode_CAPI_s']),output_IDs_SHA256=r['generated_token_ids_SHA256'],profile_prefill={k:v for k,v in r['pipeline_prefill'].items()if k!='selected'},profile_decode={k:v for k,v in r['pipeline_decode'].items()if k!='selected'})
 return row,frames,count,len(ids)

def screen_summary(screen):
 assert len(screen)<=32 and all(not r['diagnostic']for r in screen)
 return dict(planned=32,validated=len(screen),correct=sum(r.get('correct',False)for r in screen),parse_failure=sum(r.get('parse_failure',False)for r in screen),natural_EOS=sum(r['natural_EOS']for r in screen),cap_hit=sum(r['cap_hit']for r in screen),missing_or_unvalidated=32-len(screen),strict_gt15=sum(r['backend_decode_tok_s']is not None and r['backend_decode_tok_s']>15 for r in screen))

def audit_post(w,rows,owners,workers,strict=False):
 post=w.get('post')or w.get('failure_post');assert post and post['raw']['rc']==0 and {v['node']for v in post['nodes']}==set(NODES)
 expected=json.loads(next(a.split('=',1)[1]for a in post['raw']['argv']if a.startswith('GLM470_EXPECTED_PIDS=')));assert set(expected)<=set(NODES)
 steps={n['actual_step']for n in post['nodes']};assert len(steps)==1;post_step=steps.pop();assert re.fullmatch(r'910371[.]\d+',post_step)and post_step not in [GPU,'910371.905']and successful(rows[post_step])
 for n in post['nodes']:
  assert n['KFD_available']and not n['kfd_pids']and not n['matching_Model_processes']and not n['expected_owned_PIDs_still_live']and not n['expected_UIDsocket_listeners_still_live']
  assert {str(k)for k in n['ports_bindable']}=={'43843','43844'}and all(n['ports_bindable'].values())
  before=next(v for v in w['before']['nodes']if v['node']==n['node']);assert len(n['physical_PCI'])==len(set(n['physical_PCI']))==8 and set(n['physical_PCI'])==set(before['physical_PCI'])
 for i,v in enumerate(workers):
  es=expected.get(NODES[i],[]);assert len(es)==len({e['PID']for e in es})
  if v:
   assert any(e['PID']==v['PID']and str(e['startticks'])==str(v['startticks'])for e in es)
   for child in v.get('owners',[]):assert any(e['PID']==child['PID']and str(e['startticks'])==str(child['startticks'])for e in es)
  if strict:assert len(es)==9
 for q in owners:
  if q:
   e=next(e for e in expected.get(q['node'],[])if e['PID']==q['PID']);assert str(e['startticks'])==str(q['startticks'])and set(e['socket_inodes'])==set(q.get('owned_socket_inodes',[]))
 return post_step

def main(root,accounting_file,private_plan,source,watch_path=None):
 root=Path(root);source=Path(source)
 rows=accounting(json.loads(Path(accounting_file).read_bytes()));assert rows[GPU]['normalized_state']in TERMINAL and rows[KEEPER]['normalized_state']in TERMINAL,'actual GPU and keeper terminal required; no stale RUNNING consumption'
 sm=(source/'SOURCE_MANIFEST.json').read_bytes();assert H(sm)==MF and H((source/'GPU_MANIFEST.json').read_bytes())==GMF
 mf=json.loads(sm);assert all(H((source/n).read_bytes())==h for n,h in mf['files_SHA256'].items())
 inp=json.loads((source/'INPUT_BINDING.json').read_bytes());raw=Path(private_plan).read_bytes();assert H(raw)==inp['private_plan']['SHA256']and len(raw)==inp['private_plan']['bytes'];plan=json.loads(raw)
 assert plan['planned']==len(plan['requests'])==32 and [p['ordinal']for p in plan['requests']]==ORDINALS
 assert len({p['case_id']for p in plan['requests']})==32 and sorted(p['suite']for p in plan['requests'])==['gsm8k']*16+['mmlu']*16
 for p,f in zip(plan['requests'],inp['private_plan']['requests']):assert all(p[k]==f[k]for k in ['ordinal','case_id','suite','prompt_tokens','prompt_IDs_SHA256','original_private_SHA256','body_SHA256','max_new_tokens','prefill_window'])
 originals={};index=[]
 for name in [LABEL+'-watch.json']+['owner-%02d.json'%i for i in range(16)]+['worker-%d.json'%i for i in range(2)]:
  path=Path(watch_path)if watch_path and name==LABEL+'-watch.json'else root/name
  if path.exists():
   b=path.read_bytes();originals[name]=json.loads(b);index.append(dict(name=name,original_path=str(path),bytes=len(b),SHA256=H(b)))
 w=originals[LABEL+'-watch.json'];assert w['completed']and w['epoch']==EPOCH and w['CPU_keeper']==KEEPER
 assert w['GPU']in(None,GPU)and w['planned']==33;collector=collector_state(w)
 owners=[originals.get('owner-%02d.json'%i)for i in range(16)];workers=[originals.get('worker-%d.json'%i)for i in range(2)]
 spec=importlib.util.spec_from_file_location('frozen_scoring_r474',source/'scoring.py');sc=importlib.util.module_from_spec(spec);spec.loader.exec_module(sc)
 ranks=[];root_rows=[];hashes=[];diagnostics=[];all_valid=True;keyhashes=set();uidhashes=set();provider_pins=json.loads((source/'CPU_BINDING.json').read_bytes())['actual_project_DSO_SHA256']
 for rank,q in enumerate(owners):
  errors=[];validated=[];frames=None;forward_count=0;controls=0
  if q is None:ranks.append(dict(rank=rank,missing=True,validation_errors=['missing original primary owner']));all_valid=False;continue
  try:
   assert q['global_rank']==rank and q['node']==NODES[rank//8]and q['device']==rank%8 and q['actual_GPU']==GPU and q['gpu_epoch']==EPOCH
   assert q['owner_TID']==q['PID']and q['UID']==154459 and q['source_manifest_SHA256']==MF and q['GPU_manifest_SHA256']==GMF and q['Native_SHA256']==NATIVE and q['private_input_SHA256']==inp['private_plan']['SHA256']
   assert q['all78layers_loaded']and q['loaded_parameters']==q['checkpoint_key_count']==117060 and q['actual_layers']==list(range(78))
   geo=q['constructor_geometry'];post=q['postload_geometry'];assert post['loaded_parameters']==117060 and post['Model_READY']and geo['TP']==geo['world_size']==16 and geo['PP']==1 and geo['world_rank']==geo['tp_rank']==rank and geo['rows']==1 and geo['capacity_per_row']==32768
   assert geo['layer_begin']==0 and geo['layer_end']==78 and geo['local_attention_heads']==4 and geo['local_index_heads']==2 and geo['embedding_vocab_begin']==rank*9680 and geo['embedding_vocab_end']==(rank+1)*9680
   assert geo['replicated_embedding_BF16_bytes']==0 and geo['embedding_BF16_local_bytes']==118947840 and geo['cache_logical_bytes']['sum']==8355053568
   assert set(q['CAPI_bindings'])=={'id_size','id','create','error','keys','load_batch','finish','forward','control','reset','geometry','destroy'}and all(v['SHA256']==NATIVE for v in q['CAPI_bindings'].values())and set(q['observer_CAPI_bindings'])=={'prepare','policy','next_decode','read'}and all(v['SHA256']==NATIVE for v in q['observer_CAPI_bindings'].values())
   assert set(q['actual_provider_maps'])==set(provider_pins)and all(q['actual_provider_maps'][p]['SHA256']==h for p,h in provider_pins.items())
   assert q['profile_getter_binding']['SHA256']==NATIVE and q['profile_getter_binding']['count']==6 and q['profile_getter_binding']['symbol']=='modern_glm_pipeline_last_profile'
   assert q['observer_pool_events']==512 and q['observer_source_event_bound']==250
   assert q['resets']==66 and len(q['requests'])==33 and q['observer_disabled_for_screen']and q['completed']and q['passed']and q['normal_close']and q['worker_shutdown_completed']and q['Runtime_weak_expired_CABI_assertion']and not q.get('error')
   assert q['weights']and not q['Model_READY']and not q['Model_qualified']and not q['HTTP']
   dm=q['driver_memory'];assert dm['min_observed']>=FLOOR and not dm['errors']and dm['samples']>0 and math.isfinite(dm['max_gap_s'])and dm['max_gap_s']>=0
   assert dm['min_observed']==min(v['free']for v in dm['lowpoints'])==min(v['min_free']for v in dm['byphase'].values())
   assert all(b['floor']==FLOOR and b['free']>=FLOOR and b['total']==TOTAL for b in q['memory_boundaries'])
   selected_boundaries=[b for b in q['memory_boundaries']if b['phase']in ['before_observer_prepare','after_observer_prepare']];assert len(selected_boundaries)==2;labels={b['phase']:b for b in selected_boundaries};assert set(labels)=={'before_observer_prepare','after_observer_prepare'}
   common=q['original_common_CPUmask'];assert len(common)>=64 and len(set(common))==len(common)and q['initial_owner_CPUmask']==common[(rank%8)*8:(rank%8)*8+4]
   keyhashes.add(q['checkpoint_keys_SHA256']);uidhashes.add(tuple(q['UID_SHA256']))
   for i,(r,a)in enumerate(zip(q['requests'],schedule())):
    ci=a['case_index'];row,f,fc,cc=request(r,plan['requests'][ci],inp['private_plan']['requests'][ci],a,i,rank,common,inp,forward_count);forward_count+=fc;controls+=cc;validated.append(row)
    e=next(e for e in q['events']if e['phase']=='request.enter'and e['ordinal']==i);assert r['request_enter_ns']/1e9<=e['monotonic']+1e-6
    if f is not None:frames=f;assert q['observer_diagnostic']==r['observer_report']
    if rank==0:
     score=sc.grade(plan['requests'][ci]['case'],r['output_text'],natural_EOS=r['natural_EOS'],cap_hit=r['cap_hit'],unexpected_tool=False);assert score==r['score'];row.update(correct=score['correct'],parse_failure=score['parse_failure']);root_rows.append(row)
   assert q['Native_forward_attempts']==q['Native_forward_completed']==forward_count and q['control_calls']==controls
   assert all(r['observer_mark_calls_before']==r['observer_mark_calls_after']==q['requests'][0]['observer_mark_calls_after']for r in q['requests'][1:])
  except Exception as e:errors.append(type(e).__name__+': '+str(e)[:200]);all_valid=False
  ranks.append(dict(rank=rank,node=q.get('node'),PCI=q.get('physical_PCI'),PID=q.get('PID'),startticks=q.get('startticks'),producer_passed=q.get('passed'),producer_normal_close=q.get('normal_close'),producer_error=q.get('error'),saved_requests=len(q.get('requests',[])),independently_validated_requests=len(validated),Native_forward_expected=forward_count,control_expected=controls,driver_min_sampled=q.get('driver_memory',{}).get('min_observed'),driver_max_gap_s=q.get('driver_memory',{}).get('max_gap_s'),loading_seconds=q.get('loading_seconds'),validation_errors=errors))
  hashes.append([r['output_IDs_SHA256']for r in validated]);diagnostics.append(dict(rank=rank,report=frames))
 cross_errors=[];post_clean=False;post_step=None
 try:
  assert len(hashes)==16 and all(len(h)==33 and h==hashes[0]for h in hashes)and len(keyhashes)==len(uidhashes)==1
  assert len({(r['node'],r['PCI'])for r in ranks})==16
  for i,v in enumerate(workers):assert v['completed']and v['passed']and v['actual_GPU']==GPU and v['gpu_epoch']==EPOCH and not v.get('forced_TERM')and not v.get('forced_KILL')and [r['phase']for r in v['rounds']]==phases()
  assert [r['digest']for r in workers[0]['rounds']]==[r['digest']for r in workers[1]['rounds']]
 except Exception as e:cross_errors.append('Model cohort: '+type(e).__name__+': '+str(e)[:200]);all_valid=False
 try:
  post_step=audit_post(w,rows,owners,workers,strict=all_valid);post_clean=True
 except Exception as e:cross_errors.append(type(e).__name__+': '+str(e)[:200]);all_valid=False
 producer_requests=[];raw_root=owners[0].get('requests',[])if owners[0]else[]
 for i,a in enumerate(schedule()):
  r=raw_root[i]if i<len(raw_root)else None
  producer_requests.append(dict(index=i,diagnostic=a['diagnostic'],screen_index=a['screen_index'],present=r is not None,completed=bool(r and r.get('completed')),natural_EOS=r.get('natural_EOS')if r else None,cap_hit=r.get('cap_hit')if r else None,producer_score={k:r.get('score',{}).get(k)for k in ['correct','parse_failure']}if r else None,qualification='Original producer status only; independent audit is separate, no raw text or token IDs exported.'))
 screen=[r for r in root_rows if not r['diagnostic']];diag=[r for r in root_rows if r['diagnostic']]
 passed=all_valid and successful(rows[GPU])and successful(rows[KEEPER])and w['passed']and w.get('normal16',False)and not w.get('forced_cleanup')and post_clean
 summary=screen_summary(screen)
 return dict(schema='GLM906-original-screen32-root-v1',GPU=GPU,keeper=KEEPER,GPU_accounting=rows[GPU],keeper_accounting=rows[KEEPER],GPU_accounting_success=successful(rows[GPU]),keeper_accounting_success=successful(rows[KEEPER]),keeper_original_producer_passed=w['passed'],original_collector_observations=collector,original_watch_error=w.get('error'),engineering_full_entry_passed=passed,all16_original_owner_validation_passed=all_valid,post_step=post_step,post_clean=post_clean,original_artifacts=index,original_artifact_count=len(index),accounting_original_SHA256=H(Path(accounting_file).read_bytes()),private_plan_SHA256=H(raw),source_manifest_SHA256=MF,GPU_manifest_SHA256=GMF,Native_SHA256=NATIVE,planned_requests=33,diagnostic_planned=1,unique_screen_planned=32,root_requests=root_rows,original_producer_planned_status=producer_requests,screen32=summary,diagnostic_excluded_from_quality_denominator=diag,rank_validation=ranks,cross_validation_errors=cross_errors,diagnostic_frames=diagnostics,normal_destroy_owners=sum(bool(q and q.get('normal_close'))for q in owners),performance_baseline_paired=False,performance_win_proven=False,wide_quality_qualified=False,formal_NI_proven=False,client_SLO_qualified=False,independent_rawlogit_or_SDK_replay=False,promote=False,limits=['Frozen scorer reused, not independent semantic/SDK replay. Wrong answers/parsefailures/caps remain in32screen denominator; diagnostic repetition is excluded.','Missing/invalid original evidence cannot become PASS. Actual GPU and keeper accounting are independent; later clean evidence never washes original keeper failure.','Event/host windows overlap and include enqueue idle;512eventpool persists during screen, not oldNative no-pool performance baseline.','Driver boundaries/samples/maxgap are observed; initialization/subinterval peaks and future fit remain UNKNOWN.','Reader source/CPU fixtures are engineering only until executed against exact original906 terminal artifacts.'])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--original-directory',required=True);p.add_argument('--accounting',required=True);p.add_argument('--private-plan',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--watch');a=p.parse_args();o=Path(a.output);assert not o.exists();r=main(a.original_directory,a.accounting,a.private_plan,a.source,a.watch);r['reader_SHA256']=H(Path(__file__).read_bytes());o.write_text(json.dumps(r,indent=2)+'\n',encoding='utf8');print(json.dumps({k:r[k]for k in ['GPU','keeper','engineering_full_entry_passed','GPU_accounting_success','keeper_accounting_success','post_clean','screen32']}))
