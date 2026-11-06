"""Read original900 ten complete resident requests and three paired units.
Stdlib only; no Native/SDK/compiler/Slurm/HTTP/tokenizer/model execution.
Original producer profile buckets and frozen scorer are explicitly reused.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,statistics
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
JH=lambda x:hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
LABEL='GLM-global-TP16-resident-ownercores-r473-v2'
MF='11d2d38545b9edff7034cf1ddc7f6e515029c8427c2f8989c036ec72c52e62d3'
GMF='3ba6e8fba67e7cde7c6fff195e729d35ed241088526170da1816ba781f2c0166'
NATIVE='af4f13006dd123b62d56426d82830cae792c27a60127897b84fcf9c14c263983'
FLOOR=8589934592
NODES=['a02r3n15','a02r3n16']

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def bucket(b,count,prompt,fragments,prefill):
 assert b['count']==count and len(b['selected'])<=24
 for values in [b['sum_s'],b['min_s'],b['max_s']]:
  assert len(values)==6 and all(math.isfinite(x)and x>=0 for x in values)
 for i in range(6):
  assert b['min_s'][i]<=b['max_s'][i]
  assert b['min_s'][i]*count-1e-8<=b['sum_s'][i]<=b['max_s'][i]*count+1e-8
 seen=[]
 for v in b['selected']:
  index=v['index'];assert 1<=index<=count and index not in seen;seen.append(index)
  assert len(v['seconds'])==6 and all(math.isfinite(x)and b['min_s'][i]-1e-10<=x<=b['max_s'][i]+1e-10 for i,x in enumerate(v['seconds']))
  assert v['tokens']==(fragments[index-1]['tokens']if prefill else 1)
  assert v['past']==(fragments[index-1]['offset']if prefill else prompt+index-1)
 assert seen==sorted(seen)
 if prefill:assert seen==list(range(1,count+1))


def read(root,accounting,private_plan,raw_plan_receipt):
 root=Path(root);source=Path(__file__).resolve().parents[1]/'paired-ownercores-v2'
 assert H(source/'SOURCE_MANIFEST.json')==MF and H(source/'GPU_MANIFEST.json')==GMF
 sm=json.loads((source/'SOURCE_MANIFEST.json').read_bytes());assert all(H(source/n)==v for n,v in sm['files_SHA256'].items())
 trial=module('frozen_resident_trial',source/'resident_trial.py');sc=module('frozen_scoring',source/'scoring.py');schedule=trial.schedule()
 index=json.loads((root/'original-index.json').read_bytes());assert len(index)==19
 expected_names={LABEL+'-watch.json'}|{'owner-%02d.json'%i for i in range(16)}|{'worker-%d.json'%i for i in range(2)}
 assert {v['name']for v in index}==expected_names
 for v in index:assert H(root/v['name'])==v['SHA256']and(root/v['name']).stat().st_size==v['bytes']
 w=json.loads((root/(LABEL+'-watch.json')).read_bytes());assert w['completed']and w['passed']and w['GPU']=='910371.900'and w['CPU_keeper']=='910407.1111'and w['srun_rc']==0 and w['planned']==10 and w['normal16']and w['terminal_visibility_complete']and not w.get('forced_cleanup')
 assert w['GPU_accounting'][2:4]==['COMPLETED','0:0']and w['terminal_accounting_observations'][-1]==w['GPU_accounting']
 observed=json.loads(Path(accounting).read_bytes());assert observed['rc']==0;observed=observed['out'];observed=json.loads(observed)if isinstance(observed,str)else observed
 rows={r.split('|')[0]:r.split('|')for r in observed['accounting']}
 assert rows['910371.900'][2:4]==rows['910407.1111'][2:4]==['COMPLETED','0:0']
 inp=json.loads((source/'INPUT_BINDING.json').read_bytes());plan=json.loads(Path(private_plan).read_bytes());assert len(plan['requests'])==plan['planned']==2
 raw=json.loads(Path(raw_plan_receipt).read_bytes())['out'].encode();candidates=[raw,raw[:-1]];exact=[b for b in candidates if len(b)==inp['private_plan']['bytes']and hashlib.sha256(b).hexdigest()==inp['private_plan']['SHA256']];assert len(exact)==1 and json.loads(exact[0])==plan
 expected=json.loads(next(v.split('=',1)[1]for v in w['post']['raw']['argv']if v.startswith('GLM470_EXPECTED_PIDS=')))
 assert w['post']['raw']['rc']==0 and set(expected)==set(NODES)
 workers=[json.loads((root/('worker-%d.json'%i)).read_bytes())for i in range(2)]
 owners=[json.loads((root/('owner-%02d.json'%i)).read_bytes())for i in range(16)]
 rank_rows=[];output_hashes=[];keyhashes=set();uidhashes=set();root_rows=[]
 for rank,q in enumerate(owners):
  node=NODES[rank//8];geo=q['constructor_geometry'];post=q['postload_geometry']
  assert q['global_rank']==rank and q['node']==node and q['device']==rank%8 and q['actual_GPU']=='910371.900'and q['gpu_epoch']==w['epoch']
  assert q['owner_TID']==q['PID']and q['UID']==154459 and q['completed']and q['passed']and q['normal_close']and q['worker_shutdown_completed']and q['Runtime_weak_expired_CABI_assertion']and not q.get('error')
  assert q['weights']and not q['Model_READY']and not q['Model_qualified']and not q['HTTP']
  assert q['all78layers_loaded']and q['loaded_parameters']==q['checkpoint_key_count']==post['loaded_parameters']==117060 and q['actual_layers']==list(range(78))and post['Model_READY']
  assert q['private_input_SHA256']==inp['private_plan']['SHA256']and q['source_manifest_SHA256']==MF and q['GPU_manifest_SHA256']==GMF and q['Native_SHA256']==NATIVE and all(v['SHA256']==NATIVE for v in q['CAPI_bindings'].values())
  getter=q['profile_getter_binding'];assert getter['symbol']=='modern_glm_pipeline_last_profile'and getter['SHA256']==NATIVE and getter['count']==6
  assert geo['TP']==geo['world_size']==16 and geo['PP']==1 and geo['world_rank']==geo['tp_rank']==rank and geo['rows']==1 and geo['capacity_per_row']==32768 and geo['local_attention_heads']==4 and geo['local_index_heads']==2 and geo['layer_begin']==0 and geo['layer_end']==78
  assert geo['Model_constructed']and geo['cache_allocated']and not geo['weights_loaded']and geo['replicated_embedding_BF16_bytes']==0 and geo['embedding_BF16_local_bytes']==118947840 and geo['embedding_vocab_begin']==rank*9680 and geo['embedding_vocab_end']==(rank+1)*9680 and geo['cache_logical_bytes']['sum']==8355053568
  assert q['resets']==20 and len(q['requests'])==10 and q['Native_forward_attempts']==q['Native_forward_completed']
  assert all(v['floor']==FLOOR and v['free']>=FLOOR and v['total']==68702699520 for v in q['memory_boundaries'])
  dm=q['driver_memory'];assert dm['min_observed']>=FLOOR and not dm['errors']and dm['samples']>0
  assert dm['min_observed']==min(v['free']for v in dm['lowpoints'])==min(v['min_free']for v in dm['byphase'].values())
  owner=next(v for v in expected[node]if v['PID']==q['PID']);assert owner['startticks']==q['startticks']and set(owner['socket_inodes'])==set(q['owned_socket_inodes'])and len(expected[node])==9
  keyhashes.add(q['checkpoint_keys_SHA256']);uidhashes.add(tuple(q['UID_SHA256']));hashes=[];forward_count=0;controls=0
  common=q['original_common_CPUmask'];assert len(common)>=64 and q['initial_owner_CPUmask']==trial.owner_mask(common,rank%8,4)
  for i,(r,arm)in enumerate(zip(q['requests'],schedule)):
   p=plan['requests'][arm['case_index']];frozen=inp['private_plan']['requests'][arm['case_index']];ids=r['generated_token_ids'];ticks=r['sampled_ns'];prompt=p['prompt_token_ids'];fr=r['prefill_fragment_plan']
   assert r['ordinal']==p['ordinal']==frozen['ordinal']and r['case_id']==p['case_id']==frozen['case_id']and r['suite']==p['suite']==frozen['suite']
   assert r['prompt_tokens']==len(prompt)==frozen['prompt_tokens']and r['prompt_IDs_SHA256']==JH(prompt)==frozen['prompt_IDs_SHA256']and r['body_SHA256']==frozen['body_SHA256']
   assert r['planned_max_new_tokens']==4096 and r['runtime_prefill_chunk']==512 and r['original_request_prefill_window']==4096 and r['reset_before']and r['reset_after']and r['completed']
   for name in ['arm','pair','warmup','owner_cores']:assert r[name]==arm[name]
   assert r['owner_CPUmask']==trial.owner_mask(common,rank%8,arm['owner_cores'])
   assert r['pipeline_profile_names']==list(trial.PROFILE_NAMES)
   offset=0
   for f in fr:
    chunk=prompt[offset:offset+512];assert f==dict(offset=offset,tokens=len(chunk),IDs_SHA256=JH(chunk));offset+=len(chunk)
   assert offset==len(prompt)and len(fr)==len(r['prefill_CAPI_s'])and all(math.isfinite(v)and v>=0 for v in r['prefill_CAPI_s']+r['decode_CAPI_s'])
   assert 0<len(ids)<=4096 and len(ids)==len(ticks)and all(type(v)is int and 0<=v<154880 for v in ids)and all(b>a for a,b in zip(ticks,ticks[1:]))
   assert r['generated_token_ids_SHA256']==JH(ids)and ids[-1]in inp['eos_token_ids']and r['natural_EOS']and not r['cap_hit']
   fc=len(fr)+len(ids)-1;forward_count+=fc;controls+=len(ids)
   assert r['Native_forward_end']-r['Native_forward_begin']==fc and len(r['decode_CAPI_s'])==len(ids)-1 and r['past_after']==len(prompt)+len(ids)-1
   assert r['Native_forward_begin']==forward_count-fc and r['Native_forward_end']==forward_count
   bucket(r['pipeline_prefill'],len(fr),len(prompt),fr,True);bucket(r['pipeline_decode'],len(ids)-1,len(prompt),fr,False)
   visible=[t for token,t in zip(ids,ticks)if token not in set(inp['eos_token_ids']+inp['tokenizer']['special_ids'])]
   rate=(len(visible)-1)*1e9/(visible[-1]-visible[0]);assert len(visible)==r['visible_nonEOS_nonspecial']and math.isclose(rate,r['backend_decode_tok_s'],rel_tol=1e-12)
   begin=next(e['monotonic']for e in q['events']if e['phase']=='request.enter'and e['ordinal']==i)
   inferred=visible[0]/1e9-r['backend_sampled_TTFT_s'];first=next(e['monotonic']for e in q['events']if e['phase']=='forward.enter'and e['monotonic']>=begin)
   assert begin<=inferred<=first<=visible[0]/1e9
   if rank==0:
    score=sc.grade(p['case'],r['output_text'],natural_EOS=r['natural_EOS'],cap_hit=r['cap_hit']);assert score==r['score']
    itl=sorted((b-a)/1e9 for a,b in zip(visible,visible[1:]));pct=lambda f:itl[math.ceil(f*len(itl))-1]
    root_rows.append(dict(index=i,ordinal=r['ordinal'],suite=r['suite'],arm=r['arm'],pair=r['pair'],warmup=r['warmup'],owner_cores=r['owner_cores'],prompt_tokens=len(prompt),prefill_fragments=[f['tokens']for f in fr],generated_including_EOS=len(ids),visible_tokens=len(visible),natural_EOS=True,cap_hit=False,correct=score['correct'],parse_failure=score['parse_failure'],backend_decode_tok_s=rate,backend_TTFT_s=r['backend_sampled_TTFT_s'],TTFT_from_request_enter_s=visible[0]/1e9-begin,backend_ITL_P95_s=pct(.95),backend_ITL_P99_s=pct(.99),whole_backend_s=r['whole_backend_s'],prefill_CAPI_sum_s=sum(r['prefill_CAPI_s']),decode_CAPI_sum_s=sum(r['decode_CAPI_s']),pipeline_prefill={k:v for k,v in r['pipeline_prefill'].items()if k!='selected'},pipeline_decode={k:v for k,v in r['pipeline_decode'].items()if k!='selected'},output_IDs_SHA256=r['generated_token_ids_SHA256']))
   hashes.append(r['generated_token_ids_SHA256'])
  assert q['Native_forward_completed']==forward_count and q['control_calls']==controls
  output_hashes.append(hashes)
  rank_rows.append(dict(rank=rank,node=node,PCI=q['physical_PCI'],loading_seconds=q['loading_seconds'],elapsed_seconds=q['elapsed_s'],forward_returns=forward_count,control_calls=controls,resets=20,driver_min_sampled=dm['min_observed'],driver_max_gap_s=dm['max_gap_s'],driver_samples=dm['samples'],normal_destroy=True))
 assert len(keyhashes)==len(uidhashes)==1 and all(h==output_hashes[0]for h in output_hashes)and len({(r['node'],r['PCI'])for r in rank_rows})==16
 for i,v in enumerate(workers):
  assert v['passed']and v['completed']and v['actual_GPU']=='910371.900'and v['gpu_epoch']==w['epoch']and not v.get('forced_TERM')and not v.get('forced_KILL')
  assert [r['phase']for r in v['rounds']]==trial.phases()and [r['digest']for r in v['rounds']]==[r['digest']for r in workers[0]['rounds']]
  assert any(e['PID']==v['PID']and e['startticks']==v['startticks']for e in expected[NODES[i]])
 for p in w['post']['nodes']:
  assert p['actual_step']=='910371.901'and p['KFD_available']and not p['kfd_pids']and not p['matching_Model_processes']and not p['expected_owned_PIDs_still_live']and not p['expected_UIDsocket_listeners_still_live']and all(p['ports_bindable'].values())
 assert {v['node']for v in w['post']['nodes']}==set(NODES)
 assert all(set(v['physical_PCI'])=={r['PCI']for r in rank_rows if r['node']==v['node']}for v in w['post']['nodes'])
 pairs=[]
 for pair in range(3):
  a=next(r for r in root_rows if r['pair']==pair and r['arm']=='A');b=next(r for r in root_rows if r['pair']==pair and r['arm']=='B')
  assert a['ordinal']==b['ordinal']and not a['warmup']and not b['warmup']
  pairs.append(dict(pair=pair,order='AB'if a['index']<b['index']else'BA',ordinal=a['ordinal'],A_decode_tok_s=a['backend_decode_tok_s'],B_decode_tok_s=b['backend_decode_tok_s'],B_over_A_decode=b['backend_decode_tok_s']/a['backend_decode_tok_s'],B_over_A_TTFT=b['backend_TTFT_s']/a['backend_TTFT_s'],same_output_ID_hash=a['output_IDs_SHA256']==b['output_IDs_SHA256']))
 measured=[r for r in root_rows if not r['warmup']]
 layers_host=sum(r['pipeline_decode']['sum_s'][1]for r in measured);layers_stream=sum(r['pipeline_decode']['sum_s'][4]for r in measured);capi=sum(r['decode_CAPI_sum_s']for r in measured)
 return dict(schema='GLM900-original-resident-paired-independent-v1',GPU='910371.900',GPU_passed=True,GPU_elapsed_s=int(rows['910371.900'][4]),keeper='910407.1111',keeper_passed=True,keeper_elapsed_s=int(rows['910407.1111'][4]),post='910371.901',post_clean=True,original_index=index,accounting_original_SHA256=H(accounting),private_plan_parsed_SHA256=H(private_plan),raw_private_plan_receipt_SHA256=H(raw_plan_receipt),scorer_SHA256=H(source/'scoring.py'),trial_SHA256=H(source/'resident_trial.py'),source_manifest_SHA256=MF,GPU_manifest_SHA256=GMF,Native_SHA256=NATIVE,planned=10,completed_requests=10,failed_requests=0,rejected=0,canceled=0,missing=0,natural_EOS=10,correct=sum(r['correct']for r in root_rows),warmup_requests=4,measured_requests=6,paired_units=3,unique_questions=2,normal_destroy_owners=16,parameters_per_owner=117060,all16_output_ID_hashes_match=True,full_prompt_coverage_checked=True,owner_thread_affinity_verified=True,allocation_CPUs_per_node=64,driver_min_sampled=min(r['driver_min_sampled']for r in rank_rows),observed_floor_margin_min=min(r['driver_min_sampled']for r in rank_rows)-FLOOR,driver_max_gap_s=max(r['driver_max_gap_s']for r in rank_rows),rank_reports=rank_rows,requests=root_rows,pairs=pairs,median_B_over_A_decode=statistics.median(p['B_over_A_decode']for p in pairs),measured_layers_host_sum_s=layers_host,measured_layers_stream_sum_s=layers_stream,measured_decode_CAPI_sum_s=capi,layers_host_over_CAPI_enclosure=layers_host/capi,layers_stream_over_CAPI_enclosure=layers_stream/capi,profile_qualification='Recomputed summaries from original producer six-scalar buckets plus selected sample/count/bounds validation; not independent GPU trace or exclusive busy/FLOPs. Host and stream windows overlap and include host enqueue idle; never add them.',backend_rate_over15=sum(r['backend_decode_tok_s']>15 for r in root_rows),client_SLO_qualified=False,wide_quality_qualified=False,formal_NI_proven=False,independent_rawlogit_or_SDK_replay=False,performance_win_proven=False,promote=False,decision='Retain4core default. Do not promote2core arm or broaden ownercore grid; three pairs are order-sensitive and long-run rate drift is material. Next isolate actual layers/host-enqueue/kernel costs before another resident comparison.',limits='10 complete requests but only2unique questions,4warm+6measure. Reused frozen scorer/Native assertions, not independent semantic/logit/SDK replay. CPU mask affects only calling owner thread, not full process or64CPU allocation. Driver sampled floor and actual maximum gap are observations, not hard peak or future fit guarantees.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('original_directory');p.add_argument('accounting_original');p.add_argument('private_plan');p.add_argument('raw_plan_receipt');p.add_argument('output');a=p.parse_args();out=Path(a.output);assert not out.exists();r=read(a.original_directory,a.accounting_original,a.private_plan,a.raw_plan_receipt);r['reader_SHA256']=H(__file__);out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:r[k]for k in ['GPU','GPU_passed','keeper_passed','post_clean','completed_requests','correct','normal_destroy_owners','driver_min_sampled','observed_floor_margin_min','backend_rate_over15','median_B_over_A_decode','layers_stream_over_CAPI_enclosure']}))
