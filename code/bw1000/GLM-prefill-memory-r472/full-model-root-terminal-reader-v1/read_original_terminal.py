"""Consume original895 once, retaining keeper1108 failure independently of owner success.
Stdlib only: no Native, SDK, compiler, Slurm, HTTP, tokenizer or Model execution.
Original raw logit/SDK numerical replay and full benchmark qualification are absent.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,statistics
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
JH=lambda x:hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
LABEL='GLM-global-TP16-prefill512-full-Model-entry-r472-v1'
MF='c06a5a8f7447b5725c54ce558b90471d37e44e9daccc905048abdcd32c16cd1a'
GMF='1d948a16fb7bf03b6cb06d734dc9e0eef8ca9505fa40697b4d836f6dc8b92565'
NATIVE='af4f13006dd123b62d56426d82830cae792c27a60127897b84fcf9c14c263983'
FLOOR=8589934592
NODES=['a02r3n15','a02r3n16']

def read(root,accounting):
 root=Path(root);index=json.loads((root/'original-index.json').read_bytes());assert len(index)==19
 for v in index:assert H(root/v['name'])==v['SHA256'] and (root/v['name']).stat().st_size==v['bytes']
 w=json.loads((root/(LABEL+'-watch.json')).read_bytes());assert w['completed'] and not w['passed'] and w['GPU']=='910371.895' and w['CPU_keeper']=='910407.1108' and w['srun_rc']==0 and w['planned']==2
 assert w['terminal_accounting_observations']==[w['GPU_accounting']] and w['GPU_accounting'][2:4]==['RUNNING','0:0']
 assert 'line 48' in w['traceback'] and "row[2:4]==['COMPLETED','0:0']" in w['traceback'] and not w.get('forced_cleanup')
 x=json.loads(Path(accounting).read_bytes());assert x['rc']==0;a=json.loads(x['out']);rows={r.split('|')[0]:r.split('|')for r in a['accounting']}
 assert rows['910371.895'][2:4]==['COMPLETED','0:0'] and rows['910407.1108'][2:4]==['FAILED','1:0'] and rows['910371.897'][2:4]==['COMPLETED','0:0']
 source=Path(__file__).resolve().parents[1]/'full-model-entry-v1'
 inp=json.loads((source/'INPUT_BINDING.json').read_bytes());plan=json.loads((root/'private-plan-parsed.json').read_bytes());assert plan['planned']==2 and len(plan['requests'])==2
 # Binding raw SHA is retained in original input receipt; reconstructed parsed file is a separate private copy.
 assert inp['private_plan']['SHA256']=='a9bde5228f9f753c37d9c54d19f39d72233260378459d2bb3219536dd47099d7'
 raw_plan=json.loads((root.parent/'GLM472-original-private-plan-first.json').read_bytes())['out'].encode()
 candidates=[raw_plan,raw_plan[:-1]];exact=[b for b in candidates if len(b)==inp['private_plan']['bytes']and hashlib.sha256(b).hexdigest()==inp['private_plan']['SHA256']];assert len(exact)==1 and json.loads(exact[0])==plan
 spec=importlib.util.spec_from_file_location('original_frozen_scoring',source/'scoring.py');sc=importlib.util.module_from_spec(spec);spec.loader.exec_module(sc)
 expected=json.loads(next(v.split('=',1)[1]for v in w['failure_post']['raw']['argv']if v.startswith('GLM470_EXPECTED_PIDS=')))
 assert set(expected)==set(NODES) and w['failure_post']['raw']['rc']==0
 workers=[json.loads((root/('worker-%d.json'%i)).read_bytes())for i in range(2)]
 owners=[json.loads((root/('owner-%02d.json'%i)).read_bytes())for i in range(16)]
 result=[];keyhashes=set();uidhashes=set();outputhashes=[]
 for rank,q in enumerate(owners):
  node=NODES[rank//8];geo=q['constructor_geometry'];post=q['postload_geometry']
  assert q['global_rank']==rank and q['node']==node and q['device']==rank%8 and q['actual_GPU']=='910371.895' and q['gpu_epoch']==w['epoch']
  assert q['owner_TID']==q['PID'] and q['UID']==154459 and q['completed'] and q['passed'] and q['normal_close'] and q['worker_shutdown_completed'] and q['Runtime_weak_expired_CABI_assertion']
  assert q['weights'] and not q['Model_READY'] and not q.get('error') and not q['Model_qualified']
  assert q['all78layers_loaded'] and q['loaded_parameters']==q['checkpoint_key_count']==post['loaded_parameters']==117060 and q['actual_layers']==list(range(78)) and post['Model_READY']
  assert q['private_input_SHA256']==inp['private_plan']['SHA256']
  assert q['source_manifest_SHA256']==MF and q['GPU_manifest_SHA256']==GMF and q['Native_SHA256']==NATIVE and all(v['SHA256']==NATIVE for v in q['CAPI_bindings'].values())
  assert geo['TP']==geo['world_size']==16 and geo['PP']==1 and geo['world_rank']==geo['tp_rank']==rank and geo['rows']==1 and geo['capacity_per_row']==32768 and geo['local_attention_heads']==4 and geo['local_index_heads']==2 and geo['layer_begin']==0 and geo['layer_end']==78
  assert geo['Model_constructed'] and geo['cache_allocated'] and not geo['weights_loaded'] and geo['replicated_embedding_BF16_bytes']==0 and geo['embedding_BF16_local_bytes']==118947840 and geo['embedding_vocab_begin']==rank*9680 and geo['embedding_vocab_end']==(rank+1)*9680 and geo['cache_logical_bytes']['sum']==8355053568
  assert q['resets']==4 and q['Native_forward_attempts']==q['Native_forward_completed']==389 and q['control_calls']==386 and len(q['requests'])==2
  assert all(v['floor']==FLOOR and v['free']>=FLOOR and v['total']==68702699520 for v in q['memory_boundaries'])
  dm=q['driver_memory'];assert dm['min_observed']>=FLOOR and not dm['errors'] and dm['samples']>0
  assert dm['min_observed']==min(v['free'] for v in dm['lowpoints'])==min(v['min_free']for v in dm['byphase'].values())
  owner=next(v for v in expected[node]if v['PID']==q['PID']);assert owner['startticks']==q['startticks'] and set(owner['socket_inodes'])==set(q['owned_socket_inodes']) and len(expected[node])==9
  keyhashes.add(q['checkpoint_keys_SHA256']);uidhashes.add(tuple(q['UID_SHA256']));hashes=[]
  for case_index,(r,p)in enumerate(zip(q['requests'],plan['requests'])):
   frozen=inp['private_plan']['requests'][case_index];ids=r['generated_token_ids'];ticks=r['sampled_ns'];prompt=p['prompt_token_ids'];fr=r['prefill_fragment_plan']
   assert r['ordinal']==p['ordinal']==frozen['ordinal'] and r['case_id']==p['case_id']==frozen['case_id'] and r['suite']==p['suite']==frozen['suite']
   assert r['prompt_tokens']==len(prompt)==frozen['prompt_tokens'] and r['prompt_IDs_SHA256']==JH(prompt)==frozen['prompt_IDs_SHA256'] and r['body_SHA256']==frozen['body_SHA256']
   assert r['planned_max_new_tokens']==4096 and r['runtime_prefill_chunk']==512 and r['original_request_prefill_window']==4096 and r['reset_before'] and r['reset_after'] and r['completed']
   offset=0
   for f in fr:
    chunk=prompt[offset:offset+512];assert f==dict(offset=offset,tokens=len(chunk),IDs_SHA256=JH(chunk));offset+=len(chunk)
   assert offset==len(prompt) and len(fr)==len(r['prefill_CAPI_s']) and all(math.isfinite(v) and v>=0 for v in r['prefill_CAPI_s']+r['decode_CAPI_s'])
   assert 0<len(ids)<=4096 and len(ids)==len(ticks) and all(type(v)is int and 0<=v<154880 for v in ids) and all(b>a for a,b in zip(ticks,ticks[1:]))
   assert r['generated_token_ids_SHA256']==JH(ids) and ids[-1]in inp['eos_token_ids'] and r['natural_EOS'] and not r['cap_hit']
   assert r['Native_forward_end']-r['Native_forward_begin']==len(fr)+len(ids)-1 and len(r['decode_CAPI_s'])==len(ids)-1 and r['past_after']==len(prompt)+len(ids)-1
   visible=[t for token,t in zip(ids,ticks)if token not in set(inp['eos_token_ids']+inp['tokenizer']['special_ids'])]
   rate=(len(visible)-1)*1e9/(visible[-1]-visible[0]);assert len(visible)==r['visible_nonEOS_nonspecial'] and math.isclose(rate,r['backend_decode_tok_s'],rel_tol=1e-12)
   begin=next(e['monotonic']for e in q['events']if e['phase']=='request.enter'and e['ordinal']==case_index)
   inferred_begin=visible[0]/1e9-r['backend_sampled_TTFT_s'];first_forward=next(e['monotonic']for e in q['events']if e['phase']=='forward.enter'and e['monotonic']>=begin)
   assert begin<=inferred_begin<=first_forward<=visible[0]/1e9
   if rank==0:
    score=sc.grade(p['case'],r['output_text'],natural_EOS=r['natural_EOS'],cap_hit=r['cap_hit']);assert score==r['score']
   hashes.append(r['generated_token_ids_SHA256'])
  outputhashes.append(hashes)
  result.append(dict(rank=rank,node=node,PCI=q['physical_PCI'],loading_seconds=q['loading_seconds'],elapsed_seconds=q['elapsed_s'],loaded_parameters=q['loaded_parameters'],Native_full78_forward_returns=389,normal_destroy=True,driver_min_sampled=dm['min_observed'],driver_max_gap_s=dm['max_gap_s'],memory_boundary_min=min(v['free']for v in q['memory_boundaries']),driver_samples=dm['samples']))
 assert len(keyhashes)==len(uidhashes)==1 and all(h==outputhashes[0]for h in outputhashes) and len({(r['node'],r['PCI'])for r in result})==16
 for i,v in enumerate(workers):
  assert v['passed'] and v['completed'] and v['actual_GPU']=='910371.895' and not v.get('forced_TERM') and not v.get('forced_KILL')
  assert [r['phase']for r in v['rounds']]==['created','loaded','case0','case1','quiescent','closed'] and [r['digest']for r in v['rounds']]==[r['digest']for r in workers[0]['rounds']]
  assert any(e['PID']==v['PID']and e['startticks']==v['startticks']for e in expected[NODES[i]])
 for p in w['failure_post']['nodes']:
  assert p['actual_step']=='910371.897' and p['KFD_available'] and not p['kfd_pids'] and not p['matching_Model_processes'] and not p['expected_owned_PIDs_still_live'] and not p['expected_UIDsocket_listeners_still_live'] and all(p['ports_bindable'].values())
 assert {v['node']for v in w['failure_post']['nodes']}==set(NODES)
 assert all(set(v['physical_PCI'])=={r['PCI']for r in result if r['node']==v['node']}for v in w['failure_post']['nodes'])
 roots=[]
 for case_index,r in enumerate(owners[0]['requests']):
  request_event=next(e['monotonic']for e in owners[0]['events']if e['phase']=='request.enter'and e['ordinal']==case_index)
  visible=[t for token,t in zip(r['generated_token_ids'],r['sampled_ns'])if token not in set(inp['eos_token_ids']+inp['tokenizer']['special_ids'])];itl=sorted((b-a)/1e9 for a,b in zip(visible,visible[1:]));percentile=lambda p:itl[math.ceil(p*len(itl))-1]
  roots.append(dict(ordinal=r['ordinal'],suite=r['suite'],prompt_tokens=r['prompt_tokens'],prefill_fragment_counts=[v['tokens']for v in r['prefill_fragment_plan']],generated_including_EOS=len(r['generated_token_ids']),visible_tokens=len(visible),natural_EOS=True,cap_hit=False,correct=r['score']['correct'],parse_failure=r['score']['parse_failure'],backend_decode_tok_s=r['backend_decode_tok_s'],backend_TTFT_s=r['backend_sampled_TTFT_s'],TTFT_from_request_enter_event_s=visible[0]/1e9-request_event,backend_ITL_P95_s=percentile(.95),backend_ITL_P99_s=percentile(.99),whole_backend_s=r['whole_backend_s'],prefill_CAPI_sum_s=sum(r['prefill_CAPI_s']),decode_CAPI_sum_s=sum(r['decode_CAPI_s'])))
 return dict(schema='GLM895-original-terminal-independent-v1',GPU='910371.895',GPU_passed=True,GPU_elapsed_s=int(rows['910371.895'][4]),keeper='910407.1108',keeper_passed=False,keeper_elapsed_s=int(rows['910407.1108'][4]),producer_accounting=w['GPU_accounting'],keeper_failure='Controller breaks on any accounting row, including RUNNING after srun0; originalfailure retained, laterGPUCOMPLETED does not washkeeperFAIL.',post='910371.897',post_clean=True,original_index=index,accounting_original_SHA256=H(accounting),private_parsed_plan_SHA256=H(root/'private-plan-parsed.json'),scorer_SHA256=H(source/'scoring.py'),planned=2,completed_requests=2,failed_requests=0,rejected=0,canceled=0,missing=0,natural_EOS=2,correct=2,normal_destroy_owners=16,parameters_per_owner=117060,full78_forward_returns_per_owner=389,all16_output_ID_hashes_match=True,full_prompt_coverage_checked=True,weights_loaded_owners=16,owner_full_model_entry_checks_passed=True,floor_bytes=FLOOR,driver_min_sampled=min(r['driver_min_sampled']for r in result),observed_floor_margin_min=min(r['driver_min_sampled']for r in result)-FLOOR,driver_max_gap_s=max(r['driver_max_gap_s']for r in result),rank_reports=result,requests=roots,backend_rate_over15=0,client_SLO_qualified=False,wide_quality_qualified=False,formal_NI_proven=False,independent_rawlogit_or_SDK_replay=False,qualified=False,promote=False,limits='Two frozen prompts only, no paired throughput baseline or clientHTTP; driver5ms target missed subinterval peaks (actualgap recorded), observedfloor not hardallocatorupperbound; request.enter includes actual local/remote report-save before producer TTFT start; conservative event-to-firstsampleTTFT also shown; same frozen scorer reused on originaltext, no independent semantic evaluator. KeeperFAIL blocks automatic modelpromotion.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('original_directory');p.add_argument('accounting_original');p.add_argument('output');a=p.parse_args();out=Path(a.output);assert not out.exists();r=read(a.original_directory,a.accounting_original);r['reader_SHA256']=H(__file__);out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in ['GPU','GPU_passed','keeper_passed','post_clean','completed_requests','correct','normal_destroy_owners','driver_min_sampled','observed_floor_margin_min','backend_rate_over15']}))
