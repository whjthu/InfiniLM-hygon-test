"""Read sealed r447 original bytes. Capture completion is not Model quality."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math
H=lambda raw:hashlib.sha256(raw).hexdigest()
BASE=Path(__file__).resolve().parents[2]/'Deep-T1-MoE-activation-quant-r438/model-diagnostic-v2/analysis-device-1378/analyze_archive.py'
assert H(BASE.read_bytes())=='51d39885c9098450c8fbbb31fea67f1c91115753642dd8266220a810d661d2f2'
spec=importlib.util.spec_from_file_location('preserved1378_stdlib_raw_helpers',BASE)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
POINTS=['L2exit','L3entry','L3attn_norm','L3attn_value','L3attn_residual','L3ffn_collapsed','L3ffn_norm']

def analyze(files,manifest):
 for prefix,expected in [('source/boundary','d384669c5a1ae29d8b0911b8766868a6564376a7aad0150b2f00df938f4423c6'),('source/entry','ef23893b335815aa66ec8582b29c0b79a2a3bf0a2f05f4c2196252f1a2763311'),('source/controller','aab302e4f43e7854bcae6311814c8cc502c37c195b3bb22c454671cd12f61fc9')]:
  assert H(files[prefix+'/SOURCE_MANIFEST.json'])==expected
  source=json.loads(files[prefix+'/SOURCE_MANIFEST.json'])
  for name,h in source['files_SHA256'].items():assert H(files[prefix+'/'+name])==h
 for name,expected in [('CPU921','a731cc2cd374ffe10a238c9252b41194a8bc37d8ca7faf84500ea9d73985c692'),('CPU925','973bb6322c64c8fcd0109ef2a49757dbf1584c493245e1f4d0c7069a8da447b5'),('Context','f510393b69f198c035048f600637317419a6ef9f06947b58307e8c9b369030f0')]:assert H(files['proof/'+name+'.json'])==expected
 inputs=json.loads(files['run/private-input-IDs.json']);prompt,teacher=inputs['prompt_token_ids'],inputs['teacher_token_ids']
 assert len(prompt)==8489 and len(teacher)==83 and old.IDS_H(prompt)=='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'and old.IDS_H(teacher)=='596df3e1131e55184edc034f57f8447e025b91a38c7d6a20a69eaa8b61f39513'
 assert all(type(v)is int and 0<=v<129280 for v in prompt+teacher)and 1 not in teacher
 q=json.loads(files['run/typed/report.json']);assert q['completed']and q['passed']and not q['cleanup_errors']
 assert q['planned_modes']==[0,0]and q['max_sequences']==2 and q['max_forwards']==184
 assert 0<q['completed_forwards']<=q['actual_forward_attempts']<=184 and 0<len(q['sequences'])<=2
 tensors={};resources={};sequences=[];seen_windows=set();payload=0;nonfinite=[];continuity=[]
 for seq in q['sequences']:
  p=seq['sequence'];assert p in [0,1]and seq['mode']==0 and seq['actual_reset_before']
  assert seq['completed_forwards']==seq['actual_forward_attempts']<=92
  assert [v['output_index']for v in seq['decisions']]==list(range(len(seq['decisions'])))
  decisions=[old.decision_check(v,teacher,prompt,files)for v in seq['decisions']]
  if seq['completed']:assert seq['completed_forwards']==92 and len(decisions)==84
  counters=seq['actual_owner_counters_after'];assert len(counters)==8
  for c in counters:
   assert c['entered_valid_MoE']==c['host_returned_not_GPU_completed']==43*seq['completed_forwards']
   assert c['T1']==43*max(0,seq['completed_forwards']-9)and c['last_mode']==0
   assert c['unwound']==c['armed_mode1_T1']==c['routed_hits']==c['shared_hits']==0
   for role in ['routed','shared']:assert c[role+'_calls']==c[role+'_fallback']
  windows=[]
  for capture in seq['captures']:
   cp,o,f=capture['tag'];assert cp==p and o in[82,83]and f==9+o and(p,o)not in seen_windows
   seen_windows.add((p,o));assert not capture['partial']and len(capture['owners'])==8 and {v['owner']for v in capture['owners']}==set(range(8))
   declared_nonfinite=[];window_owners=[]
   for owner_summary in capture['owners']:
    owner=owner_summary['owner'];folder='run/typed/pass%d-output%d-boundaries/'%(p,o)
    meta_raw=files[folder+'owner%d.json'%owner];assert H(meta_raw)==owner_summary['metadata_SHA256'];meta=json.loads(meta_raw)
    assert meta['owner']==owner and meta['tag']==[p,o,f]and not meta['partial']and meta['CPU_bytes_only']and meta['actual_whole_Native_forward_completed_before_read']and meta['whole_engine_quiescence_before_read']
    assert len(meta['records'])==owner_summary['records']==7
    raw=files[folder+meta['binary']];assert len(raw)==meta['bytes']==owner_summary['raw_bytes']==131072 and H(raw)==meta['SHA256']==owner_summary['SHA256']
    payload+=len(raw)+len(meta_raw);seen=set();cursor=0;identity=set()
    for record in meta['records']:
     header=record['header'];assert len(header)==15
     rp,ro,rf,actual_owner,layer,point,code,ndim,d0,d1,d2,n,runtime,stream,thread=header
     assert(rp,ro,rf,actual_owner)==(p,o,f,owner)and point in range(7)and point not in seen and layer==(2 if point==0 else 3)
     seen.add(point);shape=[d0,d1,d2]if ndim==3 else[d0,d1];residual=point in[0,1,4]
     assert shape==([1,4,4096]if residual else[1,4096])and n==(32768 if residual else 8192)and code==1 and runtime and thread
     assert record['name']==POINTS[point]and record['shape']==shape and record['dtype']=='BF16'and record['offset']==cursor and record['bytes']==n
     chunk=raw[cursor:cursor+n];cursor+=n;assert len(chunk)==n and H(chunk)==record['SHA256'];identity.add((runtime,stream,thread))
     finite=all(math.isfinite(v)for v in old.values(chunk,'BF16'));assert finite==record['finite']
     if not finite:declared_nonfinite.append(dict(owner=owner,point=POINTS[point]));nonfinite.append(dict(pass_index=p,output_index=o,owner=owner,point=POINTS[point]))
     tensors[(p,o,owner,point)]=chunk
    assert len(seen)==7 and cursor==len(raw)and len(identity)==1;resources[(p,o,owner)]=next(iter(identity))
    same=tensors[(p,o,owner,0)]==tensors[(p,o,owner,1)]
    assert same==meta['L2exit_equals_L3entry']==owner_summary['L2exit_equals_L3entry']
    continuity.append(dict(pass_index=p,output_index=o,owner=owner,bytes_equal=same))
    window_owners.append(dict(owner=owner,L2exit_equals_L3entry=same,raw_bytes=len(raw)))
   assert declared_nonfinite==capture['nonfinite']
   windows.append(dict(output_index=o,forward_1based=f,past=8488+o,owners=window_owners))
  mismatches=[dict(output_index=v['output_index'],Native_chosen=v['Native_chosen'],teacher=teacher[v['output_index']],observable_BF16_margin=v['observable_BF16_margin'])for v in decisions if v['output_index']<83 and v['Native_chosen']!=teacher[v['output_index']]]
  sequences.append(dict(pass_index=p,mode=0,completed=seq['completed'],forwards=seq['completed_forwards'],decisions=decisions,first_teacher_mismatch=mismatches[0]if mismatches else None,teacher_mismatch_count=len(mismatches),teacher_mismatches=mismatches,windows=windows))
 if not q['early_stop']:assert len(sequences)==2 and q['completed_forwards']==q['actual_forward_attempts']==184 and len(seen_windows)==4
 for owner in range(8):assert len({v for key,v in resources.items()if key[2]==owner})==1
 cross=[]
 for o in [82,83]:
  if(0,o)not in seen_windows or(1,o)not in seen_windows:continue
  points=[]
  for point,name in enumerate(POINTS):
   rows=[]
   for owner in range(8):
    a,b=tensors[(0,o,owner,point)],tensors[(1,o,owner,point)];metrics=old.compare(a,b,'BF16')
    rows.append(dict(owner=owner,SHA256=[H(a),H(b)],**metrics))
   points.append(dict(point=point,name=name,different_owners=[v['owner']for v in rows if not v['bytes_equal']],owners=rows))
  first=next((v for v in points if v['different_owners']),None)
  cross.append(dict(output_index=o,first_different_boundary=None if first is None else first['name'],owners_at_first=None if first is None else first['different_owners'],points=points,ordering='Semantic source order only; no cross-owner clock ordering'))
 choice=[];logit=[]
 if len(sequences)==2:
  a,b=sequences[0]['decisions'],sequences[1]['decisions']
  for x,y in zip(a,b):
   if x['Native_chosen']!=y['Native_chosen']:choice.append(dict(output_index=x['output_index'],Native_chosen=[x['Native_chosen'],y['Native_chosen']],BF16_margin=[x['observable_BF16_margin'],y['observable_BF16_margin']]))
   if x['logits_SHA256']!=y['logits_SHA256']:logit.append(x['output_index'])
 raw_logits=sum(len(v)for n,v in files.items()if n.startswith('run/typed/pass')and n.endswith('-logits-BF16.bin'));budget=payload+raw_logits+2097152
 assert budget==q['raw_budget_used']<=33554432
 assert sum(len(files[n])for n in ['run/typed/report.json','run/private-input-IDs.json'])<=2097152
 watch=json.loads(files['control/watch.json']);entry=json.loads(files['run/entry-report.json'])
 assert watch['completed']and watch['passed']and not watch['forced_cleanup']and entry['completed']and entry['passed']
 assert entry['normal_Native_shutdown']and entry['engine_worker_shutdown_completed']and entry['observed_all8_threads_exited']
 assert watch['entry_report_SHA256']==H(files['run/entry-report.json'])and entry['diagnostic']['SHA256']==H(files['run/typed/report.json'])
 assert watch['publication_commit']=='3d8e405eeb2faaf4c132cc047e710bf1f5f08bb2'and watch['source_manifest_SHA256']=='aab302e4f43e7854bcae6311814c8cc502c37c195b3bb22c454671cd12f61fc9'
 assert entry['Native_SHA256']=='8b1d1dacee6ba4b4f4d08dd38aa46c7bc529e5d3d89dc1ab753c204fe1b8084b'and entry['Native_derivation_verified']and entry['loaded_key_count']==67569
 assert entry['TP']==8 and entry['PP']==1 and entry['HC_mode']==0 and entry['diagnostic_only']and entry['HTTP']==entry['free_generation']==0
 provider='/refactor/candidates/Deep-layer2-3-boundary-diagnostic-r447-CPU-v2/libDeep_layer_boundary_r447.so'
 assert set(entry['actual_providers_before_weights'])=={'create','load','finish','forward','reset','destroy'}and set(entry['actual_providers_before_weights'].values())=={provider}
 assert len(entry['actual_diagnostic_providers']['providers'])==11 and set(entry['actual_diagnostic_providers']['providers'].values())=={provider}and entry['actual_diagnostic_providers']['boundary_new_CABI']==7
 assert watch['actual_GPU']==entry['actual_step']=='910374.1384'and watch['CPU_keeper']=='910407.927'and watch['original_GPU_start_unix']==entry['gpu_start_epoch']==1793739562.3785217
 post=watch['post'];assert post['node']=='a02r4n01'and not post['kfd_pids']and not post['matching_component_processes']and post['ports_bindable']=={'43882':True}and len(set(post['physical_PCI']))==8
 accounts=json.loads(files['control/accounting.json']);assert all(any(r[0]==step and r[2:4]==['COMPLETED','0:0']for r in accounts)for step in ['910374.1384','910407.927',post['actual_step']])
 return dict(audit_passed=True,scope='Original1384 two-OFF fixed-teacher seven-boundary diagnostic only',Model_quality_or_performance_qualified=False,original1374_quality_FAIL_unchanged=True,root_cause='UNKNOWN',actual_GPU=watch['actual_GPU'],CPU_keeper=watch['CPU_keeper'],post=post['actual_step'],modes=[0,0],forwards=q['completed_forwards'],forward_attempts=q['actual_forward_attempts'],early_stop=q['early_stop'],passes=sequences,windows=len(seen_windows),real_boundary_records=56*len(seen_windows),cross_OFF_comparisons=cross,first_chosen_difference=choice[0]if choice else None,chosen_difference_rows=choice,producer_logit_hash_differences=len(logit),first_producer_logit_hash_difference=logit[0]if logit else None,nonfinite_boundaries=nonfinite,L2exit_L3entry_continuity_all=all(v['bytes_equal']for v in continuity),boundary_raw_bytes=131072*8*len(seen_windows),boundary_bytes_including_owner_metadata=payload,owner_metadata_bytes=payload-131072*8*len(seen_windows),raw_logits_bytes=raw_logits,budget_including_2MiB_metadata_reserve=budget,limit_bytes=33554432,normal8owner_and_post_clean=True,limitation='Only output82/83 boundaries captured; producer hashes/top8 at earlier decisions do not expose earlier activations. Returned BF16 bit-expanded scores/margins are not precast F32 decisions. Capture adds seven completion-returning D2H copies per owner/window and alters scheduling/pool; equality or divergence is not a repair/quality/performance claim.',verified_original_files=len(files))

def main():
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output');a=p.parse_args();files,m,digest=old.load(a.archive);q=analyze(files,m);q['archive_SHA256']=digest;raw=json.dumps(q,indent=2)+'\n'
 if a.output:Path(a.output).write_text(raw,encoding='utf-8')
 print(raw,end='')
if __name__=='__main__':main()
