"""Stdlib raw capture audit. Diagnostic samples are not quality/performance gates."""
import argparse,hashlib,json,math,struct,tarfile
from pathlib import Path,PurePosixPath
H=lambda b:hashlib.sha256(b).hexdigest()
IDS_H=lambda ids:H(json.dumps(ids,separators=(',',':')).encode())
FIELDS={0:['hidden','route_ids','route_weights','original_GU','snapshot_GU','reference_activation','reference_I8','reference_scale','candidate_activation','candidate_I8','candidate_scale'],1:['original_GU','snapshot_GU','reference_activation','reference_I8','reference_scale','candidate_activation','candidate_I8','candidate_scale']}
TYPES={'BF16':(1,2),'I8':(2,1),'F32':(3,4),'I64':(4,8)}
MODES=[0,1,0,1]

def load(path):
 raw=Path(path).read_bytes();files={}
 with tarfile.open(fileobj=__import__('io').BytesIO(raw),mode='r:gz')as tar:
  for item in tar:
   name=PurePosixPath(item.name);assert item.isfile()and not name.is_absolute()and '..'not in name.parts and item.name not in files
   assert item.size<=134217728;files[item.name]=tar.extractfile(item).read();assert len(files[item.name])==item.size
 manifest=json.loads(files.pop('manifest.json'));assert len(manifest['files'])==len(files)and len({v['name']for v in manifest['files']})==len(files)
 for item in manifest['files']:assert item['name']in files and len(files[item['name']])==item['bytes']and H(files[item['name']])==item['SHA256']
 return files,manifest,H(raw)

def values(raw,dtype):
 if dtype=='BF16':return [struct.unpack('<f',struct.pack('<I',v[0]<<16))[0]for v in struct.iter_unpack('<H',raw)]
 return [v[0]for v in struct.iter_unpack({'F32':'<f','I64':'<q','I8':'<b'}[dtype],raw)]

def compare(a,b,dtype):
 assert len(a)==len(b);x,y=values(a,dtype),values(b,dtype);different=sum(v!=w for v,w in zip(x,y));finite=all(math.isfinite(v)for v in x+y)
 out=dict(bytes_equal=a==b,elements=len(x),different_values=different,finite=finite)
 if finite:
  out['max_abs_error']=max((abs(v-w)for v,w in zip(x,y)),default=0)
  ref=math.fsum(v*v for v in x);err=math.fsum((v-w)*(v-w)for v,w in zip(x,y));out['NRMSE']=math.sqrt(err/ref)if ref>0 else 0. if err==0 else None
 return out

def layer_summary(differences,compared):
 if not differences:return dict(compared=compared,different=0,first_layer=None)
 first=min(v['layer']for v in differences)
 return dict(compared=compared,different=len(differences),first_layer=first,owners_roles_at_first_layer=[dict(owner=v['owner'],role=v['role'])for v in differences if v['layer']==first],first_eight=sorted(differences,key=lambda v:(v['layer'],v['role'],v['owner']))[:8],ordering='First semantic layer only; concurrent owners have no claimed wall-time ordering')

def decision_check(row,teacher,prompt,files):
 index=row['output_index'];assert 0<=index<=83 and row['forward_1based']==9+index
 part=prompt[-297:]if index==0 else[teacher[index-1]];assert row['input_past']==8192 if index==0 else row['input_past']==8489+index-1
 assert row['input_count']==len(part)and row['input_IDs_SHA256']==IDS_H(part)
 assert row['finite']and 0<=row['Native_chosen']<129280
 top=row['top8'];assert len(top)==8 and len({v['id']for v in top})==8 and all(type(v['id'])is int and 0<=v['id']<129280 and math.isfinite(v['score'])for v in top)
 assert all(a['score']>=b['score']for a,b in zip(top,top[1:]))
 assert row['observable_BF16_margin']==top[0]['score']-top[1]['score']and row['selected_scores']['Native_chosen']['id']==row['Native_chosen']and row['selected_scores']['Native_chosen']['score']==top[0]['score']
 assert row['selected_scores']['original_correct83']['id']==30722 and row['selected_scores']['original_incorrect83']['id']==9155
 if index<83:assert row['selected_scores']['teacher']['id']==teacher[index]
 summary=dict(output_index=index,Native_chosen=row['Native_chosen'],observable_BF16_margin=row['observable_BF16_margin'],maximum_tie_count=row['maximum_tie_count'],top8=top,selected_scores=row['selected_scores'],logits_SHA256=row['logits_SHA256'],full_raw_independently_checked=False)
 if 'raw_logits'in row:
  ref=row['raw_logits'];raw=files['run/typed/'+ref['file']];assert len(raw)==ref['bytes']==129280*2 and H(raw)==ref['SHA256']==row['logits_SHA256'];scores=values(raw,'BF16');assert all(math.isfinite(v)for v in scores)
  order=sorted(range(129280),key=lambda k:(-scores[k],k));assert top==[dict(id=k,score=scores[k])for k in order[:8]]
  assert scores[row['Native_chosen']]==scores[order[0]]and sum(v==scores[order[0]]for v in scores)==row['maximum_tie_count']
  assert all(v['score']==scores[v['id']]for v in row['selected_scores'].values());summary['full_raw_independently_checked']=True
 return summary

def analyze(files,manifest):
 inputs=json.loads(files['run/private-input-IDs.json']);prompt,teacher=inputs['prompt_token_ids'],inputs['teacher_token_ids'];ref=json.loads(files['source/TEACHER_REFERENCE.json'])
 assert len(prompt)==8489 and len(teacher)==83 and IDS_H(prompt)==ref['prompt_IDs_SHA256']=='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'and IDS_H(teacher)==ref['teacher_IDs_SHA256']=='596df3e1131e55184edc034f57f8447e025b91a38c7d6a20a69eaa8b61f39513'
 assert all(type(v)is int and 0<=v<129280 for v in prompt+teacher)and 1 not in teacher
 partial=not manifest['all_passes_or_terminal']
 report=json.loads(files['run/typed/pass0-report-snapshot.json']if partial else files['run/typed/report.json']);sequences=[report['sequence']]if partial else report['sequences']
 assert report['planned_modes']==MODES and report['teacher_IDs_SHA256']==IDS_H(teacher)and report['prompt_IDs_SHA256']==IDS_H(prompt)
 if partial:assert len(sequences)==1 and sequences[0]['sequence']==0 and sequences[0]['completed']
 else:
  assert report['completed']and report['passed']and not report['cleanup_errors'];assert 0<report['completed_forwards']<=report['actual_forward_attempts']<=368
  assert len({v['sequence']for v in sequences})==len(sequences)
  if not report['early_stop']:assert len(sequences)==4 and report['completed_forwards']==report['actual_forward_attempts']==368
 tensors={};snapshots={};resources={};summary=[];shadow=[];all_finite=True;seen_windows=[];physical_payload_bytes=0
 for seq in sequences:
  p=seq['sequence'];assert p in range(4)and seq['mode']==MODES[p]and seq['actual_reset_before'];assert seq['completed_forwards']==seq['actual_forward_attempts']<=92
  decisions=seq['decisions'];assert [v['output_index']for v in decisions]==list(range(len(decisions)))
  if seq['completed']:assert seq['completed_forwards']==92 and len(decisions)==84 and seq['last_total']==8572
  checked=[decision_check(v,teacher,prompt,files)for v in decisions]
  mismatches=[dict(output_index=v['output_index'],teacher=teacher[v['output_index']],Native_chosen=v['Native_chosen'],observable_BF16_margin=v['observable_BF16_margin'])for v in checked if v['output_index']<83 and v['Native_chosen']!=teacher[v['output_index']]]
  counters=seq['actual_owner_counters_after'];assert len(counters)==8
  for c in counters:
   t1=43*(seq['completed_forwards']-9);assert c['entered_valid_MoE']==c['host_returned_not_GPU_completed']==43*seq['completed_forwards']and c['T1']==t1 and c['armed_mode1_T1']==MODES[p]*t1 and c['last_mode']==MODES[p]and c['unwound']==0
   for role in ['routed','shared']:assert c[role+'_hits']==MODES[p]*t1 and c[role+'_calls']==c[role+'_hits']+c[role+'_fallback']
  window_summaries=[]
  for capture in seq['captures']:
   cp,o,f=capture['tag'];assert cp==p and o in[82,83]and f==9+o and (p,o)not in seen_windows;seen_windows.append((p,o));assert len(capture['owners'])==8 and {v['owner']for v in capture['owners']}==set(range(8))
   full=0;same_gu=True;different_roles=[];window_fields=0
   for owner_summary in capture['owners']:
    owner=owner_summary['owner'];folder='run/typed/pass%d-output%d-owners/'%(p,o);meta_raw=files[folder+'owner%d.json'%owner];assert H(meta_raw)==owner_summary['metadata_SHA256'];meta=json.loads(meta_raw)
    assert meta['owner']==owner and meta['tag']==[p,o,f]and meta['CPU_bytes_only']and meta['Native_forward_completed_before_parent_read']and len(meta['records'])==owner_summary['records']==86
    raw=files[folder+meta['binary']];assert len(raw)==meta['bytes']==owner_summary['raw_bytes']and H(raw)==meta['SHA256']==owner_summary['SHA256'];physical_payload_bytes+=len(raw)+len(meta_raw)
    seen=set();cursor=0;owner_resources=set()
    for record in meta['records']:
     header=record['header'];assert len(header)==12;rp,ro,rf,layer,rowner,role,nfields,nbytes,runtime,stream,thread,original_gu_ptr=header
     assert (rp,ro,rf,rowner)==(p,o,f,owner)and layer in range(43)and role in[0,1]and(layer,role)not in seen and nfields==len(FIELDS[role]);seen.add((layer,role));owner_resources.add((runtime,stream,thread));assert runtime and thread and original_gu_ptr
     assert set(record['tensors'])==set(FIELDS[role]);fields={};size=0
     for name in FIELDS[role]:
      v=record['tensors'][name];assert v['dtype']in TYPES and (v['dtype_code'],v['width'])==TYPES[v['dtype']];a,b=v['shape'];assert v['bytes']==a*b*v['width']and v['offset']==cursor
      rows=6 if role==0 else 1;expected=('BF16',[1,4096])if name=='hidden'else('I64',[1,6])if name=='route_ids'else('F32',[1,6])if name=='route_weights'else('BF16',[rows,512])if name.endswith('GU')else('BF16',[rows,256])if name.endswith('activation')else('I8',[rows,256])if name.endswith('I8')else('F32',[rows,1]);assert(v['dtype'],v['shape'])==expected
      chunk=raw[cursor:cursor+v['bytes']];cursor+=v['bytes'];size+=v['bytes'];assert len(chunk)==v['bytes']and H(chunk)==v['SHA256'];fields[name]=(chunk,v['dtype']);parsed=values(chunk,v['dtype']);finite=all(math.isfinite(x)for x in parsed);all_finite&=finite
      if name.endswith('scale'):assert finite and min(parsed)>0
      if name.endswith('I8'):assert all(-127<=x<=127 for x in parsed)
      if name=='route_ids':assert all(0<=x<256 for x in parsed)
      tensors[(p,o,owner,layer,role,name)]=chunk;window_fields+=1
     assert size==nbytes;same=fields['original_GU'][0]==fields['snapshot_GU'][0];same_gu&=same;assert record['same_actual_GU']==same
     metrics={name:compare(fields['reference_'+name][0],fields['candidate_'+name][0],fields['reference_'+name][1])for name in ['activation','I8','scale']}
     differing=[name for name,v in metrics.items()if not v['bytes_equal']];assert differing==record['shadow_differing_fields']
     if differing:different_roles.append(dict(owner=owner,layer=layer,role=role,fields=differing,metrics=metrics))
     full+=1
    assert len(seen)==86 and cursor==len(raw)and len(owner_resources)==1;resource=next(iter(owner_resources));resources[(p,o,owner)]=resource
   declared=capture['same_GU_shadow_differences'];assert {(v['owner'],v['layer'],v['role'],tuple(v['fields']))for v in declared}=={(v['owner'],v['layer'],v['role'],tuple(v['fields']))for v in different_roles}
   assert full==688;window_summaries.append(dict(output_index=o,forward_1based=f,owner_layer_role_records=full,fields_checked=window_fields,same_original_snapshot_GU_all=same_gu,shadow_differences=layer_summary(different_roles,688)))
   shadow.extend(dict(pass_index=p,output_index=o,**v)for v in different_roles)
  if seq['completed']:assert len(window_summaries)==2
  summary.append(dict(pass_index=p,mode=MODES[p],completed=seq['completed'],forward_attempts=seq['actual_forward_attempts'],completed_forwards=seq['completed_forwards'],decision_count=len(checked),teacher_mismatches_count=len(mismatches),first_teacher_mismatch=mismatches[0]if mismatches else None,teacher_first_mismatch_before_capture82=bool(mismatches and mismatches[0]['output_index']<82),teacher_mismatches=mismatches,windows=window_summaries,decisions=checked))
 # Stable Runtime/stream/owner-thread identity is checked only within the same
 # owner. Stream pointer zero may be a valid per-device default; no comparison
 # of absolute clocks or numeric addresses across different devices is used.
 for owner in range(8):assert len({v for key,v in resources.items()if key[2]==owner})==1
 cross=[];by_pass={v['pass_index']:v for v in summary}
 for a,b in [(0,2),(1,3),(0,1),(2,3),(1,2),(0,3)]:
  if a not in by_pass or b not in by_pass:continue
  decisions_a,decisions_b=by_pass[a]['decisions'],by_pass[b]['decisions'];n=min(len(decisions_a),len(decisions_b));choice_diff=[dict(output_index=i,IDs=[decisions_a[i]['Native_chosen'],decisions_b[i]['Native_chosen']])for i in range(n)if decisions_a[i]['Native_chosen']!=decisions_b[i]['Native_chosen']]
  logit_diff=[i for i in range(n)if decisions_a[i]['logits_SHA256']!=decisions_b[i]['logits_SHA256']]
  windows=[]
  for o in [82,83]:
   if(a,o)not in seen_windows or(b,o)not in seen_windows:continue
   field_summary={}
   for name in ['hidden','route_ids','route_weights','original_GU','snapshot_GU','reference_activation','reference_I8','reference_scale','candidate_activation','candidate_I8','candidate_scale']:
    differences=[];compared=0
    for owner in range(8):
     for layer in range(43):
      for role in range(2):
       x=(a,o,owner,layer,role,name);y=(b,o,owner,layer,role,name)
       if x not in tensors or y not in tensors:continue
       compared+=1
       if tensors[x]!=tensors[y]:differences.append(dict(owner=owner,layer=layer,role=role,SHA256=[H(tensors[x]),H(tensors[y])]))
    field_summary[name]=layer_summary(differences,compared)
   windows.append(dict(output_index=o,fields=field_summary))
  cross.append(dict(pass_pair=[a,b],mode_pair=[MODES[a],MODES[b]],same_mode=MODES[a]==MODES[b],common_decisions=n,chosen_differences=len(choice_diff),first_chosen_difference=choice_diff[0]if choice_diff else None,chosen_difference_rows=choice_diff,producer_logit_hash_differences=len(logit_diff),first_producer_logit_hash_difference=logit_diff[0]if logit_diff else None,windows=windows))
 raw_logits=sum(len(v)for n,v in files.items()if n.startswith('run/typed/pass')and n.endswith('-logits-BF16.bin'));computed_budget=physical_payload_bytes+raw_logits+2097152
 assert all_finite
 if not partial:
  assert computed_budget==report['raw_budget_used']<=128*1024**2
  metadata=sum(len(files[n])for n in ['run/typed/report.json','run/private-input-IDs.json']if n in files);assert metadata<=2097152
  watch=json.loads(files['control/watch.json']);entry=json.loads(files['run/entry-report.json']);assert watch['completed']and watch['passed']and not watch['forced_cleanup']and entry['completed']and entry['passed']and entry['normal_Native_shutdown']and entry['engine_worker_shutdown_completed']and entry['observed_all8_threads_exited']
  assert H(files['run/entry-report.json'])==watch['entry_report_SHA256']
  assert watch['publication_commit']=='d8de58251165dce8a317f7835811d2b1078e73b5'and watch['source_manifest_SHA256']=='b7872cb1636cc22e708e5c7c0b7a633d0e89ec9bcbacc7ae5520d370be065110'
  assert H(files['source/SOURCE_MANIFEST.json'])=='cf570e363016fac276037184ffdc3bdcd03f3bc3b8896ddae1c1031e0d3b552d'
  source=json.loads(files['source/SOURCE_MANIFEST.json']);assert all(H(files['source/'+name])==source['files_SHA256'][name]for name in ['TEACHER_REFERENCE.json','MODEL_PARENT.json'])
  assert entry['Native_SHA256']=='1b81667f28892a840c8396c57d9d8134ca40f94f6d0fa7c7c2deab5d1782643f'and entry['Native_derivation_verified']and entry['loaded_key_count']==67569 and entry['HC_mode']==0 and(entry['TP'],entry['PP'])==(8,1)and entry['diagnostic_only']and entry['HTTP']==0
  provider='/refactor/candidates/Deep-T1-MoE-activation-quant-r438-model-diagnostic-cpu-v2/libDeep_T1_MoE_capture_Model_r438v2.so'
  assert set(entry['actual_providers_before_weights'])=={'create','load','finish','forward','reset','destroy'}and set(entry['actual_providers_before_weights'].values())=={provider}
  assert len(entry['actual_diagnostic_providers']['providers'])==12 and set(entry['actual_diagnostic_providers']['providers'].values())=={provider}
  assert watch['actual_GPU']==entry['actual_step']=='910374.1378'and watch['CPU_keeper']=='910407.915'and watch['original_GPU_start_unix']==entry['gpu_start_epoch']==1793729746.9283948
  post=watch['post'];assert post['node']=='a02r4n01'and not post['kfd_pids']and not post['matching_component_processes']and post['ports_bindable']=={'43882':True}and len(post['physical_PCI'])==8
  accounts=json.loads(files['control/accounting.json']);assert all(any(r[0]==step and r[2:4]==['COMPLETED','0:0']for r in accounts)for step in ['910374.1378','910407.915',post['actual_step']])
 return dict(audit_passed=True,scope='Complete-pass0-only LIVE readonly snapshot'if partial else'Original1378 terminal fixed-teacher capture numerical diagnostic',terminal_audited=not partial,all_four_passes_audited=not partial and len(summary)==4,Model_quality_or_performance_qualified=False,original1374_quality_FAIL_unchanged=True,teacher_forced_not_free_generation=True,semantic_gold_not_evaluated=True,actual_original_teacher_count=83,prompt_count=8489,planned_modes=MODES,passes=summary,total_windows_checked=len(seen_windows),total_real_scope_role_records=688*len(seen_windows),all_typed_fields_finite=all_finite,same_GU_shadow_different_role_records=len(shadow),first_shadow_differences=sorted(shadow,key=lambda v:(v['pass_index'],v['output_index'],v['layer'],v['role'],v['owner']))[:16],cross_pass_comparisons=cross,observed_raw_payload_and_metadata_budget=computed_budget,limit_bytes=134217728,original_source_dispatch_8owner_Runtime_identities_stable=True,Native_logit_precision='Returned BF16 uint16; reported float scores/margin are exact diagnostic expansion only, not precast F32 decision margin',Native_sampling_precision_and_tie_policy='Not established by these capture records alone; BF16 zero margin does not prove pre-cast tie',earlier_logits_scope='All0..83 hashes/top8 are producer records; only82/83 fullBF16 raw independently rehashed and ranked',earlier_activation_scope='Only82/83 layer hidden/route/GU captured; equality here does not prove equality of earlier activations, cache or state',primary_quantized_buffers_scope='Independent same-GU SDK/fused shadow outputs, not primary main-path down-input buffers',root_cause='UNKNOWN',performance_interpretation=False,live_readonly_IO_observer=partial,verified_file_count=len(files))

def main():
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();files,manifest,digest=load(a.archive);q=analyze(files,manifest);q['archive_SHA256']=digest;text=json.dumps(q,indent=2)+'\n'
 if a.output:a.output.write_text(text,encoding='utf-8')
 print(text,end='')
if __name__=='__main__':main()
