from pathlib import Path
import json,hashlib,ast,shutil
r=Path('.');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((r/'Kimi-TP16PP2-I192-manifest-r8669zc.json').read_text());runtime=[]
for n in old['runtime_files']:
 new=n.replace('r8669zc','r8669zh');assert not (r/new).exists()
 s=(r/n).read_text(encoding='utf-8').replace('r8669zc','r8669zh')
 s=s.replace('modern_kimi_I192_workspace_r8669zh','modern_kimi_I192_workspace_r8669zc')
 s=s.replace('results/Kimi-I192-compact-CPU-r8669zh.json','results/Kimi-I192-smoke-static-closure-r8669zg.json')
 s=s.replace('CPU211compactI192native/eagerexactTP16/defaultOFFoptimization','CPU214linked/sharedactualcaller612564/smoke853qualified/defaultOFFoptimization')
 s=s.replace('Bind the CPU211 private compact-workspace Model','Bind the smoke853-qualified shared-actual-caller Model')
 (r/new).write_text(s,encoding='utf-8',newline='\n');runtime.append(new)

gold='results/kimi-TP16PP2-I192-r8669zc-TP16PP2-I1928669za-f29b4d4db70b0b8b-node2.json'
goldsha='5c4c52c8895858be6eed7d5d48a60d2499c0d779c170cf55d11b7d64e11b91f2'
helper='''def frozen648_reference_r8669zh(r,inputs,eos,actual,report,save):
 p=r/"GOLDPATH";raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=="GOLDSHA"
 q=json.loads(raw);assert(q['TP'],q['PP'],q['node'],q['PP_stage'])==(16,2,2,1)
 assert q['cache_geometry']==dict(max_rows=16,pages=68,page_size=256,KDA_pool_rows=17,pages_per_request=4,max_total_tokens=1024)
 cases=[v for v in q['serving_cases']if v['label']=='TP16-B16M16-O-full-reference128'];assert len(cases)==1
 z=cases[0];assert(z['B'],z['frontend_cohort_M'],z['policy'],z['I192_candidate_ON'],z['timed_decode_token_only'])==(16,16,1,False,False)
 assert z['input_lengths']==[512]*16 and len(inputs)==16 and all(len(v)==512 for v in inputs)
 expected=z['generated_ids'];assert len(expected)==16 and all(len(v)==128 for v in expected)
 first=[]
 for i,(a,b)in enumerate(zip(expected,actual)):
  pos=next((j for j,(x,y)in enumerate(zip(a,b))if x!=y),None)
  if pos is not None or len(a)!=len(b):first.append({'request':i,'first_different_ordinal':None if pos is None else pos+1,'expected':None if pos is None else a[pos],'actual':None if pos is None else b[pos],'expected_length':len(a),'actual_length':len(b)})
 report['frozen648_original_reference']={'report_path':str(p),'raw_SHA256':"GOLDSHA",'label':z['label'],'matched_full_IDs_EOS':actual==expected,'first_differences':first,'input_mapping':'r4917 frozen two natural I512 prompts alternated rows1..16, prefillM2/decodeM16/TP16PP2/flags3/head0/SpecMath0/serving1/I192OFF','input_ids_JSON_SHA256':hashlib.sha256(json.dumps(inputs,separators=(',',':')).encode()).hexdigest(),'eos_token_ids':sorted(eos),'scope':'648 original OFF arm completed; 648 candidate/session FAILED. Independent frozen original baseline only, not candidate-generated gold or inherited TPS.'};save()
 assert actual==expected,('baseline_reproducibility_FAILED_OBSERVED',first)
 return expected
'''.replace('GOLDPATH',gold).replace('GOLDSHA',goldsha)
p=r/'kimi_TP16PP2_I192_protocol_r8669zh.py';s=p.read_text();s=helper+'\n'+s
needle=" fixed=generate(inputs,16,'TP16-B16M16-O-full-reference128',128,False)";assert s.count(needle)==1
s=s.replace(needle,needle+'\n fixed=frozen648_reference_r8669zh(r,inputs,eos,fixed,report,save)')
s=s.replace("'OriginalI192genericfallback/SpecMath0/flags3, oneprocessfrozenfull128IDs+EOS, nothistoricalTP8gold orcandidate-generatedgold'","'Fresh original OFF flags3 fullIDs/EOS must reproduce independent frozen648 OFF16x128;648 candidate/session failed. Candidate uses this independent reference;ordinarybothflags7.'")
p.write_text(s,encoding='utf-8',newline='\n')

failfast='''def owned_native_failfast_r8669zh(root,row,jobname,expected_run,watch,persist,signal_step):
 # Only the unique watcher may signal its exactly observed original numeric step.
 # No regex glob, other nonce, generic warnings, or partial token mismatch triggers.
 if watch.get('owner_error_failfast')is not None:return False
 handle=row['JobID']
 assert row['JobName']==jobname and jobname=='Kimi-TP16PP2-I192-'+expected_run
 assert watch['run']==expected_run and watch['actual_handle']==handle
 assert handle.startswith('910375.')and handle.split('.')[1].isdigit()and handle.count('.')==1
 for node in range(4):
  p=root/'logs'/f'{jobname}-{handle}-{node}.log'
  if not p.is_file():continue
  with p.open('rb')as f:
   f.seek(0,2);size=f.tell();f.seek(max(0,size-1024*1024));raw=f.read()
  line=next((v for v in raw.decode(errors='replace').splitlines()if 'exception during forward'in v or 'I192launchcode='in v),None)
  if line is None:continue
  watch['owner_error_failfast']={'run':expected_run,'actual_handle':handle,'node':node,'path':str(p),'tail_SHA256':hashlib.sha256(raw).hexdigest(),'tail_bytes':len(raw),'original_file_bytes':size,'first_observed_line':line,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'action':'exact owned step TERM once; follow same terminal/audit/post','signal_attempted':True};persist()
  try:
   z=signal_step(['scancel','--signal=TERM',handle],capture_output=True,text=True,timeout=30)
   watch['owner_error_failfast'].update(signal_returncode=z.returncode,signal_stdout=z.stdout[-2000:],signal_stderr=z.stderr[-2000:])
  except BaseException as e:watch['owner_error_failfast']['signal_error']=repr(e)
  persist();return True
 return False
'''
p=r/'launch_Kimi_TP16PP2_I192_r8669zh.py';s=p.read_text();s=s.replace("run='TP16PP2-I1928669za-'","run='TP16PP2-I1928669zh-'")
needle='if len(sys.argv)==1:';assert s.count(needle)==1;s=s.replace(needle,failfast+'\n'+needle)
needle="    if norm(row['State']) in TERMINAL:";assert s.count(needle)==1
s=s.replace(needle,"    if norm(row['State']) not in TERMINAL:owned_native_failfast_r8669zh(r,row,jobname,run,w,lambda:write(wp,w),subprocess.run)\n"+needle)
s=s.replace('post-r8669z\'','post-r8669zh\'').replace('before-r8669z\'','before-r8669zh\'');p.write_text(s,encoding='utf-8',newline='\n')
p=r/'audit_Kimi_TP16PP2_I192_r8669zh.py';s=p.read_text();needle="  assert q['library_sha256']==m['native_DSO_SHA256'];";assert s.count(needle)==1
s=s.replace(needle,"  assert q['frozen648_original_reference']['matched_full_IDs_EOS']and q['frozen648_original_reference']['raw_SHA256']=='"+goldsha+"'\n"+needle);p.write_text(s,encoding='utf-8',newline='\n')

cpu=json.loads((r/'Kimi-I192-smoke-static-closure-r8669zg.json').read_text());native=cpu['receipt']['native'];nh=cpu['receipt']['native_SHA256'];ch=cpu['raw_receipt_SHA256'];assert nh=='6125640af7f5878e08f00f92527b3c975738bf3fc3563cb2b6afabb5dbbf6fc8'
m=old.copy();m.update(revision='8669zh',status='PREPARED_PENDING_ROOT_FINAL_MODEL_GO',GPU='NOT_SUBMITTED',native=native,native_DSO_SHA256=nh,CPU_receipt_SHA256=ch,runtime_files=runtime,root_final_GO='PENDING_ROOT_FINAL_SHA_REVIEW; CPU/package preparation authorized after853 actual-caller smoke passed.',scope='TP16PP2/32BW1000/B16M16 I512/O128; same612564 actualcaller DeviceGuard fix qualified by853 synthetic caller contract. Frozen648 original OFF16x128 reference;freshOFF must reproduce beforecandidate;ordinaryOFF7/ON7/ON7/OFF7 only. Same compact/reset/memorygates, oneweightsload;noacceptedgain before finalIDs/EOS/QA/32normal/audit/post.')
m['remote_evidence_sha256']=dict(old['remote_evidence_sha256']);m['remote_evidence_sha256'].update({'results/Kimi-I192-smoke-static-closure-r8669zg.json':ch,native.replace('/refactor/',''):nh,gold:goldsha})
m['frozen_original_reference']={'path':gold,'raw_SHA256':goldsha,'label':'TP16-B16M16-O-full-reference128','producer':'910375.648 originalOFF arm completed;candidate/session failed','mapping':'same r4917 immutable I512 two alternating natural prompts,rows1..16,prefillM2,decodeM16,head0,flags3/Math0/serving1/I192OFF'}
m['owner_error_failfast']={'source':'unique original watcher only','match':['exception during forward','I192launchcode='],'logs':'exact thisrun/actualstep/node0..3 only, bounded1MiB tail','action':'one exact scancel --signal=TERM parent.step; persistoriginalerror then followoriginalterminal/audit/post','never':'no differentnonce/parentjob/broadcancel/timeoutreplay'}
m['native']=''+native;m['source_sha256']={n:sha(r/n)for n in runtime}
mp=r/'Kimi-TP16PP2-I192-manifest-r8669zh.json';mp.write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8',newline='\n')
for n in runtime:
 if n.endswith('.py'):ast.parse((r/n).read_text())
print('Prepared7runtime+manifest; same612 native; NO GPU GO/submit',sha(mp),sha(r/'launch_Kimi_TP16PP2_I192_r8669zh.py'))
