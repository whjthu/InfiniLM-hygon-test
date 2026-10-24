import datetime,hashlib,json,pathlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
tag='910374.754'
names={'report':f'results/deep-B64-M64-r8644d-{tag}.json','audit':f'results/deep-B64-M64-r8644d-{tag}-audit.json','watcher':'results/deep-B64-M64-r8644d-watcher.json','launch':'results/Deep-B64-M64-r8644d-launch.json'}
data={k:json.loads((r/v).read_text())for k,v in names.items()}
q,a,w,launch=[data[k]for k in ['report','audit','watcher','launch']]
assert q['completed'] and q['engine_worker_shutdown_completed'] and q['outcome']=='NEGATIVE_QUALITY_STOP'
assert not q['quality_passed'] and not q['performance_qualified'] and not q['performance_adopted']
assert a['completed'] and a['normal8owner_shutdown'] and not a['performance_qualified']
assert w['status']=='TERMINAL_POSTKFD_VERIFIED'and w['audit']['exit']==0 and w['postKFD']['passed']
assert w['terminal'][0]==tag and w['terminal'][2:4]==['COMPLETED','0:0']
manifest=r/'incoming/Deep-B64-M64-config-r8644d-manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='04530171428b003132dda883ee2cd9c7ecf4c5a6da940c815d8f3d82b81c5649'
for n,h in json.loads(manifest.read_text())['sources_SHA256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h,n
ref=json.loads((r/'incoming/Deep-B32-samegeom-reference-r7269.json').read_text())
assert len(q['config_trials_r8644'])==1
c=q['config_trials_r8644'][0]
assert c['micro']==64 and c['gate26_passed'] and not c['complete'] and c['failure']['step']==30
assert c['failure']['finite'] and c['failure']['CPUargmax_equal']
assert len(c['generated_ids'])==64 and all(len(v)==31 for v in c['generated_ids'])
diff=[]
for i,actual in enumerate(c['generated_ids']):
 gold=ref['generated_ids'][i%32][:31]
 where=[s for s,(x,y)in enumerate(zip(actual,gold))if x!=y]
 if where:diff.append({'row':i,'first_zero_based_step':where[0],'actual_ID':actual[where[0]],'reference_ID':gold[where[0]]})
assert [v['row']for v in diff]==list(range(1,64,2))and all(v['first_zero_based_step']==30 and v['actual_ID']==8842 and v['reference_ID']==7831 for v in diff)
assert len(c['CAPI_windows'])==46
assert q['finite_before_passed']and q['after_QA_skipped']=='first_quality_failure'
for key in ['owner_affinity_before_weight_load','owner_affinity_after_weight_load','owner_affinity']:
 assert q[key]['verified']and q[key]['width']==4
sources={k:{'remote_path':str(r/v),'bytes':(r/v).stat().st_size,'SHA256':hashlib.sha256((r/v).read_bytes()).hexdigest()}for k,v in names.items()}
rows=[v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-24','-j','910374','-n','-P','-o','JobID,JobName%80,State,ExitCode,Elapsed,NodeList%100'],text=True).splitlines()if v.split('|')[1]in ['Deep-B64-M64-r8644d','Deep-B64-M64-postidle-r8644']]
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'REJECTED_CURRENT_0e051_B64_M64_FULL_ID_GATE','actual_step':tag,'Slurm':rows,'source_commit':'efb63d0','manifest_SHA256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'DSO_SHA256':q['library_sha256'],'SDK_SHA256':q['sdk_sha256'],'geometry':{'B':64,'M':64,'I':512,'maximum_O':128,'TP':8,'PP':1,'cards':8,'cache_rows':128,'cache_capacity':1024,'projection_tile':16,'C4_tile':16,'precision':5,'layout_HC_tail':[0,0,0],'prefill_chunk':4},'input_scope':q['input_scope'],'loading_seconds':q['loading_seconds'],'before_QA6_passed':True,'gate26_passed':True,'gate26_trajectory_seconds':c['gate26_elapsed_seconds'],'first_failure_zero_based_step':30,'first_failure_one_based_output_token':31,'mismatched_requests':32,'first_differences':diff,'finite_and_CPUargmax_at_failure':True,'actual_prefixes_all64':c['generated_ids'],'correctness_scope':'All64 rows exact first30 outputs; odd32rows differ at output31. Notfull128/EOSqualified. Actualfailedprefixes preserved; no rebaseline.','collected_trials':1,'M32_control_executed':False,'ONNO_executed':False,'later_arms_skipped':'Firstqualityfailure, persealedprotocol','after_QA_skipped':'first_quality_failure','performance_adopted':False,'performance_qualified':False,'throughput':'NOT_ADOPTED_OR_REPORTED_FOR_REJECTED_PREFIX','memory_scope':'Afterload/pre/post free-total snapshots only, notpeak/admissionguarantee','device_memory_snapshots':q['device_memory_snapshots'],'normal8owner_shutdown':True,'original_audit_exit':0,'original_watch_status':w['status'],'original_postKFD':w['postKFD'],'sources':sources,'no_resubmit':True,'decision':'STOP_B64_M64_ON_CURRENT_PROVIDER; keep699B32/M32 baseline. NoadditionalModelretry/shapegrid.'}))
