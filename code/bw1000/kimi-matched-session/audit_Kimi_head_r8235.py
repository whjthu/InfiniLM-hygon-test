import pathlib,json,hashlib,subprocess,datetime,statistics,ast,math
def audit():
 r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();a=json.loads((r/'Kimi-head-Model-r8235-launch.json').read_text());jobs=[s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-23','-j','910371','-n','-P','-o','JobID,JobName%90,State,ExitCode,Elapsed'],text=True).splitlines()if s.split('|')[1]=='Kimi-head-Model-r8235'];out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'jobs':jobs,'candidate_qualified':False,'performance_adopted':False,'issues':[],'cases':[],'nodes':[]}
 if len(jobs)!=1 or jobs[0][2]in['RUNNING','PENDING','COMPLETING','CONFIGURING']:return out
 issues=out['issues'];reports=[]
 if jobs[0][2:4]!=['COMPLETED','0:0']:issues.append('SlurmnotCOMPLETED0')
 for n in range(4):
  p=r/'results'/f'kimi-head-r8235-{a["run"]}-node{n}.json';q=json.loads(p.read_text());reports.append(q);out['nodes'].append({'node':n,'SHA256':sha(p),'completed':q.get('completed'),'closed':q.get('all8_owned_workers_shutdown_completed'),'resource_NOT_RUN':q.get('smallB_NOT_RUN'),'error':q.get('error')})
  if not(q.get('completed')and q.get('all8_owned_workers_shutdown_completed')and q.get('protocol_before_destroy_passed'))or q.get('error')or q.get('shutdown_error'):issues.append(f'{n}:lifecycle')
  if q.get('library_sha256')!=a['DSO_SHA256']or q.get('libkimi_grouped_prefill.so_sha256')!=a['helper_SHA256']:issues.append(f'{n}:binary')
  if q.get('final_head_policy')!=0 or q.get('final_N4_flag')!='0'or q.get('final_serving_policy')!='0':issues.append(f'{n}:finalpolicies')
 fp=r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json';assert sha(fp)=='ab686988a0952fe094d76773902e784e5d4be577c44c57c23493d702cc1d5625';f=json.loads(fp.read_text());gold16=next(z['fixed_complete_IDs']for z in f['geometries']if(z['B'],z['M'])==(16,4));eos=set(f['eos_token_ids']);tail=reports[3];base=tail.get('B1_baseline');gold1=None
 if base:
  p=pathlib.Path(base['path'].replace('/refactor',str(r),1));assert sha(p)==base['SHA256'];gold1=json.loads(p.read_text())['generated_ids'];out['B1_baseline']={**base,'historical_status':'NOT_AVAILABLE_MATCHED_ONLY'}
 expected_labels=['head-history-B16M4','head-freeze-B1M1','head-qualityON-B1M1','head-qualityON-B16M4']+[f'head-ABBA-B{B}-r{i}'for B in[1,16]for i in range(4)]+['head-restore-B1M1','head-restore-B16M4']
 checks=tail.get('smallB_checks',[]);lookup=[{c['label']:c for c in q.get('serving_cases',[])}for q in reports]
 # Reuse frozen8199 metric implementation verbatim except the explicit B1 acceptance.
 source=(r/'incoming/audit_Kimi_N4_Model_r8199.py').read_text();tree=ast.parse(source);ns={'math':math,'statistics':statistics};nodes=[v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name in ['summary','metrics']];code=ast.unparse(ast.Module(body=nodes,type_ignores=[])).replace('len(ids) == len(ts) == len(first) == 16','len(ids) == len(ts) == len(first) and len(ids) in [1, 16]');exec(code,ns)
 if any(q.get('smallB_checks',[])!=checks for q in reports):issues.append('crossnodechecks')
 if base and any(q.get('B1_baseline',{}).get('SHA256')!=base['SHA256']for q in reports):issues.append('B1baselinecrossnode')
 for check in checks:
  label=check['label'];assert label in expected_labels;B=check['B'];c=lookup[3][label];ref=gold1 if B==1 else gold16;exact=ref is not None and c['generated_ids']==ref
  if not exact:issues.append(label+':completeIDs')
  if not all(label in z and z[label]['generated_ids']==c['generated_ids']for z in lookup):issues.append(label+':crossnode')
  if (c['B'],c['frontend_cohort_M'],c['cache_pages_per_request'],c['head_policy'])!=(B,1 if B==1 else 4,4,check['candidate']):issues.append(label+':geometry')
  out['cases'].append({'label':label,'B':B,'head_policy':c['head_policy'],'reference_exact':exact,'scope':'same-loadOFFmatched-only'if B==1 else'historicalr4917','metrics':ns['metrics'](c,eos)})
 chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text());known=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text())
 for n,q in enumerate(reports):
  cases=lookup[n];labels=[c['label']for c in q.get('local_forward_samples',[])];rows=q.get('head_dispatch',[])
  if [c['label']for c in rows]!=labels:issues.append(f'{n}:headcountlabels')
  for row in rows:
   expected=([0,1,row['B']]if row['policy']else[1,0,0])if n==3 else[0,0,0]
   if row['delta']!=[expected]*8 or row['expected']!=expected:issues.append(f'{n}:headcounter')
  for epoch,value in[('OFF-before',0),('ON-before',1),('OFF-after',0)]:
   qa=[v for v in q.get('QA',[])if v['epoch']==epoch]
   if len(qa)!=2*len(chat['cases']):issues.append(f'{n}:QAcount:{epoch}')
   for z in qa:
    label=f'QA-{epoch}-p{z["policy"]}-{z["task"]}';c=cases[label];key='sort'if z['task'].startswith('sort_probe')else z['task'];gold=[x['generated_ids'][0]for x in known['cases']if x['task']==key and x['grouped']]
    if c['generated_ids'][0]not in gold or c['head_policy']!=value:issues.append(f'{n}:QAIDs:{label}')
  for row in q.get('N4_dispatch_r8167',[]):
   if row['delta']!=[[0,0]]*8:issues.append(f'{n}:N4mustOFF')
  if len(labels)!=len(set(labels)):issues.append(f'{n}:duplicateforwardlabels')
  for key,width in [('packed_W2_dispatch',8),('bounded_BF16_dispatch',4)]:
   if [v['label']for v in q.get(key,[])]!=labels:issues.append(f'{n}:{key}:labels')
   if any(v['enabled']is not False or v['delta']!=[[0]*width]*8 for v in q.get(key,[])):issues.append(f'{n}:{key}')
  if any(v['enabled']is not False for v in q.get('metadata_dispatch',[])):issues.append(f'{n}:metadataOFF')
 out['matrix_complete']=[c['label']for c in checks]==expected_labels and all(q.get('smallB_matrix_completed')and q.get('smallB_quality_passed')and q.get('smallB_restore_status')=='PASS'and not q.get('quality_failure')and not q.get('smallB_NOT_RUN')for q in reports);out['post_KFD_passed']=a.get('post_KFD_passed',False);out['candidate_qualified']=not issues and out['matrix_complete']and out['post_KFD_passed'];out['paired']={}
 for B in[1,16]:
  cs=[c for c in out['cases']if c['label'].startswith(f'head-ABBA-B{B}-')]
  if len(cs)==4:out['paired'][str(B)]={key:{'OFF':statistics.mean(cs[i]['metrics'][key]for i in[0,3]),'ON':statistics.mean(cs[i]['metrics'][key]for i in[1,2])}for key in ['aggregate_decode_tok_s','aggregate_decode_tok_s_per_card']};out['paired'][str(B)]['firstready_user_mean']={'OFF':statistics.mean(cs[i]['metrics']['first_ready_request_rate_summary']['mean']for i in[0,3]),'ON':statistics.mean(cs[i]['metrics']['first_ready_request_rate_summary']['mean']for i in[1,2])}
 out['numeric_failure']=any(not c['reference_exact']for c in out['cases'])or any(q.get('quality_failure')for q in reports);out['resource_NOT_RUN']=tail.get('smallB_NOT_RUN',[]);out['scope']='B1 same-loadmatched-only; B16 historical+QA proof; no Model VRAMsaving; CPUwall measurements do not claim GPUtimeline';return out
if __name__=='__main__':print(json.dumps(audit()))
