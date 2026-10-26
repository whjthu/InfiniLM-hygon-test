import pathlib,json,hashlib,math,sys,subprocess,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';inc=r/'incoming/GLM-deferred-Model-r9226';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def audit():
 launch=json.loads((r/'GLM-deferred-Model-r9226-launch.json').read_bytes());m=json.loads((inc/'GPU-manifest.json').read_bytes());assert sha(inc/'GPU-manifest.json')==launch['manifest_SHA256'];run=launch['run'];result={'qualified':False,'run':run,'owners':[],'Model_SHA256':m['Model_SHA256'],'same_native_M16':True,'token_identity_gate':False,'scope':'TP8PP2 16cards same32distinct ABBA closedcohorts, serial vsmax2deferredfeedback; C16singlemicrocontrol, qualitypilot notformalnoninferiority'}
 for n,h in m['source_SHA256'].items():assert sha(inc/n)==h,n
 for n,h in m['existing_remote_SHA256'].items():assert sha(r/n)==h,n
 plan=json.loads((inc/'case_plan_r9226.json').read_bytes());program=json.loads((r/'results/GLM-B32-tokenized-r9183.json').read_bytes());ids=[c['id']for c in program['service_cases']];assert len(set(ids))==32
 for stage in [0,1]:
  for rank in range(8):
   path=r/'results'/f'glm-deferred-feedback-r9226-{run}-stage{stage}-rank{rank}.json';q=json.loads(path.read_bytes());assert q['completed']and q['worker_shutdown_completed']and q['post_close_Torch_liveness']and q['normal_reset_close']and q['benchmark_complete']and q['warm_CCL_progress_passed'];assert q['token_exactness_gate']is False and q['library_sha256']==m['Model_SHA256'];assert q['cache_after_destroy']['active']==q['cache_after_destroy']['active_bytes']==0;assert q['minimum_driver_free_bytes']>=512*1024**2 and q['host_bank_after_destroy_r4504'][:3]==[38 if stage==0 else 40]*2+[0];assert len(q['warmups'])==2 and [a['deferred_feedback']for a in q['warmups']]==[False,True];assert len(q['service'])==6
   for e in q['native_events']:assert e['ragged_counter_delta']==e['expected_ragged_counter_delta']and math.isfinite(e['native_seconds'])and e['native_seconds']>=0 and e['input_tokens']<=4096
   for a,w in zip(q['service'],plan['cohorts']):assert (a['label'],a['C'],a['M'],a['deferred_feedback'])==(w['label'],w['C'],w['M'],w['deferred_feedback'])and not a['budget_stop']
   result['owners'].append({'stage':stage,'rank':rank,'SHA256':sha(path),'actual_step':q['actual_producer'],'warmups':q['warmups'],'minimum_driver_free_bytes':q['minimum_driver_free_bytes'],'normal':True})
   if stage==1 and rank==0:
    for a in q['service']:
     assert [x['id']for x in a['requests']]==ids[:a['C']]
     for req in a['requests']:
      assert req['status']=='completed'and req['stop_reason']in['EOS','max_new_tokens'];ee=req['events'];assert [v['time']for v in ee]==sorted(v['time']for v in ee)and all(v['visible']and v['kind']=='committed'and not any(v[z]for z in ['EOS','pad','draft','replay'])for v in ee)
    result.update(service=q['service'],paired_GSM=q['paired_GSM'],warmups=q['warmups'],coverage=q['benchmark_coverage'])
 result['qualified']=True;return result
if __name__=='__main__':print(json.dumps(audit()))
