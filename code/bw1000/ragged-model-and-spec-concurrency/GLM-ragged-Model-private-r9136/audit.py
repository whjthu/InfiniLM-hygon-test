import pathlib,json,hashlib,math,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';inc=r/'incoming/GLM-ragged-Model-r9136';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def audit():
 launch=json.loads((r/'GLM-G100A-r9136-launch.json').read_text());m=json.loads((inc/'GPU-manifest.json').read_text());assert sha(inc/'GPU-manifest.json')==launch['manifest_SHA256'];run=launch['run'];out={'qualified':False,'run':run,'owners':[],'scope':'ActualnewMMLU32 anddistinctGSM32 C16M8/C16M16/C32M16 nativeclosedcohorts; runtime/performancebasicQA notformalqualitynoninferiority','quality_noninferiority_proven':False}
 for name,h in m['source_SHA256'].items():assert sha(inc/name)==h,name
 for rel,h in m['DSO_SHA256'].items():assert sha(r/rel)==h,rel
 program=json.loads((r/'results/GLM-actual-concurrency-tokenized-r9132.json').read_text());assert sha(r/'results/GLM-actual-concurrency-tokenized-r9132.json')==m['program_SHA256']
 for stage in [0,1]:
  for rank in range(8):
   path=r/'results'/f'glm-actual-concurrency-r9136-{run}-stage{stage}-rank{rank}.json';q=json.loads(path.read_text());assert q['completed']and q['worker_shutdown_completed']and q['post_close_Torch_liveness']and q['normal_reset_close']and q['token_exactness_gate']is False;assert q['library_sha256']==m['model_SHA256'];assert q['cache_after_destroy']['active']==q['cache_after_destroy']['active_bytes']==0;assert q['host_bank_after_destroy_r4504'][:3]==[38 if stage==0 else 40]*2+[0];assert q['benchmark_complete']and len(q['service'])==4 and len(q['MMLU'])==2
   for e in q['native_events']:assert e['ragged_counter_delta']==e['expected_ragged_counter_delta'];assert e['native_seconds']>=0 and math.isfinite(e['native_seconds'])and e['input_tokens']<=4096 and len(set(e['rows']))==len(e['rows'])
   out['owners'].append({'stage':stage,'rank':rank,'path':str(path),'SHA256':sha(path),'normal_closed':True,'actual_producer':q['actual_producer'],'minimum_driver_free_bytes':q['minimum_driver_free_bytes'],'final_cohort_counts':q['actual_final_cohort_counts']})
   if stage==1 and rank==0:
    for a in q['service']:
     assert len(a['requests'])==a['C']and len({x['id']for x in a['requests']})==a['C']and not a['budget_stop']
     for x in a['requests']:
      assert x['status']=='completed'and x['stop_reason']in ['EOS','max_new_tokens'];ev=x['events'];assert all(y['visible']and y['kind']=='committed'and not any(y[z]for z in ['EOS','pad','draft','replay'])for y in ev);assert [y['time']for y in ev]==sorted(y['time']for y in ev)
    assert len(q['MMLU'][0]['cases'])==len(q['MMLU'][1]['cases'])==32
    for arm in q['MMLU']:
     assert [x['id']for x in arm['cases']]==[x['id']for x in program['cases']]
     for row,case in zip(arm['cases'],program['cases']):assert len(row['choice_loglikelihoods'])==4 and all(math.isfinite(v)for v in row['choice_loglikelihoods'])and row['gold']==case['gold']and row['prediction']==max(range(4),key=lambda j:row['choice_loglikelihoods'][j])and row['flexible_correct']==(row['prediction']==row['gold'])
    out.update(MMLU=q['MMLU'],paired_MMLU=q.get('paired_MMLU'),service=q['service'],paired_GSM=q.get('paired_GSM'),coverage=q['benchmark_coverage'])
 rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910376','-n','-P','-o','JobID,JobName%100,State,ExitCode,Elapsed,NodeList%100'],text=True).splitlines()if x.split('|')[1]=='glm-G100A-Model-r9136'];assert len(rows)==1;out['Slurm']=rows;out['qualified']=rows[0][2:4]==['COMPLETED','0:0'];return out
if __name__=='__main__':print(json.dumps(audit()))
