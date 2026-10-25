import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';a=json.loads((r/'Kimi-capacity96-service-r8669aaj-launch.json').read_text());w=json.loads((r/'Kimi-capacity96-service-r8669aaj-watch.json').read_text());run=a['run'];out={'qualified':False,'owners':[],'terminal':w.get('terminal'),'post_KFD_passed':w.get('post_KFD_passed',False)}
try:
 reports=[]
 for node in range(4):
  p=r/'results'/f'kimi-capacity96-service-r8669aaj-{run}-node{node}.json';q=json.loads(p.read_text());reports.append(q);out['owners'].append({'node':node,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'normal8':q.get('all8_owned_workers_shutdown_completed',False),'error':q.get('error'),'shutdown_error':q.get('shutdown_error')})
  assert q.get('completed')and not q.get('error')and not q.get('shutdown_error')
  assert q['cache_geometry']==dict(max_rows=96,pages=291,page_size=256,KDA_pool_rows=97,pages_per_request=3,max_total_tokens=768)
  cases=[v for v in q['serving_cases']if v['label'].startswith(('B64-','B96-'))];assert [v['B']for v in cases]==[64,96]
  assert all(v['frontend_cohort_M']==16 and v['policy']==1 and v['max_output_tokens']==128 and v['matched_baseline_complete_IDs'] is True and not v['instrumented_baseline'] for v in cases)
  assert len(q['QA'])==8 and all(v['complete_frozen_IDs_exact']for v in q['QA'])
  assert q['geometry_memory_gate']['minimum_free_bytes']>4*1024**3 and q['preload_memory_gate']['minimum_free_bytes']>4*1024**3
  assert all(v['delta']==[v['expected']]*8 for v in q['counter_dispatch'])
 assert all(v['normal8']for v in out['owners'])and w['post_KFD_passed']and w['terminal']['State']=='COMPLETED'and w['terminal']['ExitCode']=='0:0'
 for B in (64,96):
  arms=[next(v for v in q['serving_cases']if v['label']=='B'+str(B)+'-M16-ordinary128')for q in reports];assert all(v['generated_ids']==arms[0]['generated_ids']for v in arms)
 out['all32normal']=True;out['qualified']=True;out['scope']='Sameone-loadB64M16/B96M16 completefrozenmappedIDs/EOS+QA; singleorder notABBA; nohistoricalprofilegainclaim.'
except BaseException as e:out['error']=repr(e);raise
finally:
 p=r/'results'/f'Kimi-capacity96-service-r8669aaj-audit-{run}.json';assert not p.exists();p.write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps({'qualified':out['qualified'],'error':out.get('error')}))
