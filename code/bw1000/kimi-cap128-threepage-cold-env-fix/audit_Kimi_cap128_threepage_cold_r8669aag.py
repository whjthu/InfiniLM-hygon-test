import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';name='Kimi-cap128-threepage-cold-r8669aag';a=json.loads((r/(name+'-launch.json')).read_text());w=json.loads((r/(name+'-watch.json')).read_text());run=a['run'];out={'run':run,'qualified':False,'capacity_qualified':False,'cleanup_qualified':False,'Model_fit_or_TPS':False,'owners':[],'original_failures':[]}
try:
 reports=[]
 for n in range(4):
  p=r/'results'/f'kimi-cap128-3page-cold-r8669aag-{run}-node{n}.json'
  if not p.exists():out['original_failures'].append({'node':n,'error':'missingreport'});continue
  q=json.loads(p.read_text());reports.append(q);out['owners'].append({'node':n,'report_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'normal8':q.get('all8_owned_workers_shutdown_completed',False),'completed':q.get('completed',False),'snapshots':q.get('snapshots',[]),'gate':q.get('global_memory_gate')})
  if q.get('error')or q.get('shutdown_error'):out['original_failures'].append({'node':n,'error':q.get('error'),'shutdown_error':q.get('shutdown_error')})
 out['terminal']=w.get('terminal');out['unique_post']=w.get('post_KFD');out['cleanup_qualified']=len(reports)==4 and all(v['normal8']for v in out['owners'])and w.get('post_KFD_passed',False)
 assert len(reports)==4,'missingreports'
 for q in reports:
  assert q['weight_load_calls']==q['finish_calls']==q['forward_calls']==0
  assert q['library_sha256']=='f6f10c2b243e0336074b872859e6ca6acb08effdcd56d86b9ac4df1bf2ddb5e9'
  assert q.get('metrics',{}).get('engine_calls',0)==0
  if q.get('completed'):
   assert [x['phase']for x in q['snapshots']]==['initial4pages_cap1','selected3pages_cap1','cap128_threepages_after_existingreset']
   assert q['snapshots'][-1]['geometry']==dict(max_rows=128,pages=387,page_size=256,KDA_pool_rows=129,pages_per_request=3,max_total_tokens=768)
   assert all(x[0]>3*1024**3 for x in q['snapshots'][0]['memory'])and q['global_memory_gate']['minimum_free_bytes']>4*1024**3
 out['capacity_qualified']=not out['original_failures']and all(q.get('completed')for q in reports)
 out['qualified']=out['capacity_qualified']and out['cleanup_qualified']and w['terminal']['State']=='COMPLETED'and w['terminal']['ExitCode']=='0:0'
 out['scope']='Coldpostcreate cap1->threepages->cap128existingreset admission only. Tinyoldcache/allocatorretention remain measured; no weights/activationpeak/finaltokens/performance qualification.'
 assert out['qualified'],('coldcapacity_not_qualified',out['original_failures'])
except BaseException as e:out['audit_error']=repr(e);raise
finally:
 p=r/'results'/f'{name}-audit-{run}.json';assert not p.exists();p.write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps({k:out.get(k)for k in ['qualified','capacity_qualified','cleanup_qualified','original_failures','audit_error']}))
