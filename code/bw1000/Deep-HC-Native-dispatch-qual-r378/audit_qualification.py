"""Read-only original378 report/watch audit. No Native, torch, GPU or model imports."""
import pathlib,json,hashlib,math,sys
PHASES=['defaultOFF','epilogueON_fullpreOFF','fullpreON','clear_restoreOFF']
NATIVE='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
BRIDGE='bc3bafbe37176d67a61c9c73cd0268e46d6b3513eaf8b729a7f92888a739b25d'
def audit(path):
 path=pathlib.Path(path);j=json.loads(path.read_bytes());rw=j['report'];ww=j['watch'];r=rw['data'];w=ww['data'];p=r['protocol']
 for item in [rw,ww]:assert hashlib.sha256((json.dumps(item['data'],indent=2)+'\n').encode()).hexdigest()==item['SHA256']
 assert rw['SHA256']=='f16fb44051a20998b1a7b66fb9c6b39d309d4fcf0fec0f500c341acfa80ab310'
 assert w['publication_commit']=='fe2029880e9a54abeef92993f88dc6a3a4199067'and w['manifest_SHA256']=='62db684a8490dcce456693aac3d9b485428711896772f373feb509a60f245e60'
 assert r['completed']and r['passed']and w['completed']and w['passed']and r['actual_step']==w['actual_GPU']=='910374.1307'and w['CPU_keeper_actual']=='910407.758'
 assert not r['Model_loaded']and not r['performance_experiment']and r['actual_calls']==r['planned_calls']==32 and r['Native_SHA256']==NATIVE and r['bridge_SHA256']==BRIDGE
 assert not p['errors']and p['threads_joined']and len(p['records'])==4 and len(p['closes'])==8
 assert w['srun_client_exit']==0 and w['normal_close_all8']and w['threads_joined']
 for suffix in ['fresh_before','post']:
  v=w[suffix];assert v['node']=='a02r4n04'and v['KFD_available']and not v['kfd_pids']and not v['matching_component_processes']
 assert w['fresh_before']['actual_step']=='910374.1306'and w['post']['actual_step']=='910374.1308'
 assert all(v.split('|')[2:4]==['COMPLETED','0:0']for v in j['sacct'])and len(j['sacct'])==4
 assert p['records'][0]['counters_before']['old']==[[1,1,0,0,0,0,0]for _ in range(8)]
 phases=[];maxn=0.;masks=[];initial=[];tids=[];fixtures=[]
 for index,name in enumerate(PHASES):
  rec=p['records'][index];assert rec['phase']==name and rec['actual_calls']==8
  old=[[a-b for a,b in zip(x,y)]for x,y in zip(rec['counters_after']['old'],rec['counters_before']['old'])];new=[[a-b for a,b in zip(x,y)]for x,y in zip(rec['counters_after']['new'],rec['counters_before']['new'])]
  expected_old=[0,0,1,1,1 if index==1 else 0,0,0];expected_new=[1,1,1,1,0,0,0,0,0,0]if index==2 else [0]*10
  assert old==rec['old_delta']==[expected_old]*8 and new==rec['new_delta']==[expected_new]*8
  assert all(not any(row)for row in rec['counters_before']['new']);assert [a-b for a,b in zip(rec['counters_after']['canonical'],rec['counters_before']['canonical'])]==[1]*8
  assert len(rec['numerics'])==8
  for rank,call in enumerate(rec['numerics']):
   assert call==p['closes'][rank]['calls'][index]and call['phase']==name and call['output_poison_overwritten_finite']and call['canaries']and call['state_unchanged']
   assert [v['name']for v in call['numerics']]==['collapsed_BF16','post','comb']
   for v in call['numerics']:
    assert math.isfinite(v['NRMSE'])and math.isfinite(v['max_abs'])and v['NRMSE']<(.005 if v['name']=='collapsed_BF16'else 1e-4);maxn=max(maxn,v['NRMSE'])
  phases.append(dict(phase=name,owner_calls=8,per_owner_old_delta=expected_old,per_owner_fullpre_delta=expected_new))
 assert p['records'][-1]['counters_after']['old']==[[1,1,4,4,1,0,0]for _ in range(8)]
 assert p['records'][-1]['counters_after']['canonical']==[4]*8
 for rank,c in enumerate(p['closes']):
  assert c['rank']==c['device']==rank and c['normal_close']and c['Runtime_expired']==1 and c['destroy_code']==0 and len(c['calls'])==4
  assert c['state_initial']==[v['SHA256']for v in c['fixture']['weights']]
  assert c['fixture']['rank']==rank and c['fixture']['seed']==[781,1889][rank%2]and 'Synthetic'in c['fixture']['input_kind']
  mask=set(c['calls'][0]['final_allowed']);assert len(mask)==4 and all(set(v['final_allowed'])==mask for v in c['calls']);masks.append(mask);initial.append(c['initial_allowed']);tids.append(c['TID']);fixtures.append({k:c['fixture'][k]for k in ['rank','layer','kind','seed','input_SHA256','weights','CPU_reference_output_SHA256']})
 assert len(set(tids))==8 and all(v==r['parent_initial_allowed']and len(v)>=32 for v in initial)and all(not masks[i]&masks[j]for i in range(8)for j in range(i))
 assert all(v['mapped']for v in r['provider_maps'].values())and any(v['SHA256']==NATIVE for v in r['provider_maps'].values())and any(v['SHA256']==BRIDGE for v in r['provider_maps'].values())
 assert len(r['old_afc6_976f_Native_not_mapped'])==2 and r['default_OFF_restored']and r['no_ABBA']and r['no_speedup_claim']
 return dict(passed=True,scope='Actual a314 HC dispatch/counters/numerics/state/temporary lifetime/8Runtime close only',actual_GPU=w['actual_GPU'],actual_keeper=w['CPU_keeper_actual'],actual_calls=32,phases=phases,max_NRMSE=maxn,owners=8,Runtime_expired=8,threads_joined=True,logical_CPU_masks=[sorted(v)for v in masks],physical_core_or_SMT_isolation='NOT_CLAIMED',provider_maps=r['provider_maps'],old_afc6_976f_Native_not_mapped=r['old_afc6_976f_Native_not_mapped'],fixtures=fixtures,fresh=w['fresh_before'],post=w['post'],Slurm=j['sacct'],whole_driver_wall_s=r['whole_driver_wall_s'],shared_owner_setup_calls_checks_close_wall_s=p['whole_functionality_wall_s'],imports_fixture_DSO_pin_prepare_wall_s=r['whole_driver_wall_s']-p['whole_functionality_wall_s'],actual_GPU_busy='NOT_MEASURED',source_publication_commit=w['publication_commit'],GPU_manifest_SHA256=w['manifest_SHA256'],Native_SHA256=NATIVE,bridge_SHA256=BRIDGE,raw_report_path=rw['path'],raw_report_SHA256=rw['SHA256'],raw_watch_path=ww['path'],raw_watch_SHA256=ww['SHA256'],local_archive=str(path),local_archive_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),performance_ABBA=0,new_performance_round=False,Model_loaded=False,Model_qualification='NOT_RUN',Model_gain='UNMEASURED',no_new_GPU_or_Model_required=True,Native976f_375_unchanged=True)
if __name__=='__main__':print(json.dumps(audit(sys.argv[1]),indent=2)+'\n')
