import pathlib,json,hashlib,sys,subprocess,statistics,math
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';tag=sys.argv[1];p=r/'results'/('Deep-routed-budget-r7342-'+tag+'.json');q=json.loads(p.read_text());assert q['completed']and q['destroy_code']==0 and q['Runtime_expired']==1 and q['producer']==tag
assert q['library_SHA256']=='35138a938ad2541ed4f88310c530992ffca23e452e1a072bf3f51a9f1b007834';calls=q['calls'];assert len(calls)==24
assert all(v['full_output_exact']and math.isfinite(v['complete_CAPI_wall_seconds'])and v['complete_CAPI_wall_seconds']>0 for v in calls)
assert len({v['output_SHA256']for v in calls})==1
groups={}
for mode in[0,1,2]:
 records=[c for c in calls if c['mode']==mode and not c['label'].startswith('warm')];times=[c['complete_CAPI_wall_seconds']for c in records];groups[str(mode)]={'count':len(times),'median_complete_CAPI_seconds':statistics.median(times),'min_seconds':min(times),'max_seconds':max(times),'mean_seconds':statistics.mean(times),'calls':records}
rows=[v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-23','-j',tag,'-n','-P','-o','JobID,State,ExitCode'],text=True).splitlines()];assert len(rows)==1 and rows[0][1:]==['COMPLETED','0:0']
result={'completed':True,'Slurm':rows,'normal_Runtime_expired':True,'all24_full_output_exact':True,'source_report_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'groups':groups,'observer_OFF_vs_original_median_ratio':groups['1']['median_complete_CAPI_seconds']/groups['0']['median_complete_CAPI_seconds'],'event_observer_tax_ratio':groups['2']['median_complete_CAPI_seconds']/groups['1']['median_complete_CAPI_seconds'],'Model_performance_qualified':False,'scope':'SingleGPU actualTP0 expertbanks, fourprefillprompts32row subset.Not699decode/Model.Layer/CCL. Sparseeventwindows includehostenqueuegaps/waits; notexclusiveGPUbusy. Nooptimizedcandidate orModelgain.'}
ap=r/'results'/('Deep-routed-budget-r7342-'+tag+'-audit.json');assert not ap.exists();ap.write_text(json.dumps(result,indent=2));print(json.dumps(result))
