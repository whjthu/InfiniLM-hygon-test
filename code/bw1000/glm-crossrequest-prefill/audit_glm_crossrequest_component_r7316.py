import pathlib,json,hashlib,sys,subprocess,math
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';tag=sys.argv[1];p=r/'results'/('glm-crossrequest-component-r7316-'+tag+'.json');q=json.loads(p.read_text());assert q['completed']and q['library_SHA256']=='cb532ff5870415b450ed89dbfdf77f82430aaa6e947d6d2068df006d1236036f'
rows=[v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-23','-j',tag,'-n','-P','-o','JobID,State,ExitCode'],text=True).splitlines()];assert len(rows)==1 and rows[0][1:]==['COMPLETED','0:0']
owners=q['owners'];assert len(owners)==8 and sorted(o['rank']for o in owners)==list(range(8));labels=['warm-O','warm-N','O0','N0','N1','O1','O2','N2','N3','O3','restore']
for o in owners:
 assert o['completed']and o['normal_destroy_code']==0 and o['Runtime_expired']==1
 assert [a['label']for a in o['arms']]==labels
 for a in o['arms']:
  assert a['finite']and 0<=a['NRMSE']<.01 and a['total_elements']==8*512*8*256
  assert all(math.isfinite(a[k])and a[k]>0 for k in['native_attention_work_wall_seconds','complete_CAPI_call_wall_seconds'])
metrics={}
for clock in ['native_attention_work_wall_seconds','complete_CAPI_call_wall_seconds']:
 maxima={lab:max(o['arms'][i][clock]for o in owners)for i,lab in enumerate(labels)}
 ratios=[(maxima['O0']+maxima['O1'])/(maxima['N0']+maxima['N1']),(maxima['O2']+maxima['O3'])/(maxima['N2']+maxima['N3'])]
 off=[maxima[s]for s in['O0','O1','O2','O3','restore']];on=[maxima[s]for s in['N0','N1','N2','N3']]
 metrics[clock]={'per_arm_max8owner_seconds':maxima,'ABBA_ratios':ratios,'OFF_spread_over_mean':(max(off)-min(off))/(sum(off)/len(off)),'ON_spread_over_mean':(max(on)-min(on))/(sum(on)/len(on))}
result={'completed':True,'Slurm':rows,'normal8owners':True,'source_report_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'metrics':metrics,'max_NRMSE':max(a['NRMSE']for o in owners for a in o['arms']),'min_exact_fraction':min(a['exact_elements']/a['total_elements']for o in owners for a in o['arms']),'max_abs_error':max(a['max_abs_error']for o in owners for a in o['arms']),'component_argmax_differences':sum(a['component_argmax_rows']-a['component_lastaxis_argmax_equal']for o in owners for a in o['arms']if a['candidate']),'Model_qualified':False,'scope':'Synthetic QKV complete attention component only; metadata/packing/softmax/scatter within measured work; no communication/weights/ModelfinalIDs/TPS. Changed headbatch/strides may change kernel math. PostKFD separate unique watcher.'}
out=r/'results'/('glm-crossrequest-component-r7316-'+tag+'-audit.json');assert not out.exists();out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
