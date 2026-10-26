import pathlib,json,hashlib,re,sys
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');hp=pathlib.Path(sys.argv[1]);w=json.loads(hp.read_text());run=w['run'];m=json.loads((r/'incoming/Kimi-modern-PP4-cap32-r9145b/model-manifest-PP4C32-r9145b.json').read_text());q={'run':run,'qualified_runtime':False,'quality_no_regression_proven':False,'GPU_Model_perf_claim':False,'owners':[]}
try:
 assert w['all_target_terminal']and w['post_KFD_passed'];assert all(g['terminal'][2:4]==['COMPLETED','0:0']for g in w['groups']);reports=[]
 for node in range(8):
  p=r/'results'/('Kimi-modern-PP4-r9145b-'+run+'-node%d.json'%node);raw=p.read_bytes();d=json.loads(raw);reports.append(d);assert d['completed']and d['all8_owned_workers_shutdown_completed']and d['serving_protocol_complete']and not d.get('error')and not d.get('shutdown_error');assert d['library_sha256']==m['native_DSO_SHA256']and d['manifest_SHA256']==w['manifest_SHA256'];assert d['capacity_gate_passed']and d['weight_payload_opened'];q['owners'].append({'node':node,'normal8':True,'SHA256':hashlib.sha256(raw).hexdigest()})
 reference=[(c['label'],[(v['logical'],v['generated_ids'])for v in c['request_metrics']])for c in reports[6]['batch_cases']]
 for d in reports:assert [(c['label'],[(v['logical'],v['generated_ids'])for v in c['request_metrics']])for c in d['batch_cases']]==reference
 draft=w['draft'];assert draft['terminal'][2:4]==['COMPLETED','0:0'];dr=json.loads(pathlib.Path(draft['report']).read_text());assert dr['status']=='COMPLETED'and dr['normal_close']and not dr['remaining_request_keys_before_close'];logs=[]
 for node in range(8):
  g=w['groups'][node//4];p=r/'logs'/(g['name']+'-'+g['actual_handle']+'-'+str(node%4)+'.log');raw=p.read_bytes();text=raw.decode(errors='replace');assert text.count('KIMI_MODERN_BATCH_R9143_NORMAL')==1 and 'Using network IBext_v8'in text and 'Using network Socket'not in text;assert not any(re.search(r'rocBLAS.*(?:error|709)|Traceback \(most recent|HIP error|Segmentation fault',s,re.I)for s in text.splitlines());logs.append({'node':node,'SHA256':hashlib.sha256(raw).hexdigest()})
 dp=r/'logs'/(draft['name']+'-'+draft['actual_handle']+'.log');assert dp.read_text().count('KIMI_DRAFT_TRUEBATCH_PP4_R9145B_NORMAL')==1
 q.update(qualified_runtime=True,all64normal=True,draft_normal=True,logs=logs,case_count=len(reference),concurrency_coverage=sorted({c['configured_C']for c in reports[6]['batch_cases']}),scope='actual93/TP16PP4/cap32/truebatchdraft; noqualityparityorSLOclaim')
except BaseException as e:q['error']=repr(e);raise
finally:
 p=r/'results'/('Kimi-PP4-model-audit-r9154b-'+run+'.json');assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q))
