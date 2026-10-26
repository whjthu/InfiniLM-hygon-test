import pathlib,ast,json,types,copy,tempfile,hashlib
b=pathlib.Path(__file__).resolve().parent;rt=b.parent/'Kimi-modern-PP4-cap32-private-r9145b';m=json.loads((rt/'model-manifest-PP4C32-r9145b.json').read_text());pin=json.loads((b/'model-source-pins-b1.json').read_text());run='CPU_ONLY_R9154B';res=json.loads((b/'resources.json').read_text());tree=ast.parse((b/'model_keeper_b1.py').read_text());fns=[x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='capacity'];checks=0
for p in b.glob('*.py'):ast.parse(p.read_text())
with tempfile.TemporaryDirectory()as td:
 r=pathlib.Path(td);(r/'results').mkdir();ns={'r':r,'json':json,'run':run,'m':m,'pin':pin,'w':{'groups':[{'actual_handle':'910375.123'},{'actual_handle':'910381.234'}]},'common':types.SimpleNamespace(RES=res)};exec(compile(ast.Module(body=fns,type_ignores=[]),'<actual capacity function>','exec'),ns);orig=[]
 for n in range(8):
  v={'run':run,'node':n,'globalnode':n,'host':m['nodes'][n],'actual_step':ns['w']['groups'][n//4]['actual_handle'],'manifest_SHA256':pin['manifest_SHA256'],'native_DSO_SHA256':pin['native_DSO_SHA256'],'capacity_gate_passed':True,'capacity_only_mode':False,'phase':'loading_weights','weight_payload_opened_at_publish':False,'inference_forward_calls_at_publish':0,'all64_min_free_bytes':4*1024**3+1,'layout':{'PP_stage':n//2,'TP':16,'PP':4},'PP_ranges':m['static_memory']['PP_ranges'],'geometry':{'max_rows':64,'pages':260,'page_size':256,'KDA_pool_rows':65,'pages_per_request':4,'max_total_tokens':1024},'model_weight_keys_validated':True,'actual_all93_coverage_validated':True,'actual_engine_key_stage_range':m['static_memory']['PP_ranges'][n//2]};orig.append(v)
 def write(rows):
  for n,v in enumerate(rows):(r/'results'/('Kimi-PP4-capacity-r9145b-'+run+'-node'+str(n)+'.json')).write_text(json.dumps(v))
 write(orig);assert len(ns['capacity']())==8;checks+=1
 for n in range(8):
  for key,bad in [('run','foreign'),('node',99),('globalnode',99),('host','foreign'),('actual_step','910384.1'),('manifest_SHA256','bad'),('native_DSO_SHA256','bad'),('capacity_gate_passed',False),('capacity_only_mode',True),('phase','other'),('weight_payload_opened_at_publish',True),('inference_forward_calls_at_publish',1),('all64_min_free_bytes',4*1024**3),('PP_ranges',[[0,93]]),('geometry',{'pages':132}),('model_weight_keys_validated',False),('actual_all93_coverage_validated',False),('actual_engine_key_stage_range',[0,93])]:
   rows=copy.deepcopy(orig);rows[n][key]=bad;write(rows)
   try:ns['capacity']()
   except AssertionError:checks+=1
   else:raise AssertionError((n,key,'not rejected'))
 write(orig);p=r/'results'/('Kimi-PP4-capacity-r9145b-'+run+'-node7.json');p.unlink();assert ns['capacity']()is None;checks+=1
q={'CPU_only':True,'GPU_submitted':False,'actual_capacity_function_executed':True,'checks':checks,'source_syntax_PASS':True,'manifest_SHA256':pin['manifest_SHA256'],'runtime_producer_SHA256':pin['runtime_producer_SHA256'],'actual_native_partition_evidence':'../Kimi-PP4-partition-audit-private-r9156/GPU-tiny-observed-a.json','actual_constructor_keys_still_requires_newModel':True,'pure_logic_only_noRuntime_dlopen':True};(b/'CPU-launch-contract-r9154b1.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
