import pathlib,json,hashlib,ast,sys,os,importlib.util,time,traceback
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Deep-C1-decode-profile-r168a';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();assert not B.exists();B.mkdir();q={'completed':False,'CPU_only':True,'Torch_imported':False,'Native_loaded':False,'model_loaded':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':os.uname().nodename};start=time.monotonic()
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 m=json.loads((D/'manifest-r168a.json').read_bytes());q['source_manifest_SHA256']=sha(D/'manifest-r168a.json')
 for n,h in m['source_SHA256'].items():
  assert sha(D/n)==h,n
  if n.endswith('.py'):ast.parse((D/n).read_bytes())
 cp=R/'candidates/deep-c4-smallt-model-r9254/CPU-build.json';assert sha(cp)=='43d4478102462e26bbfb0784942ec2f8f05228f53fb42eae0b23c051d1ed6d4c';c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only'];assert sha(c['libraries']['full']['path'])==m['Native_SHA256'];assert len(c['link']['full']['objects'])==48
 for n,h in c['link']['full']['objects'].items():assert sha(n)==h,n
 for n,h in c['actual_DTK_bindings'].items():assert sha(n)==h['SHA256'],n
 spec=importlib.util.spec_from_file_location('frontier_bootstrap_CPU',D/'bootstrap_r168a.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);assert sha(module.BASE)==module.BASE_SHA;module.transform(module.BASE.read_bytes())
 olddir=module.BASE.parent;oldmanifest=json.loads((olddir/'manifest.json').read_bytes())
 for group,base in [('incoming_source_SHA256',olddir),('existing_incoming_dependency_SHA256',R/'incoming'),('existing_remote_SHA256',R)]:
  for n,h in oldmanifest[group].items():assert sha(base/n)==h,n
 prepared,gold,cases,data_proof=module.preflight_dataset();q['data_preflight_r168a']=data_proof
 regression=importlib.util.spec_from_file_location('regression_r168a',D/'CPU_data_regression_r168a.py');rm=importlib.util.module_from_spec(regression);regression.loader.exec_module(rm);q['shared_data_contract_regressions']=rm.run_cases(B/'CPU-MOCK-data-fixtures');assert q['shared_data_contract_regressions']['passed']
 schema=importlib.util.spec_from_file_location('schema_r168a',D/'CPU_schema_r168a.py');sm=importlib.util.module_from_spec(schema);schema.loader.exec_module(sm);assert sm.main()['passed'];import subprocess
 nm=subprocess.check_output(['nm','-D','--defined-only',c['libraries']['full']['path']],text=True);required=['deep_owner_arm_r8900','deep_owner_read_r8900','deep_attention_inner_arm_r9193','deep_attention_inner_read_r9193','modern_deep_latent_fused_calls_r681','modern_deep_latent_fallback_calls_r681','modern_deep_FP32_calls_r3597'];assert all(x in nm for x in required)
 assert 'torch'not in sys.modules;q.update(completed=True,Native_SHA256=m['Native_SHA256'],source_AST_passed=True,Native_forward_AST_reused=True,planned_questions=4,batches=[1],observer_only=True,generation_schedule_changed=False,new_Native_compile=False)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q['elapsed_s']=time.monotonic()-start;(B/'CPU-static.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','actual_step','error','elapsed_s']}));assert q['completed']
