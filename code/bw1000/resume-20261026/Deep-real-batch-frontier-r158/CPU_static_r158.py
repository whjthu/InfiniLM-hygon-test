import pathlib,json,hashlib,ast,sys,os,importlib.util,time,traceback
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Deep-real-batch-frontier-r158';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();assert not B.exists();B.mkdir();q={'completed':False,'CPU_only':True,'Torch_imported':False,'Native_loaded':False,'model_loaded':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':os.uname().nodename};start=time.monotonic()
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 m=json.loads((D/'manifest-r158.json').read_bytes());q['source_manifest_SHA256']=sha(D/'manifest-r158.json')
 for n,h in m['source_SHA256'].items():
  assert sha(D/n)==h,n
  if n.endswith('.py'):ast.parse((D/n).read_bytes())
 cp=R/'candidates/deep-c4-smallt-model-r9254/CPU-build.json';assert sha(cp)=='43d4478102462e26bbfb0784942ec2f8f05228f53fb42eae0b23c051d1ed6d4c';c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only'];assert sha(c['libraries']['full']['path'])==m['Native_SHA256'];assert len(c['link']['full']['objects'])==48
 for n,h in c['link']['full']['objects'].items():assert sha(n)==h,n
 for n,h in c['actual_DTK_bindings'].items():assert sha(n)==h['SHA256'],n
 spec=importlib.util.spec_from_file_location('frontier_bootstrap_CPU',D/'bootstrap_r158.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);assert sha(module.BASE)==module.BASE_SHA;module.transform(module.BASE.read_bytes())
 olddir=module.BASE.parent;oldmanifest=json.loads((olddir/'manifest.json').read_bytes())
 for group,base in [('incoming_source_SHA256',olddir),('existing_incoming_dependency_SHA256',R/'incoming'),('existing_remote_SHA256',R)]:
  for n,h in oldmanifest[group].items():assert sha(base/n)==h,n
 prepared=json.loads((R/oldmanifest['tokenized_path']).read_bytes());gold=json.loads((D/'dataset64.json').read_bytes());planned=[x['id']for x in gold['cases']];available={x['id']:x for x in prepared['cases']};assert len(planned)==len(set(planned))==64 and set(planned)<=set(available);assert all(available[i]['input_tokens']<=512 for i in planned)
 assert 'torch'not in sys.modules;q.update(completed=True,Native_SHA256=m['Native_SHA256'],source_AST_passed=True,ordinary_generation_AST_reused=True,planned_questions=64,batches=[32,8,16,4],new_Native_compile=False)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q['elapsed_s']=time.monotonic()-start;(B/'CPU-static.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','actual_step','error','elapsed_s']}));assert q['completed']
