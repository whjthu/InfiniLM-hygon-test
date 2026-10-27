"""Static only. Production library paths; labelled gates persist evidence before assert."""
import pathlib,json,hashlib,os,sys,ast,subprocess,importlib.util,importlib.metadata,time,traceback
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Deep-agent-p0-r143c';MODEL=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');BASE=R/'candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'static_only':True,'GPU':False,'GPU_initialized':False,'Torch_imported':False,'Native_loaded':False,'standard_server_imported':False,'model_constructed':False,'runtime_import':'DEFERRED_TO_EIGHT_ASSIGNED_GPU_JOB','actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'gates':[]};start=time.monotonic()
def gate(label,actual,expected=True):
 q['stage']=label;q['gates'].append({'label':label,'actual':actual,'expected':expected,'pass':actual==expected});assert actual==expected,label
try:
 gate('new_private_candidate',not B.exists());B.mkdir(exist_ok=False)
 gate('hidden_GPU_environment',all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']))
 manifest=D/'source-manifest-r143.json';m=json.loads(manifest.read_bytes());q['source_manifest_SHA256']=sha(manifest)
 for name,h in m['source_SHA256'].items():gate('source:'+name,sha(D/name),h)
 gate('model_config_SHA256',sha(MODEL/'config.json'),'fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09')
 gate('model_index_SHA256',sha(MODEL/'model.safetensors.index.json'),'7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a')
 cfg=json.loads((MODEL/'config.json').read_bytes());gate('model_layers',cfg['num_hidden_layers'],43);gate('model_type',cfg['model_type'],'deepseek_v4')
 cp=R/'candidates/deep-c4-smallt-model-r9254/CPU-build.json';gate('parent_CPU_SHA256',sha(cp),'43d4478102462e26bbfb0784942ec2f8f05228f53fb42eae0b23c051d1ed6d4c');c=json.loads(cp.read_bytes());native=pathlib.Path(c['libraries']['full']['path']);gate('parent_completed',c['completed']);gate('parent_CPU_only',c['CPU_only']);gate('Native_SHA256',sha(native),m['Native_SHA256']);gate('linked_object_count',len(c['link']['full']['objects']),48)
 for p,h in c['link']['full']['objects'].items():gate('object:'+p,sha(p),h)
 gate('DSO_pin_count',len(c['actual_DTK_bindings']),5)
 for p,v in c['actual_DTK_bindings'].items():gate('DSO:'+p,sha(p),v['SHA256'])
 source=R/'incoming/Deep-C4-independent64-r9278/modern_Deep_independent64_r9278.py';gate('qualified_driver_SHA256',sha(source),m['qualified_driver_SHA256']);raw=source.read_bytes();q['stage']='qualified_driver_AST';ast.parse(raw)
 q['stage']='source_AST_transform';spec=importlib.util.spec_from_file_location('deep_bootstrap_CPU',D/'native_bootstrap_r143.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);_,proof=mod.transform(raw);q['AST_loader_proof']=proof
 for name in m['source_SHA256']:
  if name.endswith('.py'):q['stage']='source_AST:'+name;ast.parse((D/name).read_bytes())
 q['stage']='dependency_metadata';q['dependencies']={n:{'present':importlib.util.find_spec(n)is not None,'version':importlib.metadata.version(n)}for n in ['torch','transformers','safetensors','numpy','fastapi','uvicorn','jsonschema']};gate('all_dependency_specs_present',all(x['present']for x in q['dependencies'].values()))
 q['stage']='ldd_r_execute';z=subprocess.run(['ldd','-r',str(native)],capture_output=True,text=True,timeout=60);q['ldd_r']={'returncode':z.returncode,'stdout':z.stdout,'stderr':z.stderr};gate('ldd_r_returncode',z.returncode,0);gate('ldd_r_missing_library','not found'not in z.stdout+z.stderr);gate('ldd_r_undefined_symbol','undefined symbol:'not in z.stdout+z.stderr)
 q['stage']='private_overlay_create';src=BASE/'python/infinilm';dst=B/'python/infinilm';dst.mkdir(parents=True);basepins={}
 for p in sorted(src.rglob('*')):
  rel=p.relative_to(src)
  if '__pycache__'in rel.parts or p.suffix=='.pyc':continue
  target=dst/rel
  if p.is_dir():target.mkdir(exist_ok=True)
  elif p.is_file():
   target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(p)
   if p.suffix=='.py'or'.so'in p.name:basepins[str(p)]=sha(p)
 overlays={}
 for rel,h in m['overlay_SHA256'].items():
  p=D/'overlay'/rel;gate('overlay_source:'+rel,sha(p),h);target=B/rel;target.parent.mkdir(parents=True,exist_ok=True)
  if target.is_symlink():target.unlink()
  gate('overlay_target_new:'+rel,not target.exists());target.write_bytes(p.read_bytes());overlays[str(target)]=sha(target)
 gate('no_Torch_or_Native_import','torch'not in sys.modules and '_infinilm'not in sys.modules)
 q.update(completed=True,Native_path=str(native),Native_SHA256=sha(native),model_directory=str(MODEL),private_overlay=str(B/'python'),baseline_files_SHA256=basepins,overlay_files_SHA256=overlays,normal_CPU_exit=True,stage='completed')
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q['elapsed_s']=time.monotonic()-start
if B.exists():(B/'CPU-preflight.json').write_text(json.dumps(q,indent=2)+'\n')
print(json.dumps({k:q.get(k)for k in ['completed','actual_step','CPU_only','static_only','stage','error','traceback','ldd_r']}));assert q['completed'],q.get('stage')
