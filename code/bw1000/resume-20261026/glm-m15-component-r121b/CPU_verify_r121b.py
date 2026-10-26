"""Metadata/preprocessor-only repair of438; never recompiles or dlopens DSO."""
import pathlib,json,hashlib,subprocess,shlex,time,os
r=pathlib.Path('/refactor');inc=pathlib.Path(__file__).resolve().parent;old=r/'candidates/GLM-M15-component-r121';out=r/'candidates/GLM-M15-component-r121b';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((inc/'manifest-r121b.json').read_bytes());assert not out.exists();out.mkdir();q={'completed':False,'CPU_only':True,'GPU_initialized':False,'dlopen':False,'GPU_TUs_compiled':0,'HOST_TUs_compiled':0,'commands':[],'reused_compiled_HOST_TUs':2};start=time.monotonic()
def run(a,name):
 q['commands'].append(a);z=subprocess.run(a,text=True,capture_output=True,timeout=90);(out/(name+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(name,z.stderr[-2000:]);return z.stdout
def deps(s):return set(shlex.split(s.replace('\\\n',' ').split(':',1)[1]))
try:
 for n,h in m['source_SHA256'].items():assert sha(inc/n)==h,n
 for n,h in m['original_artifact_SHA256'].items():assert sha(old/n)==h,n
 assert sha(pathlib.Path(m['torch_python_path']))==m['torch_python_SHA256']
 oldmanifest=r/'incoming/GLM-M15-component-r121/manifest-r121.json';assert sha(oldmanifest)==m['original_manifest_SHA256'];original=json.loads(oldmanifest.read_bytes())
 for n,h in original['source_SHA256'].items():assert sha(oldmanifest.parent/n)==h,n
 for n,h in original['actual_binding_SHA256'].items():assert sha(r/n)==h,n
 fail=json.loads((old/'CPU-build.json').read_bytes());assert not fail['completed']and fail['CPU_only']and not fail['GPU_initialized']and not fail['dlopen']and fail['CPU_mock_contract']['passed'];assert fail['commands'][-1][0]=='ldd'
 compiled=[];dependency_hashes={}
 for command in fail['commands']:
  if '-c'not in command:continue
  source=pathlib.Path(command[command.index('-c')+1]);obj=pathlib.Path(command[command.index('-o')+1]);md=pathlib.Path(command[command.index('-MF')+1]);flags=command[:command.index('-MD')];fresh=run(flags+['-M',str(source)],source.name+'.fresh-M');assert deps(fresh)==deps(md.read_text()),source.name
  for p in deps(fresh):dependency_hashes[p]=sha(p)
  compiled.append(dict(source=str(source),source_SHA256=sha(source),object=str(obj),object_SHA256=sha(obj),original_MD_SHA256=sha(md),fresh_M_equal_MD=True))
 assert len(compiled)==2
 lib=old/'libGLM_M15_component_r121.so';exports=run(['nm','-D','--defined-only',str(lib)],'exports');assert all(s in exports for s in ['glm_m15_component_forward_r121','glm_m15_component_destroy_r121','glm_fp8_cache_run_r119']);dynamic=run(['readelf','-d',str(lib)],'dynamic');assert dynamic==(old/'dynamic.log').read_text();ldd=run(['ldd',str(lib)],'ldd');assert 'not found'not in ldd;assert 'libtorch_python.so => '+m['torch_python_path']in ldd
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),compiled=compiled,dependency_SHA256=dependency_hashes,CPU_mock_contract=fail['CPU_mock_contract'],actual_binding_SHA256=original['actual_binding_SHA256'],original_failure_receipt_SHA256=sha(old/'CPU-build.json'),original_failure_preserved=True,original_manifest_SHA256=sha(oldmanifest),repair_manifest_SHA256=sha(inc/'manifest-r121b.json'),environment_fix='Addexact/usr/local/lib/python3.10/dist-packages/torch/lib toLD_LIBRARY_PATH;noTorchimport',new_dependency_SHA256={m['torch_python_path']:m['torch_python_SHA256']},ELF_dynamic=dynamic,ldd=ldd,ABI=1,normal_CPU_exit=True,no_recompile_or_relink=True)
except BaseException as e:q['error']=repr(e)
q['seconds']=time.monotonic()-start;q['producer']=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','');(out/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
