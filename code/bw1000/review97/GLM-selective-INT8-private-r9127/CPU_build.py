import pathlib,json,hashlib,subprocess,os,importlib.util,struct,re,traceback
r=pathlib.Path('/refactor');inc=pathlib.Path(__file__).parent;b=r/'candidates/GLM-selective-INT8-r9127';b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU_initialized':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
try:
 import torch,safetensors,numpy
 q['imports']={n:importlib.util.find_spec(n).origin for n in ['torch','safetensors','numpy']};assert not torch.cuda.is_initialized()
 model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');cfg=json.loads((model/'config.json').read_text());idx=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];q['identity']={'path':str(model),'resolve':str(model.resolve()),'config':cfg,'config_SHA256':sha(model/'config.json'),'index_SHA256':sha(model/'model.safetensors.index.json'),'README':(model/'README.md').read_text()[:10000]if(model/'README.md').exists()else None,'metadata_files':[p.name for p in model.iterdir()if p.name.startswith('.') or p.name in ['README.md','configuration.json']]}
 assert cfg['num_hidden_layers']==78
 keys=sorted(k for k in idx if '.experts.0.'in k);layer=min(int(re.search(r'layers[.](\d+)',k).group(1))for k in keys);selected=[k for k in keys if '.layers.'+str(layer)+'.'in k];q['selected_headers']={}
 for fn in sorted(set(idx[k]for k in selected)):
  with(model/fn).open('rb')as f:hlen=struct.unpack('<Q',f.read(8))[0];raw=f.read(hlen)
  header=json.loads(raw)
  for k in selected:
   if idx[k]==fn:q['selected_headers'][k]={'file':fn,'file_size':(model/fn).stat().st_size,'header_SHA256':hashlib.sha256(raw).hexdigest(),'tensor':header.get(k)}
 lm=r/'candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM';sdk=r/'install/hygon-glm-5f91ad7-r3672';runtime=lm/'python/infinilm/_lib/libinfinilm_runtime.so';q['dependencies']={str(p):sha(p)for p in [runtime,sdk/'lib/libinfiniops.so',sdk/'lib/libinfinirt.so']}
 assert sha(runtime)=='dc8542f14f70b85c67c227532f61756d432bbfee02308d373ff7ae5b9a540c36';assert sha(sdk/'lib/libinfiniops.so')=='0bd37990340f39bbcaf185abbc990ebeb9f56ea1b9992327eda96d627446a863'
 src=b/'i8_bridge_r9127.cpp';src.write_bytes((inc/src.name).read_bytes());lib=b/'libGLM_i8_r9127.so';cmd=['g++','-std=c++17','-O2','-shared','-fPIC','-pthread','-I'+str(lm/'csrc'),'-I'+str(sdk/'include'),'-I/opt/dtk/cuda/include','-I/opt/dtk/cuda/cuda/include',str(src),str(runtime),'-L'+str(sdk/'lib'),'-linfiniops','-linfinirt','-L/opt/dtk/cuda/cuda/lib64','-L/opt/dtk/cuda/lib64','-lcudart','-Wl,-rpath,'+str(runtime.parent),'-Wl,-rpath,'+str(sdk/'lib'),'-o',str(lib)];q['command']=cmd;proc=subprocess.run(cmd,capture_output=True,text=True,timeout=150);q['compile_output']=proc.stdout+proc.stderr;assert proc.returncode==0,q['compile_output'][-6000:]
 env=os.environ.copy();env['LD_LIBRARY_PATH']=str(sdk/'lib')+':'+str(runtime.parent)+':'+str(pathlib.Path(torch.__file__).parent/'lib')+':'+env.get('LD_LIBRARY_PATH','');dep=subprocess.run(['ldd',str(lib)],env=env,capture_output=True,text=True);q['ldd']=dep.stdout+dep.stderr;assert dep.returncode==0 and 'not found'not in q['ldd'];q.update(completed=True,library=str(lib),library_SHA256=sha(lib),source_SHA256=sha(src),conversion_CPU=__import__('conversion').selftest())
except BaseException as e:q['error']=repr(e);q['traceback']=traceback.format_exc()
(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps({k:q.get(k)for k in ['completed','producer','error','library','library_SHA256']}));assert q['completed']
