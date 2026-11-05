"""Exactly2 HOST TUs: SDK component never executes; metadata executable has no SDK."""
from pathlib import Path
import hashlib,json,os,subprocess,time,traceback,shlex,re,runpy
D=Path(__file__).resolve().parent;O=Path('/refactor/candidates/GLM-global-TP16-component-r470-CPU-v1');H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def deps(text):return set(shlex.split(text.replace(chr(92)+chr(10),' ').partition(':')[2]))
def main():
 start=time.monotonic();claim=json.loads((D/'CPU_RESOURCE_CLAIM.json').read_bytes());assert H(D/'CPU_RESOURCE_CLAIM.json')==os.environ['GLM470_CLAIM_SHA'];assert claim['CPU_GO']and not claim['GPU_GO']and os.environ['SLURM_JOB_ID']=='910407';assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);assert not O.exists();O.mkdir();q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=False,Runtime=False,Native_dlopen=False,Model=False,weights=False,new_HOST=0,new_GPU=0,commands=[])
 def save():(O/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 def run(a,name,stdin=None):
  left=265-(time.monotonic()-start);assert left>5;q['commands'].append(dict(name=name,argv=a));save()
  z=subprocess.run(a,input=stdin,capture_output=True,text=True,timeout=left-3);(O/(name+'.stdout')).write_text(z.stdout);(O/(name+'.stderr')).write_text(z.stderr);assert z.returncode==0,(name,z.returncode,z.stderr[-2500:]);return z.stdout
 try:
  assert H(D/'SOURCE_MANIFEST.json')==claim['source_manifest_SHA256'];mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(H(D/n)==h for n,h in mf['files_SHA256'].items());pin=json.loads((D/'BUILD_PIN.json').read_bytes());allowed=pin['dependency_SHA256'];tools=json.loads((D/'TOOLCHAIN.json').read_bytes())['tools']
  for path,v in tools.items():assert str(Path(path).resolve())==v['canonical']and H(path)==v['sha256']
  for path,h in pin['libraries'].items():assert H(path)==h,path
  r=pin['original_glm952_receipt'];assert H(r['path'])==r['SHA256'];q['verified_libraries']=pin['libraries'];flags=pin['compile_flags'];outputs=[];fresh={}
  for stem in ['component','check_geometry']:
   source=D/(stem+'.cpp');obj=O/(stem+'.o');md=O/(stem+'.d');text=run(['/usr/bin/g++',*flags,'-I'+str(D),'-M',str(source)],stem+'-M');before={p:H(p)for p in deps(text)};fresh[stem]=before;q['fresh_dependency_SHA256']=fresh;save()
   for p,h in before.items():assert (Path(p).parent==D and H(p)==h)or allowed.get(p)==h,('Unpinned dependency',p,h)
   run(['/usr/bin/g++',*flags,'-I'+str(D),'-MD','-MF',str(md),'-c',str(source),'-o',str(obj)],stem+'-compile');assert deps(md.read_text())==set(before)and all(H(p)==h for p,h in before.items());outputs.append(obj)
  runtime='/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/infinilm/_lib';sdk='/refactor/install/hygon-glm-5f91ad7-r3672/lib';dirs=[runtime,sdk,'/usr/local/lib/python3.10/dist-packages/torch/lib','/usr/local/lib/python3.10/dist-packages/torch.libs','/opt/dtk/cuda/cuda/lib64','/opt/dtk/lib','/usr/local/lib'];link=[x for d in dirs for x in ['-L'+d,'-Wl,-rpath,'+d,'-Wl,-rpath-link,'+d]];lib=O/'libGLM_global_TP16_component_r470.so'
  run(['/usr/bin/g++','-shared','-pthread',str(outputs[0]),*link,'-linfinilm_runtime','-linfiniops','-linfiniccl','-linfinirt','-lcudart','-lamdhip64','-lpython3.10','-Wl,--no-undefined','-o',str(lib)],'component-link')
  elf=run(['/usr/bin/readelf','-d',str(lib)],'component-ELF');nm=run(['/usr/bin/nm','-D','--defined-only',str(lib)],'component-CABI');expected={'glm_global_tp16_'+n+'_r470'for n in ['error','id_size','id','create','geometry','probe','close']};assert expected<={x.split()[-1]for x in nm.splitlines()}
  exe=O/'geometry_contract';run(['/usr/bin/g++',str(outputs[1]),'-o',str(exe)],'geometry-link');dynamic=run(['/usr/bin/readelf','-d',str(exe)],'geometry-ELF');assert not any(n in dynamic for n in ['infinilm','infiniops','infinirt','infiniccl','cuda','hip','torch','python']);config_path=Path('/old/models/ZhipuAI-GLM-5.3/config.json');config_raw=config_path.read_bytes();assert hashlib.sha256(config_raw).hexdigest()=='3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69';geom=json.loads(run([str(exe)],'geometry-contract',config_raw.decode()));assert geom['passed'];typed=runpy.run_path(str(D/'check_binding_CPU.py'))['run']();assert typed['passed'];q.update(passed=True,new_HOST=2,new_GPU=0,source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),claim_SHA256=H(D/'CPU_RESOURCE_CLAIM.json'),library=str(lib),library_SHA256=H(lib),geometry_executable=str(exe),geometry_SHA256=H(exe),geometry=geom,typed=typed,fresh_M_equals_MD=True,component_executed=False,geometry_SDK_linked=False,objects_SHA256={str(p):H(p)for p in outputs},ELF=elf)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:q.update(completed=True,seconds=time.monotonic()-start);save()
 print(json.dumps({k:q.get(k)for k in ['passed','actual_CPU','seconds','error','library_SHA256']}))
if __name__=='__main__':main()
