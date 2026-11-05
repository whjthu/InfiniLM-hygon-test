"""Only link existing1060 objects; never compile, import/dlopen Runtime orSDK."""
from pathlib import Path
import json,hashlib,subprocess,os,time,runpy,re,traceback,shlex
D=Path(__file__).resolve().parent;P=D.parent;O=Path('/refactor/candidates/GLM-global-TP16-component-r470-CPU-link-v2');H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.monotonic();assert not O.exists();O.mkdir();q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=False,Runtime=False,Native_dlopen=False,Common_dlopen=False,Model=False,weights=False,new_HOST=0,new_GPU=0,reused_completed_HOST=2,commands=[])
 def save():(O/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 def run(a,name,stdin=None):
  left=145-(time.monotonic()-start);assert left>5;q['commands'].append(dict(name=name,argv=a));save();z=subprocess.run(a,input=stdin,capture_output=True,text=True,timeout=left-3);(O/(name+'.stdout')).write_text(z.stdout);(O/(name+'.stderr')).write_text(z.stderr);assert z.returncode==0,(name,z.returncode,z.stderr[-2500:]);return z.stdout
 try:
  assert os.environ['SLURM_JOB_ID']=='910407'and all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);assert H(D/'CPU_RESOURCE_CLAIM.json')==os.environ['GLM470LINK_CLAIM_SHA'];claim=json.loads((D/'CPU_RESOURCE_CLAIM.json').read_bytes());assert claim['CPU_GO']and not claim['GPU_GO']and H(D/'SOURCE_MANIFEST.json')==claim['source_manifest_SHA256'];mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(H(D/n)==h for n,h in mf['files_SHA256'].items());reuse=json.loads((D/'REUSE.json').read_bytes());assert H(P/'SOURCE_MANIFEST.json')==reuse['source_manifest_SHA256']and H(P/'CPU_RESOURCE_CLAIM.json')==reuse['source_claim_SHA256'];oldmf=json.loads((P/'SOURCE_MANIFEST.json').read_bytes());assert all(H(P/n)==h for n,h in oldmf['files_SHA256'].items());assert H(reuse['original_receipt_path'])==reuse['original_receipt_SHA256'];r=json.loads(Path(reuse['original_receipt_path']).read_bytes());assert r['completed']and not r['passed']and r['actual_CPU']=='910407.1060'and 'component-link'in r['error'];assert [c['name']for c in r['commands']]==['component-M','component-compile','check_geometry-M','check_geometry-compile','component-link']
  for p,h in reuse['objects'].items():assert H(p)==h,p
  for stem,m in r['fresh_dependency_SHA256'].items():
   md=Path(reuse['original_receipt_path']).parent/(stem+'.d');actual=set(shlex.split(md.read_text().replace(chr(92)+chr(10),' ').partition(':')[2]));assert actual==set(m)
   for p,h in m.items():assert H(p)==h,p
  pin=json.loads((P/'BUILD_PIN.json').read_bytes());libs=dict(pin['libraries']);libs[reuse['Common']]=reuse['Common_SHA256']
  for p,h in libs.items():assert H(p)==h,p
  for p,v in json.loads((P/'TOOLCHAIN.json').read_bytes())['tools'].items():assert str(Path(p).resolve())==v['canonical']and H(p)==v['sha256']
  library=O/'libGLM_global_TP16_component_r470.so';a=list(r['commands'][-1]['argv']);a[a.index('-o')+1]=str(library);at=a.index('-linfinilm_runtime');common=str(Path(reuse['Common']).parent);a[at:at]=['-L'+common,'-Wl,-rpath,'+common,'-Wl,-rpath-link,'+common,'-l:_infinilm.cpython-310-x86_64-linux-gnu.so'];run(a,'component-link');elf=run(['/usr/bin/readelf','-d',str(library)],'component-ELF');nm=run(['/usr/bin/nm','-D','--defined-only',str(library)],'component-CABI');assert all('glm_global_tp16_'+n+'_r470'in nm for n in ['error','id_size','id','create','geometry','probe','close'])
  ldd_out=run(['/usr/bin/ldd','-r',str(library)],'component-loader-metadata');ldd_err=(O/'component-loader-metadata.stderr').read_text();ldd=ldd_out+'\n'+ldd_err;assert not re.search('not found|undefined symbol',ldd,re.I);resolved={};entries=[]
  for line in ldd.splitlines():
   m=re.match(r'\s*(\S+)\s+=>\s+(/\S+)\s+\(',line)
   if m:resolved[m[1]]=m[2];entries.append((m[1],m[2]))
  for p,h in libs.items():assert sum(n==Path(p).name for n,_ in entries)==1 and H(resolved[Path(p).name])==h,(p,resolved.get(Path(p).name))
  commondyn=run(['/usr/bin/readelf','-d',reuse['Common']],'Common-ELF');assert re.findall(r'\(NEEDED\).*?\[([^\]]+)\]',commondyn)==reuse['Common_ELF']['NEEDED']and re.findall(r'\(RUNPATH\).*?\[([^\]]+)\]',commondyn)==reuse['Common_ELF']['RUNPATH']and not re.search(r'\(SONAME\)',commondyn)
  exe=O/'geometry_contract';obj=next(p for p in reuse['objects']if p.endswith('/check_geometry.o'));run(['/usr/bin/g++',obj,'-o',str(exe)],'geometry-link');dyn=run(['/usr/bin/readelf','-d',str(exe)],'geometry-ELF');assert not any(n in dyn for n in ['infinilm','infiniops','infinirt','infiniccl','cuda','hip','torch','python']);cfg=Path('/old/models/ZhipuAI-GLM-5.3/config.json').read_bytes();assert hashlib.sha256(cfg).hexdigest()=='3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69';geom=json.loads(run([str(exe)],'geometry-contract',cfg.decode()));assert geom['passed'];typed=runpy.run_path(str(P/'check_binding_CPU.py'))['run']();assert typed['passed'];q.update(passed=True,source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),claim_SHA256=H(D/'CPU_RESOURCE_CLAIM.json'),original1060_FAIL_preserved=True,original1060_receipt_SHA256=reuse['original_receipt_SHA256'],reused_objects_SHA256=reuse['objects'],fresh_M_MD_revalidated=True,library=str(library),library_SHA256=H(library),verified_libraries=libs,actual_loader_resolution=resolved,loader_stdout_SHA256=H(O/'component-loader-metadata.stdout'),loader_stderr_SHA256=H(O/'component-loader-metadata.stderr'),Common_ELF=reuse['Common_ELF'],geometry=geom,typed=typed,geometry_executable_SHA256=H(exe),component_executed=False,geometry_SDK_linked=False,ELF=elf)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:q.update(completed=True,seconds=time.monotonic()-start);save()
 print(json.dumps({k:q.get(k)for k in ['passed','actual_CPU','seconds','error','library_SHA256']}))
if __name__=='__main__':main()
