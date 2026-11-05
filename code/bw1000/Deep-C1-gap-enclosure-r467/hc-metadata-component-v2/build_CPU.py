"""OneHOST compile/link/staticELF. Never execute SDK/Runtime on CPU."""
from pathlib import Path
import hashlib,json,os,shlex,subprocess,time,traceback
D=Path(__file__).resolve().parent;O=Path('/refactor/candidates/Deep-HC-metadata-component-r467-CPU-v2');H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def deps(t):return set(shlex.split(t.replace('\\\n',' ').partition(': ')[2]))
def main():
 assert not O.exists(),'Reconcile original intent/receipt, never repeat';O.mkdir();start=time.monotonic();q=dict(completed=False,passed=False,GPU=False,Runtime=False,Native_dlopen=False,Model=False,weights=False,commands=[],actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
 def save():(O/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 def run(a,n):
  left=250-(time.monotonic()-start);assert left>5;q['commands'].append(dict(name=n,argv=a));save();z=subprocess.run(a,capture_output=True,text=True,timeout=left-3);(O/(n+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(n,z.returncode,z.stderr[-2500:]);return z.stdout
 try:
  assert os.environ['SLURM_JOB_ID']=='910407'and all(os.environ[k]=='-1'for k in['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);assert H(D/'SOURCE_MANIFEST.json')==os.environ['DEEP467_HC_SOURCE_SHA'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(H(D/n)==h for n,h in m['files_SHA256'].items());pin=json.loads((D/'BUILD_PIN.json').read_bytes());assert H(pin['parent_receipt_path'])==pin['parent_receipt_SHA256']and H(pin['parent_750_path'])==pin['parent_750_SHA256'];r=json.loads(Path(pin['parent_receipt_path']).read_bytes());b=json.loads(Path(pin['parent_750_path']).read_bytes());assert r['passed']and r['retained_count']==49 and r['new_HOST']==1
  for p,h in pin['library_pins'].items():assert H(p)==h,p
  allowed=dict(b['dependencies_SHA256']);allowed.update(r['dependency_SHA256']);extra=json.loads((D/'ADDITIONAL_DEPENDENCIES.json').read_bytes());assert len(extra['dependencies'])==84;allowed.update({p:v['SHA256']for p,v in extra['dependencies'].items()});caller=json.loads((D/'CALLER_DEPENDENCIES.json').read_bytes());assert caller['actual_CPU']=='910407.1046'and len(caller['new_source_dependencies'])==4 and not caller['old_dependency_mismatch'];allowed.update(caller['new_source_dependencies'])
  source=D/'probe.cpp';text=run(['g++',*pin['compile_flags'],'-M',str(source)],'fresh-M');(O/'probe.M').write_text(text);before={p:H(p)for p in deps(text)};q['fresh_dependency_SHA256']=before;save()
  for p,h in before.items():assert p==str(source)or allowed.get(p)==h,('Unpinned dependency',p,h)
  obj=O/'probe.o';md=O/'probe.d';run(['g++',*pin['compile_flags'],'-MD','-MF',str(md),'-c',str(source),'-o',str(obj)],'compile');assert deps(md.read_text())==set(before)and all(H(p)==h for p,h in before.items())
  runtime='/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM/python/infinilm/_lib';sdk='/refactor/install/hygon-deep-5f91ad7-r3672/lib';dirs=[runtime,sdk,'/opt/dtk/cuda/cuda/lib64','/opt/dtk/lib','/usr/local/lib','/usr/local/lib/python3.10/dist-packages/torch/lib','/usr/local/lib/python3.10/dist-packages/torch.libs'];exe=O/'HC_metadata_probe';args=['g++','-pthread',str(obj),'-o',str(exe)];args+=[x for d in dirs for x in['-L'+d,'-Wl,-rpath,'+d,'-Wl,-rpath-link,'+d]];args+=['-linfinilm_runtime','-linfiniops','-linfinirt','-lcudart','-lamdhip64','-lpython3.10'];run(args,'link');elf=run(['readelf','-d',str(exe)],'ELF');q.update(passed=True,source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),source_SHA256=H(source),object_SHA256=H(obj),MD_SHA256=H(md),executable=str(exe),executable_SHA256=H(exe),parent_libraries=pin['library_pins'],new_HOST=1,new_GPU=0,probe_executed=False,fresh_M_equals_MD=True,dynamic=elf)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:q.update(completed=True,seconds=time.monotonic()-start);save()
 print(json.dumps({k:q.get(k)for k in['passed','actual_CPU','seconds','error','executable_SHA256']}))
if __name__=='__main__':main()
