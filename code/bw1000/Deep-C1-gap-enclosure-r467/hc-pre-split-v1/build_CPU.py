"""One private HC HOST object, 49 exact7f objects reused; no dlopen/Runtime."""
from pathlib import Path
import hashlib,json,os,re,shlex,subprocess,time,traceback
from derive import derive
D=Path(__file__).resolve().parent;R=Path('/refactor');O=R/'candidates/Deep-HCpre-phase-r467-CPU-v1';H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def deps(p):return set(shlex.split(Path(p).read_text().replace('\\\n',' ').partition(': ')[2]))
def main():
 assert not O.exists(),'Reconcile originalCPU; never repeat';O.mkdir();start=time.monotonic();q=dict(completed=False,passed=False,GPU=False,Native_dlopen=False,Runtime=False,weights=False,Model=False,commands=[],actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
 def save():(O/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n')
 def run(args,name):
  left=250-(time.monotonic()-start);assert left>5;q['commands'].append(dict(name=name,argv=args));save();z=subprocess.run(args,capture_output=True,text=True,timeout=left-3);(O/(name+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(name,z.returncode,z.stderr[-2000:]);return z.stdout
 try:
  assert all(os.environ.get(k)=='-1'for k in['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);assert os.environ['SLURM_JOB_ID']=='910407';m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert H(D/'SOURCE_MANIFEST.json')==os.environ['DEEP467_HC_SOURCE_SHA'];assert all(H(D/n)==h for n,h in m['files_SHA256'].items())
  pin=json.loads((D/'BUILD_PIN.json').read_bytes());a=R/'candidates/Deep-HC-fullpre-Native-r377-cpu-v1/CPU-build.json';p=R/'candidates/Deep-C4-current-host-split-CPU-v1/CPU-build.json';c=R/'candidates/Deep-C4-compact-bank-CPU-private-state-v2/CPU-build.json'
  assert H(a)==pin['a314_receipt']and H(p)==pin['split_receipt']and H(c)==pin['compact_receipt'];base=json.loads(a.read_bytes());prior=json.loads(p.read_bytes());compact=json.loads(c.read_bytes());old=Path(compact['library']);assert H(old)==pin['7f'];assert compact['retained_objects']==49 and compact['replaced_objects']==1
  commands={r['label']:r['argv']for r in prior['commands']};link=list(commands['link']);objects=dict(base['all50_objects_SHA256'])
  for i,replaced in enumerate(prior['replaced']):
   obj=commands['compile-'+str(i)][commands['compile-'+str(i)].index('-o')+1];objects.pop(replaced['parent']);objects[obj]=replaced['SHA256']
  oldcsa=commands['compile-0'][commands['compile-0'].index('-o')+1];newcsa=str(c.parent/Path(oldcsa).name);objects.pop(oldcsa);objects[newcsa]=compact['object_sha256'];link[link.index(oldcsa)]=newcsa;assert len(objects)==50 and set(objects)=={x for x in link if x.endswith('.o')}
  for path,h in objects.items():assert H(path)==h,path
  hc=Path(base['new_HC_object']);original=Path('/refactor/incoming/Deep-HC-fullpre-Native-r377/cpu-v1/deepseek_v4_hyper_connection_native_r377.cpp');assert H(original)==pin['originalHC'];assert H(hc)==pin['HC_object'];assert H(hc.with_suffix('.d'))==pin['HC_MD']
  code=derive(original.read_bytes());assert hashlib.sha256(code).hexdigest()==m['files_SHA256']['derived_HC.cpp'];source=O/'HC.cpp';source.write_bytes(code);obj=O/'HC.o';md=O/'HC.d'
  cmd=list(next(x for x in base['commands']if '-c'in x and str(original)in x));cmd[1:1]=['-I'+str(D),'-I/refactor/candidates/deep-owner-device-r8918/src']
  for oldp,newp in[(original,source),(hc,obj),(hc.with_suffix('.d'),md)]:assert cmd.count(str(oldp))==1;cmd[cmd.index(str(oldp))]=str(newp)
  flags=[x for i,x in enumerate(cmd)if i not in [0,cmd.index('-c'),cmd.index('-c')+1,cmd.index('-o'),cmd.index('-o')+1,cmd.index('-MF'),cmd.index('-MF')+1]and x!='-MD'];fresh=O/'HC.M';text=run([cmd[0],*flags,'-M',str(source)],'fresh-M');fresh.write_text(text);expected=deps(fresh);before={x:H(x)for x in expected};allowed=dict(base['dependencies_SHA256']);allowed['/refactor/candidates/deep-owner-device-r8918/src/deep_owner_events_r8900.hpp']=pin['owner_header']
  # All added project and system bytes must already match original successful
  # HC dependencies, except the two explicit private source/header files.
  for path,h in before.items():
   if path==str(source):continue
   if Path(path).resolve().is_relative_to(D.resolve()):assert m['files_SHA256'].get(Path(path).name)==h,path
   else:assert allowed.get(path)==h,('Unexpected dependency, stop beforecompile',path,h)
  run(cmd,'compile');assert deps(md)==expected and all(H(p)==h for p,h in before.items())
  link[link.index(str(hc))]=str(obj);lib=O/'libDeep_HCpre_phase_r467.so';link[link.index('-o')+1]=str(lib);run(link,'link')
  sym=lambda p,n:{x.split()[-1]for x in run(['nm','-D','--defined-only',str(p)],n).splitlines()if len(x.split())>=3 and x.split()[-2]in['T','D','B','R']}
  oldsym=sym(old,'oldexports');newsym=sym(lib,'newexports');assert oldsym<=newsym and newsym-oldsym=={'deep_hc_pre_split_read_r467'}
  dynamic=lambda p,n:re.findall(r'\((NEEDED|RPATH|RUNPATH|SONAME)\).*?\[(.*?)\]',run(['readelf','-d',str(p)],n));assert dynamic(old,'oldELF')==dynamic(lib,'newELF')
  q.update(passed=True,parent_7f_SHA256=H(old),retained_objects={p:h for p,h in objects.items()if p!=str(hc)},retained_count=49,new_HOST=1,new_GPU=0,source_SHA256=H(source),object_SHA256=H(obj),MD_SHA256=H(md),fresh_M_equals_MD=True,dependency_SHA256=before,library=str(lib),library_SHA256=H(lib),source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),strip_equals_original=True,no_new_sync=True,old_strong_exports_retained=True,new_CABI=['deep_hc_pre_split_read_r467'],SDK_unchanged=True)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:q.update(completed=True,seconds=time.monotonic()-start);save()
 print(json.dumps({k:q.get(k)for k in['passed','actual_CPU','library_SHA256','error','seconds']}))
if __name__=='__main__':main()
