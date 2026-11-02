from pathlib import Path
import hashlib,json,sys,ast,tempfile,subprocess
from read_CPU_lifetime_v2 import read
import launch_once_lifetime_v2 as launch
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(observed):
 cpu=read(observed);q=json.loads(Path(observed).read_bytes())['receipt']['data']
 overlays=['GPU_run_lifetime_v2.py','GPU_inside_lifetime_v2.sh','controller_lifetime_v2.py','resource_probe_lifetime_v2.py','launch_once_lifetime_v2.py']
 for n in overlays:assert (D/n).read_bytes()==(D/'lifetime-v2'/n).read_bytes()
 for p in D.glob('*.py'):ast.parse(p.read_text());compile(p.read_bytes(),str(p),'exec')
 calls=[]
 def recorded(args,**kw):
  calls.append(args);p=subprocess.Popen([sys.executable,'-c','pass'],stdin=subprocess.DEVNULL,stdout=kw['stdout'],stderr=kw['stderr'],start_new_session=True);p.wait(timeout=10);return p
 with tempfile.TemporaryDirectory(prefix='r398-life-launch-')as temp:
  root=Path(temp);(root/'logs').mkdir();p,args=launch.start_keeper(root,root/'go.json',popen=recorded);assert p.returncode==0 and len(calls)==1 and args==launch.keeper_command(root,root/'go.json')
  assert str(next(a for a in args if a.endswith('.py'))).endswith('/controller_lifetime_v2.py')or str(next(a for a in args if a.endswith('.py'))).endswith('\\controller_lifetime_v2.py')
  try:launch.start_keeper(root,root/'go.json',popen=recorded)
  except FileExistsError:pass
  else:raise AssertionError('duplicate launch')
 result=dict(passed=True,actual_CPU798_reader=cpu,actual_keeper_Popen=True,duplicate_rejected=True,no_Slurm=True,GPU=0,Native=False,old_CPU796_and_789_retained=True)
 (D/'GPU-launch-lifetime-v2-local-CPU-result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
 old=json.loads((D/'GPU_SOURCE_MANIFEST.json').read_bytes());files=list(old['files_SHA256'])+overlays+['read_CPU_lifetime_v2.py','freeze_GPU_lifetime_v2.py','GPU-launch-lifetime-v2-local-CPU-result.json']
 m=dict(old);m.update(files_SHA256={n:sha(D/n)for n in files},CPU_receipt='/refactor/candidates/GLM-V-on-demand-r398-lifetime-v2-CPU/CPU-build.json',CPU_receipt_SHA256=cpu['receipt_SHA256'],library=q['library'],library_SHA256=cpu['library_SHA256'],CPU_source_manifest_SHA256=cpu['source_manifest_SHA256'],GPU_GO=True,prepared=False,final_root_read_pending=False,Model_GO=False,lifetime_fix=dict(CPU='910407.798',source_publication='a5bad50e5102a74402b6975e28eae9a18723bfb5',CXX_edits=3,added_sync=0,math_changed=False,Runtime_exception_guarantee_assumed=False,HOST_selection_owner='Work member survives destructor body drain during unwind'),execution_entry='launch_once_lifetime_v2.py',controller_entry='controller_lifetime_v2.py',GPU_entry='GPU_run_lifetime_v2.py',cardcost_authority='ActualGPU Slurm Start/End/Elapsed*8; owner_service_residency_s is originalepoch duration including preflight/queue, never exact allocatedcardseconds')
 p=D/'GPU_SOURCE_MANIFEST_lifetime_v2.json';p.write_text(json.dumps(m,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(GPU_manifest_SHA256=sha(p),files=len(files),GPU_GO=True,Model_GO=False)))
if __name__=='__main__':main(sys.argv[1])
