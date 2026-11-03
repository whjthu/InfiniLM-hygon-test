"""One bounded actual Python entry closure. No compiler/linker/Native dlopen."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,time
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/Kimi-TP8PP4-ordinary-r410-runtime-CPU-v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 assert 'torch'not in sys.modules and not os.environ.get('LD_PRELOAD')
 mf=D/'SOURCE_MANIFEST.json';m=json.loads(mf.read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 assert not B.exists();B.mkdir();started=time.monotonic()
 q=dict(completed=False,passed=False,CPU_only=True,GPU=0,weights=0,Native_dlopen=False,Runtime_created=False,Model_qualified=False,new_HOST_TUs=0,new_GPU_TUs=0,source_manifest_SHA256=sha(mf),actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
 try:
  for p in D.glob('*.py'):ast.parse(p.read_bytes())
  p=subprocess.run([sys.executable,'-B',str(D/'check_entry_CPU.py')],capture_output=True,text=True,timeout=250)
  log=B/'actual-main.log';log.write_text(p.stdout+p.stderr);q['actual_main_log_SHA256']=sha(log);q['actual_main_rc']=p.returncode
  if p.returncode:raise RuntimeError('Actual Python consumer closure failed; see '+str(log))
  suite=json.loads(p.stdout);assert suite['completed']and suite['passed']and suite['actual_source_manifest_SHA256']==sha(mf)
  pins=json.loads((D/'bindings.json').read_bytes());q.update(completed=True,passed=True,whole_actual_Python_entry=suite,candidate_receipt_SHA256=pins['candidate_receipt_SHA256'],Native_SHA256=pins['candidate_Native_SHA256'],qualification='Real main/preengine+4 TCP leaders+typed recording CABI+same loader/head/key/fixture+codec/parser and finally consumers. No actual Native/device/weights, quality or Model performance qualification.')
 except BaseException as e:q['error']=repr(e);raise
 finally:
  q['elapsed_s']=time.monotonic()-started;(B/'CPU-result.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','elapsed_s','error','Native_SHA256']}))
if __name__=='__main__':main()
