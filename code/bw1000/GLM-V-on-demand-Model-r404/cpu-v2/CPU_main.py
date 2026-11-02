"""One bounded changed-entry CPU closure. No compile/dlopen/weights/GPU."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,time,traceback
D=Path(__file__).resolve().parent;S=D.parent;sys.path.insert(1,str(S));R=Path('/refactor');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
B=R/'candidates/GLM-V-on-demand-Model-r404-CPU-v2';B.mkdir(exist_ok=False);start=time.monotonic();step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
q=dict(completed=False,passed=False,actual_CPU=step,GPU=0,Model=0,Native_dlopen=False,weights=False,compile=False,source_manifest_SHA256=sha(S/'SOURCE_MANIFEST.json'),CPU_source_manifest_SHA256=sha(D/'CPU_SOURCE_MANIFEST.json'))
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 assert q['source_manifest_SHA256']==os.environ['GLM_R404_CPU_PARENT_SOURCE_SHA']=='a3de385519b8e5ce238d08dcd2ae2ad153c8ac5d42fbeb925164bbb36c408477'
 assert q['CPU_source_manifest_SHA256']==os.environ['GLM_R404_CPU_V2_SOURCE_SHA']
 for n,h in json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(D/n)==h,n
 m=json.loads((S/'SOURCE_MANIFEST.json').read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(S/n)==h,n
 for p in D.rglob('*.py'):ast.parse(p.read_bytes());compile(p.read_bytes(),str(p),'exec')
 out=R/'results'/('GLM404-whole-glue-'+step);assert not out.exists()
 with(B/'actual-whole-entry.log').open('x')as log:p=subprocess.run([sys.executable,'-B','-u',str(D/'CPU_whole_glue.py')],stdout=log,stderr=subprocess.STDOUT,timeout=230)
 whole=out/'CPU-result.json';q['whole_entry_result_path']=str(whole)
 if not whole.exists():raise RuntimeError('Original child failed before producing CPUresult; inspect actual-whole-entry.log')
 body=json.loads(whole.read_bytes());q['whole_entry_result_SHA256']=sha(whole);q['whole_entry_result']=body
 assert p.returncode==0 and body['completed']and body['passed']and body['GPU']==body['Model']==body['Native_Context']==0,body.get('error')
 assert body['actual_samectx_changed_entry']['passed']and len(body['actual_samectx_changed_entry']['cases'])==8
 assert body['actual_r403_source_graph']['library_SHA256']=='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
 from check_controller_CPU import run
 q['actual_controller_main_recording']=run();assert q['actual_controller_main_recording']['passed']
 q.update(passed=True,owner_manifest_SHA256=sha(S/'owner/SOURCE_MANIFEST.json'),Native_CPU804_recompiled=False,actual_samectx_success_and_failure_glue=True,real_prepared_queue_and_fixture_encoding=True,real_HTTP_transport=False,Native_math_qualified=False,real_GPU_GO=False,real_Model_GO=False)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
trace=R/'results'/('GLM404-samectx-v2-'+step+'.json')
if trace.exists():q['phase_trace_path']=str(trace);q['phase_trace_SHA256']=sha(trace);q['phase_trace']=json.loads(trace.read_bytes())
q.update(completed=True,elapsed_s=time.monotonic()-start)
(B/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','elapsed_s','error']}));raise SystemExit(0 if q['passed']else 1)
