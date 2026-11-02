"""Independent terminal PID/KFD/port read only; no imports that initialize GPUs."""
import pathlib,json,hashlib,runpy,contextlib,io,subprocess,socket,os
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
p=R/'incoming/verify_gpu_idle_r5738.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='45b7d14902965ca8569000a8c20882d589c7b2468f365e96d0cb552cf6f57b53'
output=io.StringIO()
with contextlib.redirect_stdout(output):runpy.run_path(str(p),run_name='__main__')
q=json.loads(next(l for l in output.getvalue().splitlines()if l.startswith('{')))
assert q['node']=='a02r4n01'and q['kfd_pids']==[]
rows=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()
owned=[line for line in rows if 'Deep-forward-position-proof-r387/model-v1/owner/'in line]
assert not owned,owned
with socket.socket(socket.AF_INET,socket.SOCK_STREAM)as s:
 s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('0.0.0.0',43882))
q.update(actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
 owned_Model_PIDs=owned,HTTP43882_bindable=True,GPU_imported=False,new_GPU_qualification=False)
print(json.dumps(q))
