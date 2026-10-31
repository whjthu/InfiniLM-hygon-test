"""Read-only original KFD probe plus the actual Deep HTTP port; no GPU import."""
import pathlib,hashlib,runpy,io,contextlib,json,socket,os
p=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/verify_gpu_idle_r5738.py');assert hashlib.sha256(p.read_bytes()).hexdigest()=='45b7d14902965ca8569000a8c20882d589c7b2468f365e96d0cb552cf6f57b53';out=io.StringIO()
with contextlib.redirect_stdout(out):runpy.run_path(str(p),run_name='__main__')
rows=[json.loads(v)for v in out.getvalue().splitlines()if v.startswith('{')];assert len(rows)==1;q=rows[0];assert q['node']=='a02r4n01'and not q['kfd_pids']and q['CUDA_not_initialized_by_probe']
with socket.socket(socket.AF_INET,socket.SOCK_STREAM)as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('0.0.0.0',43882))
q.update(actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],HTTP43882_bindable=True,GPU_imported=False);print(json.dumps(q))
