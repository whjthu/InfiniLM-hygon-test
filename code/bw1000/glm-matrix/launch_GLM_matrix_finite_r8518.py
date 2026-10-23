"""Execute only after root explicit GO; no automatic retry."""
import pathlib,subprocess,json,hashlib,datetime,sys
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sys.argv[1:] == ['ROOT_GO_R8518']
mp=i/'GLM-matrix-GPU-manifest-r8518.json';m=json.loads(mp.read_text())
for n,h in m['sources_SHA256'].items():assert sha(i/n)==h,n
lib=r/'candidates/glm-matrix-proof-r8509/libglm_matrix_r8514.so';assert sha(lib)==m['DSO_SHA256']
out=r/'glm-matrix-intent-r8518.json';assert not out.exists()and not(r/'glm-matrix-terminal-r8518.json').exists()
snap=subprocess.check_output(['sacct','-S','2026-10-24','-j','910379','-n','-P','-o','JobID,JobName%80,State,ExitCode,Elapsed,NodeList%120'],text=True)
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
for line in snap.splitlines():
 f=line.split('|')
 if'.'in f[0]and f[0].split('.')[-1].isdigit()and'a14r2n06'in f[-1]:assert f[2].split()[0].rstrip('+')in terminal,line
idle=subprocess.run(['srun','--jobid=910379','--overlap','--exact','-N1','-n1','-c1','-w','a14r2n06','--gres=dcu:8','--mem=1000M','-t','00:02:00','--job-name=glm-matrix-idle-r8518','python3',str(i/'verify_gpu_idle_r5738.py')],capture_output=True,text=True,timeout=150);assert idle.returncode==0,(idle.stdout,idle.stderr);kd=json.loads(idle.stdout);assert kd['node']=='a14r2n06'and kd['kfd_pids']==[]
args=['srun','--jobid=910379','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c60','-w','a14r2n06','--gres=dcu:8','--mem=96000M','-t','00:20:00','--job-name=glm-matrix-finite-r8518','--output='+str(r/'logs/glm-matrix-finite-r8518-%J.log'),'env','WAVE_INSIDE_R2323=GLM_matrix_finite_r8518_inside.sh','bash',str(i/'resource_wave_local_r3097_outer.sh'),'probe']
with out.open('x')as f:json.dump({'args':args,'manifest_SHA256':sha(mp),'fresh_KFD':kd,'fresh_Slurm':snap},f,indent=2)
with(r/'logs/glm-matrix-controller-r8518.log').open('x')as f:p=subprocess.Popen(['python3',str(i/'controller_GLM_matrix_finite_r8518.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'controller_pid':p.pid,'intent':str(out),'scope':'Single finite component weightload, noModel/noauto-retry'}))
