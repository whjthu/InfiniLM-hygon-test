"""Unique CPU-only launcher. Root stages files separately before invoking this."""
import pathlib,subprocess,json,os,datetime,hashlib
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
NAME='Kimi-TP32-I96-CPU-r120c'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 entry=R/'incoming/Kimi-TP32-I96-MoE-r120/CPU_inside_r120c.sh'
 note=R/'incoming/Kimi-TP32-I96-MoE-r120/CPU-entry-manifest-r120c.json'
 m=json.loads(note.read_bytes());assert sha(entry)==m['entry_SHA256'];assert sha(entry.parent/'CPU_build_r120c.py')==m['CPU_build_SHA256']
 assert sha(R/'incoming/Kimi-TP32-I96-MoE-r120/source-manifest-r120.json')==m['frozen_r120_source_manifest_SHA256']
 receipt_path=R/'results'/f'{NAME}-launch.json';assert not receipt_path.exists(),'unique launcher already used'
 # No inventory, upload, model import, CUDA call, GPU allocation or retry.
 env=dict(os.environ)
 for k in list(env):
  if k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_')) and k not in ['SLURM_CONF','SLURM_CONF_SERVER']:env.pop(k)
 for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:env.pop(k,None)
 env['WAVE_INSIDE_R2323']='Kimi-TP32-I96-MoE-r120/CPU_inside_r120c.sh'
 for k in ['APPTAINERENV_CUDA_VISIBLE_DEVICES','APPTAINERENV_HIP_VISIBLE_DEVICES','APPTAINERENV_ROCR_VISIBLE_DEVICES']:env[k]=''
 args=['srun','--jobid=910407','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c2','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:15:00','--job-name='+NAME,'--output='+str(R/'logs'/(NAME+'-%J.log')),'bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 # Exclusive creation prevents duplicate dispatch; a failed launch is not retried.
 with receipt_path.open('x',encoding='utf-8') as receipt:
  with (R/'logs'/(NAME+'-client.log')).open('x') as log:
   p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  data={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':p.pid,'args':args,'WAVE_INSIDE_R2323':env['WAVE_INSIDE_R2323'],'GPU':False,'GPU_compile':False,'HOST_TUs':1,'upload':False,'native_dlopen':False,'root_claim_commit':'5e7da03','entry_SHA256':sha(entry),'entry_note_SHA256':sha(note),'completed':False,'requires_actual_terminal_receipt':True}
  receipt.write(json.dumps(data,indent=2)+'\n')
 print(json.dumps(data))
if __name__=='__main__':main()
