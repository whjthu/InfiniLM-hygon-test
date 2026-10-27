"""Root stages separately and invokes once. Link-only CPU repair; no CXX compilation or GPU."""
import pathlib,subprocess,json,os,hashlib,datetime
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');NAME='Kimi-TP32-constructor-direct-link-r120j'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 D=R/'incoming/Kimi-TP32-constructor-direct-link-r120j';m=json.loads((D/'manifest-r120j.json').read_bytes())
 for name,h in m['source_SHA256'].items():assert sha(D/name)==h,name
 receipt=R/'results'/(NAME+'-launch.json');assert not receipt.exists(),'unique launch already used'
 env=dict(os.environ)
 for k in list(env):
  if k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_')) and k not in ['SLURM_CONF','SLURM_CONF_SERVER']:env.pop(k)
 for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:env.pop(k,None)
 env['WAVE_INSIDE_R2323']='Kimi-TP32-constructor-direct-link-r120j/CPU_inside_r120j.sh'
 for k in ['APPTAINERENV_CUDA_VISIBLE_DEVICES','APPTAINERENV_HIP_VISIBLE_DEVICES','APPTAINERENV_ROCR_VISIBLE_DEVICES']:env[k]=''
 args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=2048M','-w','a14r2n20','-t','00:10:00','--job-name='+NAME,'--output='+str(R/'logs'/(NAME+'-%J.log')),'bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with receipt.open('x') as f:
  with (R/'logs'/(NAME+'-client.log')).open('x') as log:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':p.pid,'args':args,'entry_manifest_SHA256':sha(D/'manifest-r120j.json'),'HOST_compile':0,'reused_compiled_HOST_TUs':1,'GPU':0,'link_only':True,'completed':False};f.write(json.dumps(q,indent=2)+'\n')
 print(json.dumps(q))
if __name__=='__main__':main()
