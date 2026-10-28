"""Root CPU-only prepare/preflight. No GPU GO, GPU step, Native dlopen or Runtime."""
import argparse,hashlib,json,os,pathlib,subprocess
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=R/'incoming/Kimi-agent-service-Model-r189'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);a=p.parse_args()
 assert sha(D/'service-source-manifest-r189.json')==a.manifest_sha
 m=json.loads((D/'service-source-manifest-r189.json').read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
 record=R/'results/Kimi-r189-CPU-prepare-launch.json';assert not record.exists()
 result={'CPU_only':True,'GPU':False,'Native_dlopen':False,'Runtime_created':False,'source_manifest_SHA256':a.manifest_sha,'completed':False,'steps':[],'retries':0}
 with record.open('x')as f:json.dump(result,f)
 try:
  for name,inside,nodes,cpus in [('overlay','runtime/CPU_prepare_inside_r189.sh',['a14r2n08'],1),('preflight','runtime/preflight_inside.sh',m['resources']['nodes'],64)]:
   cmd=['srun','--jobid=910380','--overlap','--exact','--immediate=30','--cpu-bind=none','-N'+str(len(nodes)),'-n'+str(len(nodes)),'--ntasks-per-node=1','-c'+str(cpus),'--gres=none','--mem=8192M','-w',','.join(nodes),'-t','00:05:00','--job-name=Kimi-r189-CPU-'+name,'--export=ALL,GLOBAL_MODEL_MODE_R2354=shca,GLOBAL_OFFSET_R2354=0,WAVE_INSIDE_R2323=Kimi-agent-service-Model-r189/'+inside,'bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
   z=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=340)
   rows=[json.loads(x)for x in z.stdout.splitlines()if x.startswith('{')]
   result['steps'].append({'name':name,'returncode':z.returncode,'rows':rows,'stderr_tail':z.stderr[-1000:]})
   record.write_text(json.dumps(result,indent=2)+'\n')
   assert z.returncode==0 and len(rows)==len(nodes)
   assert all(x.get('completed',x.get('passed'))and x['CPU_only']and not x['GPU']for x in rows)
  result['completed']=True
 except BaseException as e:result['error']=repr(e);raise
 finally:record.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'completed':result['completed'],'CPU_only':True,'GPU':False,'receipt':str(record),'receipt_SHA256':sha(record)}))
if __name__=='__main__':main()
