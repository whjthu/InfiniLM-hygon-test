import pathlib,json,os,socket,sys
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D))
import target_protocol,entry_contract_r9143,parameter_loader_r9143,native_binding_r9143,host_observer_r9183,shard_head_contract_r9183
m=json.loads((D/'model-manifest-r9183.json').read_bytes());entry_contract_r9143.validate(m);entry_contract_r9143.verify_inputs(m);node=int(os.environ['SLURM_PROCID']);assert socket.gethostname()==m['nodes'][node]and int(os.environ['SLURM_CPUS_PER_TASK'])==64 and len(os.sched_getaffinity(0))>=64;assert 'torch'not in sys.modules
for module in [target_protocol,entry_contract_r9143,parameter_loader_r9143,native_binding_r9143,host_observer_r9183,shard_head_contract_r9183]:assert pathlib.Path(module.__file__).resolve().parent==D
print(json.dumps({'passed':True,'CPU_only':True,'GPU':False,'dlopen':False,'Runtime_created':False,'node':node,'host':socket.gethostname(),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'affinity64':len(os.sched_getaffinity(0)),'manifest_SHA256':entry_contract_r9143.sha(D/'model-manifest-r9183.json')}),flush=True)
