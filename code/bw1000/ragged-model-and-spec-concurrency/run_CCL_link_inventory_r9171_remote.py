import pathlib,json,subprocess,os,datetime,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming/CCL-link-diagnostic-r9171';i.mkdir(exist_ok=False)
worker="""import pathlib,json,socket,shutil
q={'node':socket.gethostname(),'GPU_context_created':False,'tools':{n:shutil.which(n)for n in ['ib_write_bw','ib_read_bw','ib_send_bw','ibv_devinfo','rdma','all_reduce_perf']},'ports':[]}
for p in pathlib.Path('/sys/class/infiniband').glob('*/ports/*'):
 d={'path':str(p),'values':{}}
 for n in ['state','phys_state','rate','link_layer','counters/port_xmit_data','counters/port_rcv_data','counters/port_xmit_packets','counters/port_rcv_packets','counters/port_xmit_wait']:
  z=p/n
  if z.exists():
   try:d['values'][n]=z.read_text().strip()
   except OSError as e:d['values'][n]='UNREADABLE:'+type(e).__name__
 q['ports'].append(d)
print(json.dumps(q))
"""
(i/'inventory.py').write_bytes(worker.encode())
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
args=['srun','--jobid=910373','--overlap','--exact','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w','a02r3n17,a02r3n18','-t','00:02:00','--job-name=CCL-link-inventory-r9171','python3',str(i/'inventory.py')]
p=subprocess.run(args,env=env,capture_output=True,text=True,timeout=150);records=[json.loads(s)for s in p.stdout.splitlines()if s.startswith('{')];assert p.returncode==0 and len(records)==2,(p.returncode,p.stderr)
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'GPU':False,'parent':'910373','nodes':['a02r3n17','a02r3n18'],'args':args,'records':records,'CPU_source_SHA256':hashlib.sha256(worker.encode()).hexdigest()};(r/'results/CCL-link-inventory-r9171.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))
