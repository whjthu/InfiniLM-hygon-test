from pathlib import Path
import concurrent.futures,datetime,hashlib,json,os,subprocess,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/GLM-W4-fewshot-quality-r444-eight-slot-v1/deployment-v2'
assert hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()=='b56fc770fa2e4a466a732718d51dfe734ba17bc0e8f70e9fb33f8e427b498833'
m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert hashlib.sha256((D/'idle.py').read_bytes()).hexdigest()==m['files_SHA256']['idle.py']
ledger=json.loads((R/'results/GLM444-owned-ready-cleanup-20261104-a.json').read_bytes());assert ledger['completed']and ledger['STOP_committed']and not ledger.get('error')
assert all(v[2:4]==['COMPLETED','0:0']for v in ledger['last_GPU_accounting'].values())
slots=json.loads((D/'SLOTS.json').read_bytes())['slots'];by_parent={};expected={};metadata=[]
for slot in slots:
    parent=slot['parent'];by_parent.setdefault(parent,[]).extend(slot['nodes']);folder=R/'results/GLM-W4-fewshot-quality-r444'/slot['run']
    for stage,node in enumerate(slot['nodes']):
        leader=json.loads((folder/('leader-%d.json'%stage)).read_bytes());actors=json.loads((folder/('actors-%d.json'%stage)).read_bytes())
        assert leader['identity']==actors['identity'] and leader['identity']['actual_step']==ledger['slots'][slot['slot']]['GPU']
        expected[node]=sorted(set([leader['PID']]+actors['owned_process_PIDs']))
    metadata.append(dict(slot=slot['slot'],MODEL_READY_after_STOP=(folder/'MODEL_READY.json').exists(),closed_count=sum((folder/('closed-%d-%d.json'%(s,k))).exists()for s in range(2)for k in range(8))))
env={k:v for k,v in os.environ.items()if k not in['BASH_ENV','ENV','PYTHONSTARTUP','PYTHONINSPECT']and not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_'))}
env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',GLM_R425_EXPECTED_PIDS=json.dumps(expected),GLM_R425_POST_DEADLINE_UNIX=str(time.time()+90))
def one(item):
    parent,nodes=item;nodes=sorted(set(nodes));assert len(nodes)==4
    argv=['srun','--jobid='+str(parent),'--overlap','--exact','--immediate=15','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(nodes),'-t','00:02:00','--job-name=GLM444-owned-ready-post-'+str(parent)+'-20261104-a','python3','-B',str(D/'idle.py')]
    p=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=110)
    values=[json.loads(s)for s in p.stdout.splitlines()if s.startswith('{')]
    return dict(parent=parent,nodes=nodes,argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,observations=values)
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:post=list(pool.map(one,by_parent.items()))
values=[x for v in post for x in v['observations']]
passed=all(v['returncode']==0 for v in post)and len(values)==len({v['node']for v in values})==16
passed=passed and all(v['CPU_only']and not v['GPU_initialized']and not v['kfd_pids']and not v['matching_component_processes']and not v['expected_PIDs_still_exist']and v['ports_bindable']=={'43843':True}for v in values)
a=subprocess.run(['sacct','-S','2026-11-03','-j','910371,910374,910376,910380,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,Elapsed,NodeList%100'],capture_output=True,text=True,timeout=25)
lines=[s for s in a.stdout.splitlines()if'|GLM-r444-fewshot-fewshot-20261104-a-s'in s or'|GLM444-owned-'in s]
q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_post_passed=passed,new_GPU=0,post=post,original_GPU_srun_and_keeper_accounting=dict(returncode=a.returncode,stdout='\n'.join(lines),stderr=a.stderr),known_PIDs_by_node=expected,closed_marker_metadata=metadata,models_restarted=0,quality_requests_replayed=0)
p=R/'results/GLM444-owned-cleanup-independent-post-20261104-a.json'
with p.open('x')as f:json.dump(q,f,indent=2)
print(json.dumps(q))
