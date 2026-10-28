"""Materialize already-root-authorized allocation GO with actual Slurm step."""
import pathlib,hashlib,json,os,re,socket,time
R=pathlib.Path('/refactor');D=R/'incoming/GLM-PP4-capacity-r186';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stage=int(os.environ['SLURM_PROCID']);nodes=['a02r4n12','a02r4n13','a02r4n14','a02r4n15'];assert stage in range(4)and socket.gethostname().split('.')[0]==nodes[stage]
assert os.environ['SLURM_JOB_ID']=='910376'
commit=os.environ['GLM_ALLOCATION_PUBLICATION_COMMIT_R188'];assert re.fullmatch('[0-9a-f]{40}',commit)
cp=D/'CPU-result-r186.json';assert sha(cp)==os.environ['GLM_ALLOCATION_CPU_RECEIPT_SHA256_R188'];q=json.loads(cp.read_bytes());assert q['completed']and q['CPU_only']and not q['GPU_initialized']and not q['dlopen']
mf=sha(D/'source-manifest-r186.json');assert mf=='25c12e909ad6d317e163128523f79e38ac8693f2e54b9a349b302bd4114e26c9'and q['source_manifest_SHA256']==mf
p=pathlib.Path(os.environ['GLM_ALLOCATION_GO_R186']);assert p.parent==R/'control'
g={'GPU_authorized':True,'operation':'allocation_only_no_weights_no_forward','branch':'bw1000-experiments','publication_commit':commit,'source_manifest_SHA256':mf,'CPU_receipt_SHA256':sha(cp),'TP':8,'PP':4,'rows':1,'capacity':32768,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'root_authorization_materialized_only':True}
if stage==0:
 assert not p.exists();tmp=p.with_suffix('.tmp');assert not tmp.exists()
 with tmp.open('x')as f:json.dump(g,f)
 tmp.replace(p)
start=time.monotonic()
while not p.exists():
 if time.monotonic()-start>90:raise TimeoutError('Root-authorized actual-step GO visibility')
 time.sleep(.1)
assert json.loads(p.read_bytes())==g
print('GLM_ALLOCATION_R188_ACTUAL_STEP_GO',stage,flush=True)
