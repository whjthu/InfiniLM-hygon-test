"""Run actual private worker/client/controller consumers; fake GPU/HTTP/Slurm leaves only."""
import pathlib,json,hashlib,os,subprocess,time,traceback
D=pathlib.Path(__file__).resolve().parent;S=D.parent;R=pathlib.Path('/refactor')
O=R/'candidates/Deep387-Model-entry-CPU-v1';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,CPU_only=True,GPU=False,Native_dlopen=False,
 Runtime_initialized=False,weights_opened=False,Native_compilation=False,
 publication_commit=os.environ['DEEP387_PUBLICATION_COMMIT'],
 actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],commands=[])
started=time.monotonic()
try:
 mp=S/'MODEL_SOURCE_MANIFEST.json';assert sha(mp)==os.environ['DEEP387_MODEL_CPU_MANIFEST_SHA'];m=json.loads(mp.read_bytes())
 assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
 owner_sha=sha(S/'owner/SOURCE_MANIFEST.json');assert owner_sha==m['owner_manifest_SHA256']
 for name,args in [('entry',[str(D/'CPU_entry.py'),'--owner-sha',owner_sha]),('controller',[str(S/'controller/CPU_contract.py')])]:
  argv=['python3','-B',*args];q['commands'].append(argv)
  p=subprocess.run(argv,capture_output=True,text=True,timeout=180)
  (O/(name+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(name,p.stderr[-5000:],p.stdout[-2000:])
 entry=R/'candidates/Deep-forward-position-proof-r387-Model-v1-owner-CPU-v3/CPU-entry.json'
 controller=entry.with_name('controller-contract.json');e=json.loads(entry.read_bytes());c=json.loads(controller.read_bytes())
 assert e['passed']and c['passed']and e['owner_manifest_SHA256']==owner_sha
 assert all(v['passed']for v in e['cases'])
 q.update(completed=True,passed=True,source_manifest_SHA256=sha(mp),owner_manifest_SHA256=owner_sha,
  controller_manifest_SHA256=sha(S/'controller/SOURCE_MANIFEST.json'),
  entry_receipt={'path':str(entry),'SHA256':sha(entry),'cases':len(e['cases'])},
  controller_receipt={'path':str(controller),'SHA256':sha(controller),'record':c},
  entry_cases=e['cases'],actual_loader_AST=e['actual_loader_AST'],
  GPU_GO=False,scope='Actual file identity/worker report -> actual client main; frozen HC1/proof plan and actual controller.main. HOST metadata binding is real; Native/HTTP/Slurm/GPU leaves are fake.')
except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc())
q.update(completed=True,seconds=time.monotonic()-started)
(O/'CPU-whole.json').write_text(json.dumps(q,indent=2)+'\n')
print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_step','seconds','error']}));assert q['passed']
