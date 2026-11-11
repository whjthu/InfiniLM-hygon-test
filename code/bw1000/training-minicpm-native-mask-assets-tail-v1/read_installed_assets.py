
"""Independent CPU byte verification; no tensor deserialization or GPU imports."""
import hashlib,json,signal,sys,time
from pathlib import Path
stage=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/training-minicpm-native-mask-assets-v1')
expected=json.loads(sys.argv[1]);raw=(stage/'ASSET_PLAN.json').read_bytes();plan=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()=='ba5caf50af52089260abd191ab2498796d2d3d89d6161f7cc6cb8d4016ce46a9'
root=Path(plan['receiver_root']);receipt_raw=(root/'resume-complete.json').read_bytes();receipt=json.loads(receipt_raw)
assert receipt['completed'] and receipt['completed_files']==receipt['planned_files']==76
assert receipt['archive_SHA256']==expected['archive_SHA256'] and receipt['archive_bytes']==expected['archive_bytes']
assert receipt['source_bytes']==plan['source_bytes']==3864273443 and not receipt['GPU_initialization']
assert receipt['deadline']==receipt['epoch']+2400 and receipt['wall_seconds']<2400
start=time.monotonic()
def expired(*unused):raise TimeoutError('independent CPU byte audit300s limit')
signal.signal(signal.SIGALRM,expired);signal.alarm(300)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
assert sha(root/'archive.tar.gz')==expected['archive_SHA256']
verified=[]
for row in plan['files']:
 p=root/'assets'/row['name'];assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to((root/'assets').resolve())
 assert p.stat().st_size==row['bytes'] and sha(p)==row['SHA256'],row['name'];verified.append(row)
for rank in range(8):
 r=json.loads((root/'assets/masks'/('rank-%02d'%rank)/'complete.json').read_bytes())
 c=r['contract'];assert r['mode']=='capture' and c['rank']==rank and c['world_size']==8
 assert c['planned_forwards']==8 and c['accumulation']==4 and len(c['module_inventory'])==186
 assert c['binding']['case']=='T05.sft' and c['binding']['initial_trainable_SHA256']==plan['initial_trainable_SHA256']
 assert all(x['p']==.05 and x['inplace'] is False for x in c['module_inventory'])
 assert len(r['files'])==8 and r['native_dropout_calls']==1488
 for i,f in enumerate(r['files'],1):
  assert f['name']=='forward-%04d-masks.pt'%i and f['global_step']==(i-1)//4 and f['micro_ordinal']==i
  row=next(x for x in verified if x['name']=='masks/rank-%02d/'%rank+f['name'])
  assert f['bytes']==row['bytes'] and f['SHA256']==row['SHA256']
result=dict(independent_passed=True,planned_files=76,verified_files=len(verified),verified_source_bytes=sum(x['bytes'] for x in verified),archive_bytes=expected['archive_bytes'],archive_SHA256=expected['archive_SHA256'],plan_SHA256=hashlib.sha256(raw).hexdigest(),receiver_receipt_SHA256=hashlib.sha256(receipt_raw).hexdigest(),source_claim='e7a2b02e11fe1df227ec83fb49a0f0c30c7abe32',tail_claim=receipt['claim_commit'],reference_claim=plan['reference_claim'],framework_commit=plan['framework_commit'],mask_rank_manifests=8,actual_native_mask_packets=64,actual_native_mask_calls=11904,initial_trainable_file_SHA256=next(x['SHA256'] for x in verified if x['name']=='capture/rank-00/initial.pt'),initial_named_parameters_SHA256=plan['initial_trainable_SHA256'],mask_root=str(root/'assets/masks'),initial_state=str(root/'assets/capture/rank-00/initial.pt'),rank_manifest_SHA256=plan['rank_manifest_SHA256'],wall_seconds=time.monotonic()-start,reader_SHA256=sha(Path(__file__)),GPU_initialized=False,Torch_imported=False,Model_GO=False,numerics_qualified=False,observer_reference_drift_resolved=False,limits=['Asset identities only; no Native/SDK/model load or training invocation.','Original reference numerical drift remains UNKNOWN; no changed numerical threshold or observer-neutrality claim.','200 full native update/output tensors remain in original A100 private storage; this transfers only required stochastic inputs/initial state and3 metadata files.'])
signal.alarm(0)
with (root/'independent-assets.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
