import pathlib,json,hashlib,os,time,ctypes
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];path=r/'results'/('deep-owner-device-probe-r8923-'+tag+'.json');assert not path.exists()
m=json.loads((r/'incoming/Deep-owner-device-probe-manifest-r8923.json').read_text());bp=r/m['cpu_receipt_path'];sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();assert sha(bp)==m['existing_remote_SHA256'][m['cpu_receipt_path']];q=json.loads(bp.read_text());assert q['CPU_only']and q['completed']and q['binary_SHA256']==m['DSO_SHA256'];assert sha(q['binary'])==q['binary_SHA256']
for n,v in q['actual_DTK_bindings'].items():assert sha(n)==v['SHA256']
report=dict(hardware='BW1000',level='DEVICE_EVENT_DIAGNOSTIC',status='RUNNING',completed=False,step=tag,phase='metadata_passed',DSO_SHA256=q['binary_SHA256'],no_Model_weights_CCL=True)
def save():
 t=path.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2));t.replace(path)
save()
try:
 import torch
 torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0);lib=ctypes.CDLL(q['binary']);fn=lib.deep_owner_device_probe_r8922;fn.argtypes=[];fn.restype=ctypes.c_int
 report['phase']='native_probe_enter';save();start=time.monotonic();code=fn();report.update(phase='native_probe_return',native_returncode=code,elapsed_seconds=time.monotonic()-start);save();assert code==0;report.update(completed=True,status='COMPLETE',phase='complete');save()
except BaseException as e:report.update(error=repr(e),phase='exception',status='FAILED');save();raise
