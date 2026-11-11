
"""Append only a verified missing suffix after the original receiver is terminal."""
import hashlib,json,os,signal,socket,sys,time
from pathlib import Path
from append_suffix import append_verified_suffix
STAGE=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/training-minicpm-native-mask-assets-v1')
assert hashlib.sha256((STAGE/'receive_archive.py').read_bytes()).hexdigest()=='c3e5f0812774f3e5fdc6a9f6383b8b9c5a46e18ef6aedbb1dded1ab14c3d2598'
sys.path.insert(0,str(STAGE))
from receive_archive import extract_pinned_archive
cfg=json.loads(sys.argv[1]);plan_raw=(STAGE/'ASSET_PLAN.json').read_bytes();plan=json.loads(plan_raw)
assert hashlib.sha256(plan_raw).hexdigest()==cfg['plan_SHA256']
root=Path(plan['receiver_root']);part=root/'archive.tar.gz.part';failure=(root/'receive-failure.json').read_bytes()
assert hashlib.sha256(failure).hexdigest()==cfg['original_failure_SHA256']
assert cfg['original_transport_terminal_rc']!=0 and json.loads(failure)['type']=='TimeoutError'
assert not (root/'archive.tar.gz').exists() and not (root/'assets').exists()
assert part.is_file() and not part.is_symlink() and part.stat().st_size==cfg['prefix_bytes']
assert 0<cfg['prefix_bytes']<cfg['archive_bytes']<=plan['maximum_archive_bytes']
assert cfg['remaining_bytes']==cfg['archive_bytes']-cfg['prefix_bytes']
assert not (root/'resume-tail-intent.json').exists(),'reconcile original tail request; never replay'
epoch=time.time();deadline=epoch+2400;mono=time.monotonic()
def expired(*unused):raise TimeoutError('new suffix-only scope2400s deadline')
signal.signal(signal.SIGALRM,expired);signal.alarm(2400)
def bound():assert time.time()<deadline and time.monotonic()-mono<2400
intent=dict(cfg,epoch=epoch,deadline=deadline,PID=os.getpid(),host=socket.gethostname(),
            boot_id_SHA256=hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest(),
            startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
with (root/'resume-tail-intent.json').open('x') as f:json.dump(intent,f,indent=2)
remaining=cfg['remaining_bytes'];completed=[]
try:
 append_verified_suffix(part,sys.stdin.buffer,cfg,bound)
 remaining=0
 part.rename(root/'archive.tar.gz')
 extract_pinned_archive(root/'archive.tar.gz',root,plan,plan_raw,bound,completed)
 result=dict(completed=True,planned_files=76,completed_files=len(completed),source_bytes=plan['source_bytes'],
             archive_SHA256=cfg['archive_SHA256'],archive_bytes=cfg['archive_bytes'],plan_SHA256=cfg['plan_SHA256'],
             original_failed_scope_preserved=True,original_failure_SHA256=cfg['original_failure_SHA256'],
             reused_prefix_bytes=cfg['prefix_bytes'],received_only_suffix_bytes=cfg['remaining_bytes'],
             epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-mono,GPU_initialization=False,
             claim_commit=cfg['claim_commit'],qualification='Exact asset bytes after verified-prefix suffix continuation; no Model qualification.')
 with (root/'resume-complete.json').open('x') as f:json.dump(result,f,indent=2)
 signal.alarm(0);print(json.dumps(result))
except BaseException as e:
 with (root/'resume-failure.json').open('x') as f:json.dump(dict(type=type(e).__name__,message=str(e)[:200],remaining_suffix_bytes=max(0,cfg['archive_bytes']-(part.stat().st_size if part.exists() else (root/'archive.tar.gz').stat().st_size)),completed_files=len(completed)),f,indent=2)
 raise
