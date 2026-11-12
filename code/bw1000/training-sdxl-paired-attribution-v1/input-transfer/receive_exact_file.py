from pathlib import Path
import hashlib,json,os,socket,sys,time
h=Path(PLAN['existing_HIP_counterpart']);assert h.is_file() and not h.is_symlink() and h.stat().st_size==PLAN['existing_HIP_bytes'] and hashlib.sha256(h.read_bytes()).hexdigest()==PLAN['existing_HIP_SHA256']
W=Path(PLAN['target_root']);destination=Path(PLAN['target_file']);assert destination==W/'inputs/A100-operands.pt' and not W.exists();os.umask(0o077);W.mkdir(mode=0o700);(W/'inputs').mkdir(mode=0o700);epoch=time.time();rest=Path('/proc/self/stat').read_text().split(') ',1)[1].split();intent=dict(claim=CLAIM,request_id=REQUEST,source=PLAN['source_file'],destination=str(destination),bytes=PLAN['bytes'],SHA256=PLAN['SHA256'],epoch=epoch,deadline=epoch+120,host=socket.gethostname(),PID=os.getpid(),startticks=int(rest[19]),new_service=False,new_poller=False,GPU_cards=0)
with (W/'intent.json').open('x') as f:json.dump(intent,f,indent=2)
part=destination.with_suffix('.pt.part');raw=sys.stdin.buffer.read(PLAN['bytes']+1);assert time.time()<epoch+120 and len(raw)==PLAN['bytes'] and hashlib.sha256(raw).hexdigest()==PLAN['SHA256']
fd=os.open(part,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
assert not destination.exists();part.replace(destination);receipt=dict(claim=CLAIM,request_id=REQUEST,passed=True,bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),destination=str(destination),file_mode=oct(destination.stat().st_mode&0o777),wall_seconds=time.time()-epoch,raw_file_torch_loaded=False,GPU_cards=0)
with (W/'stage-receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
