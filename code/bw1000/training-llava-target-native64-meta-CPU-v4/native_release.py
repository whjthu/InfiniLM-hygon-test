"""Stdlib-only in-container handshake. No native imports before HOST release."""
import json,os,sys,time
from pathlib import Path

def need(ok,why):
    if not ok:raise ValueError(why)

def main():
    root=Path(sys.argv[1]);epoch=float(sys.argv[2]);deadline=float(sys.argv[3]);work=float(sys.argv[4])
    need(epoch<=time.time()<work<deadline==epoch+300,'ORIGINAL_NATIVE_RELEASE_EPOCH')
    f=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
    own=dict(PID=os.getpid(),PPID=int(f[1]),PGID=int(f[2]),startticks=int(f[19]),UID=os.getuid(),
             host=os.uname().nodename,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    raw=(json.dumps(dict(owner=own,epoch=epoch,deadline=deadline,work_deadline=work,
                         native_imports=0,Torch_imported=False),sort_keys=True)+'\n').encode()
    with (root/'native-ready.json').open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    end=time.monotonic()+min(20,work-time.time())
    while time.time()<work and time.monotonic()<end:
        p=root/'native-release.json'
        if p.exists():
            need(p.is_file() and not p.is_symlink() and p.stat().st_size<=65536,'BOUNDED_NATIVE_RELEASE')
            v=json.loads(p.read_bytes())
            need(v['owner']==own and v['epoch']==epoch and v['deadline']==deadline
                 and v['work_deadline']==work and v['native_release_authorized'] is True
                 and v['memory_scope_qualified'] is True,'EXACT_HOST_RELEASE_BINDING')
            source=Path(__file__).resolve().parent/'worker.py'
            os.execv('/usr/bin/python3',['/usr/bin/python3','-B',str(source),'--epoch',str(epoch),
                       '--deadline',str(deadline),'--work-deadline',str(work)])
        time.sleep(min(.05,max(0,end-time.monotonic())))
    raise TimeoutError('NO_QUALIFIED_HOST_RELEASE_NO_NATIVE')

if __name__=='__main__':main()
