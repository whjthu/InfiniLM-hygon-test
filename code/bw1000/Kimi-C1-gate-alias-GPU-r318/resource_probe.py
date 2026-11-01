"""CPU-only KFD/port check; tolerate only observed TIME_WAIT for at most90s."""
from pathlib import Path
import os,socket,json,fcntl,struct,datetime,time,errno
node=socket.gethostname().split('.')[0];p=Path('/sys/class/kfd/kfd/proc');assert p.exists()
start=time.monotonic();history=[]
while True:
    pids=sorted(int(v.name)for v in p.iterdir()if v.name.isdigit());states=[]
    for path in ['/proc/net/tcp','/proc/net/tcp6']:
        for line in Path(path).read_text().splitlines()[1:]:
            fields=line.split()
            if int(fields[1].split(':')[-1],16)==30040:states.append(fields[3])
    bindable=False;bind_errno=None
    try:
        with socket.socket()as listener:
            listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);listener.bind(('',30040))
        bindable=True
    except OSError as exc:bind_errno=exc.errno
    history.append({'elapsed_s':time.monotonic()-start,'KFD_count':len(pids),'LISTEN':states.count('0A'),
                    'TIME_WAIT':states.count('06'),'bindable':bindable,'bind_errno':bind_errno})
    passed=not pids and '0A'not in states and bindable
    if passed:break
    if pids or '0A'in states or bind_errno!=errno.EADDRINUSE or not states or any(s!='06'for s in states)or time.monotonic()-start>=90:break
    time.sleep(2)
with socket.socket(socket.AF_INET,socket.SOCK_DGRAM)as sock:
    ip=socket.inet_ntoa(fcntl.ioctl(sock.fileno(),0x8915,struct.pack('256s',b'ib0'))[20:24])
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':node,
 'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True,
 'GPU_runtime_initialized':False,'kfd_pids':pids,'KFD_empty':not pids,'control30040_bindable':bindable,
 'control30040_LISTEN':states.count('0A'),'control30040_TIME_WAIT':states.count('06'),
 'observations':history,'ib0_ipv4':ip,'passed':passed}))
raise SystemExit(0 if passed else 1)
