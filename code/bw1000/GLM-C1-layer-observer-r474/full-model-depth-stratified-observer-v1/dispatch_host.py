
import base64,hashlib,json,os,pathlib,socket,subprocess,sys,time,signal
epoch=EPOCH
assert 0<epoch<=time.time()<epoch+60000
R=pathlib.Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
W=R/"results"/("GLM474-depth-observer-model-"+RUN+"-dispatch")
W.mkdir(mode=0o700)
S=W/"src";S.mkdir(mode=0o700)
for name,(pin,b64) in FILES.items():
    assert pathlib.Path(name).name==name
    data=base64.b64decode(b64,validate=True)
    assert hashlib.sha256(data).hexdigest()==pin
    with (S/name).open("xb") as stream:stream.write(data)
def write(name,value):
    with (W/name).open("x") as stream:
        json.dump(value,stream,indent=2)
        stream.flush();os.fsync(stream.fileno())
def ticks(pid):
    return int(pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(") ",1)[1].split()[19])
sys.path.insert(0,str(S))
from budget import PAIR_SECONDS, remaining, require_pair_capacity
require_pair_capacity(epoch,1)
write("request.json",dict(run_id=RUN,request_id=REQUEST,claim_commit=COMMIT,epoch=epoch,deadline=epoch+PAIR_SECONDS,host=socket.gethostname(),PID=os.getpid(),startticks=ticks(os.getpid()),claim_SHA256=CLAIM))
runner="""import json,os,subprocess,sys,time,pathlib
w,src,run,claim,epoch=sys.argv[1:]
sys.path.insert(0,src)
from budget import PAIR_SECONDS, remaining
work=pathlib.Path(w)
left=int(remaining(float(epoch),PAIR_SECONDS))-20
if left<=0:
 rc=124
else:
 rc=subprocess.call(['/usr/bin/timeout','--signal=TERM','--kill-after=12s',str(left)+'s','/usr/local/python3.12/bin/python3','-B','-u',str(pathlib.Path(src)/'host_launch.py'),run,claim,'--launch-epoch',epoch])
with (work/'launcher-terminal.json').open('x') as stream:
 json.dump(dict(rc=rc,finished=time.time(),run_id=run),stream)
 stream.flush();os.fsync(stream.fileno())
raise SystemExit(rc)
"""
write('control-state.json',dict(request_id=REQUEST,publication=COMMIT,claim_SHA256=CLAIM,epoch=epoch,exec_request_attempted=False))
with (W/"launcher.log").open("xb") as log:
    control=dict(request_id=REQUEST,publication=COMMIT,claim_SHA256=CLAIM,epoch=epoch,exec_request_attempted=True)
    with (W/'control-state.next').open('x') as stream:json.dump(control,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(W/'control-state.next',W/'control-state.json')
    p=subprocess.Popen(["/usr/local/python3.12/bin/python3","-B","-u","-c",runner,str(W),str(S),RUN,CLAIM,str(epoch)],
                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
try:
    receipt=dict(request_id=REQUEST,run_id=RUN,PID=p.pid,startticks=ticks(p.pid),claim_commit=COMMIT,host=socket.gethostname(),boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 stage="LAUNCHER_STARTED",epoch=epoch)
    write("launcher-owner.json",receipt)
except BaseException:
    # Held unreaped Popen/newsession remains the anchor if /proc or write fails.
    if p.poll() is None:
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait(timeout=3)
    raise
print(json.dumps(receipt),flush=True)
