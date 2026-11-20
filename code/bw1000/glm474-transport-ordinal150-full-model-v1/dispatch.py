
import base64,hashlib,json,os,pathlib,socket,subprocess,sys,time
R=pathlib.Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
S = W = pathlib.Path(__file__).resolve().parent
epoch=float(sys.argv[1])
REQUEST, COMMIT=sys.argv[2:4]
claim_raw=(W/"GPU_RESOURCE_CLAIM.json").read_bytes()
claim=json.loads(claim_raw)
RUN=claim["run_id"]
CLAIM=hashlib.sha256(claim_raw).hexdigest()
assert 0 <= time.time()-epoch < 35 and REQUEST == claim["request"] and W == R/"results"/("glm474-transport-ordinal150-full-model-v1-"+REQUEST)
def write(name,value):
    with (W/name).open("x") as stream:
        json.dump(value,stream,indent=2)
        stream.flush();os.fsync(stream.fileno())
def ticks(pid):
    return int(pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(") ",1)[1].split()[19])
sys.path.insert(0,str(S))
from budget import PAIR_SECONDS, remaining, require_pair_capacity
require_pair_capacity(epoch,2)
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
 rc=subprocess.call(['/usr/bin/timeout','--signal=TERM','--kill-after=12s',str(left)+'s','/usr/local/python3.12/bin/python3','-B','-u',str(pathlib.Path(src)/'host_launch.py'),run,claim,'--pair-epoch',epoch])
with (work/'launcher-terminal.json').open('x') as stream:
 json.dump(dict(rc=rc,finished=time.time(),run_id=run),stream)
 stream.flush();os.fsync(stream.fileno())
raise SystemExit(rc)
"""
with (W/"launcher.log").open("xb") as log:
    p=subprocess.Popen(["/usr/local/python3.12/bin/python3","-B","-u","-c",runner,str(W),str(S),RUN,CLAIM,str(epoch)],
                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(request_id=REQUEST,run_id=RUN,PID=p.pid,startticks=ticks(p.pid),claim_commit=COMMIT,
             stage="LAUNCHER_STARTED",epoch=epoch)
write("launcher-owner.json",receipt)
print(json.dumps(receipt),flush=True)
