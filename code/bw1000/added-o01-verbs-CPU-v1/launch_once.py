"""One published target HIP CPU scope, after an authenticated READY handshake."""
from pathlib import Path
import base64,hashlib,json,queue,re,shlex,subprocess,sys,threading,uuid
D=Path(__file__).resolve().parent
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
B=D.parent
tag=sys.argv[1];assert re.fullmatch('[a-z0-9-]{1,30}',tag)
assert not (D/'launch-intent.json').exists(), 'Original mutating request already attempted;readonly reconciliation only'
claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit']
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/added-o01-verbs-CPU-v1.json']))
assert task['CPU_GO'] and not task['GPU_GO'] and task['max_CPU_invocations']==1 and task['actual_CPU_invocations']==0
files={};sources={}
for name,pin in task['source_SHA256'].items():
    raw=subprocess.check_output(git+['show',claim+':'+task['source']+'/'+name])
    assert hashlib.sha256(raw).hexdigest()==pin;sources[name]=raw
for target,name in task['remote_files'].items():
    raw=sources[name];files[target]=dict(b64=base64.b64encode(raw).decode(),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
request=uuid.uuid4().hex;nonce=uuid.uuid4().hex
payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nFILES='+repr(files)+'\n').encode()+sources['receive_once.py']
wire=json.dumps(dict(nonce=nonce,SHA256=hashlib.sha256(payload).hexdigest(),code=base64.b64encode(payload).decode())).encode()+b'\n'
assert len(wire)<48*1024**2
ready_program='''import base64,hashlib,json,signal,sys
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('finite originalCPU request expired')))
signal.alarm(210)
nonce=NONCE
print(json.dumps(dict(stage='AUTHENTICATED_READY',nonce=nonce)),flush=True)
raw=sys.stdin.buffer.readline(48*1024**2+1)
assert len(raw)<=48*1024**2 and raw.endswith(b'\\n')
v=json.loads(raw);assert v['nonce']==nonce
code=base64.b64decode(v['code'],validate=True);assert hashlib.sha256(code).hexdigest()==v['SHA256']
exec(compile(code,'published_original_zimage_model_identity_CPU','exec'),dict(__name__='__main__'))
'''.replace('NONCE',repr(nonce))
intent=dict(claim=claim,request_id=request,tag=tag,payload_SHA256=hashlib.sha256(payload).hexdigest(),
    exec_request_attempted=False,authenticated_ready_received=False,mutating_payload_transmitted=False,
    new_service=False,new_poller=False,GPU_cards=0)
with (D/(tag+'-authentication-intent.json')).open('x') as f:json.dump(intent,f,indent=2)
import ast
tree=ast.parse((B/'probe_ssh_transport_r9543.py').read_bytes())
assign={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
items=assign['a'].elts;known=Path(ast.literal_eval(assign['r'].args[0]))/'known_hosts'
ssh=['ssh.exe','-T','-i',str(Path.home()/'.ssh/hygon-261111.txt'),'-p',ast.literal_eval(items[3]),'-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(known),'-o','BatchMode=yes','-o','ConnectTimeout=10',ast.literal_eval(items[16]),'/usr/local/python3.12/bin/python3 -B -u -c '+shlex.quote(ready_program)]
with (D/(tag+'-stderr.private.txt')).open('xb') as err:
    p=subprocess.Popen(ssh,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
    q=queue.Queue();threading.Thread(target=lambda:q.put(p.stdout.readline()),daemon=True).start()
    out=b''
    try:
        ready=json.loads(q.get(timeout=20))
        assert ready==dict(stage='AUTHENTICATED_READY',nonce=nonce)
        intent['authenticated_ready_received']=True
        write_intent=dict(**{k:v for k,v in intent.items() if k!='exec_request_attempted'},exec_request_attempted=True,replay_allowed=False)
        with (D/'launch-intent.json').open('x') as f:json.dump(write_intent,f,indent=2)
        intent['exec_request_attempted']=True
        out,_=p.communicate(wire,timeout=200)
        intent['mutating_payload_transmitted']=True
    except BaseException as error:
        intent['transport_error_type']=type(error).__name__
        p.kill();p.wait(timeout=10)
        try:out=p.stdout.read()
        except ValueError:pass
    intent['transport_rc']=p.returncode
with (D/(tag+'-stdout.private.jsonl')).open('xb') as f:f.write(out)
unknown=intent['exec_request_attempted'] and (p.returncode in (None,255) or 'transport_error_type' in intent)
intent['remote_outcome']='UNKNOWN' if unknown else ('NO_MUTATING_PAYLOAD_SENT' if not intent['exec_request_attempted'] else 'ORIGINAL_RETURNED')
intent['stdout_SHA256']=hashlib.sha256(out).hexdigest()
intent['stderr_SHA256']=hashlib.sha256((D/(tag+'-stderr.private.txt')).read_bytes()).hexdigest()
docs=[]
for line in out.splitlines():
    try:docs.append(json.loads(line))
    except ValueError:pass
final=[v for v in docs if v.get('request_id')==request and 'accounting' in v]
if final:assert len(final)==1;intent['value']=final[0]
with (D/(tag+'-transport-outcome.json')).open('x') as f:json.dump(intent,f,indent=2)
if intent['exec_request_attempted']:
    with (D/'transport-outcome.json').open('x') as f:json.dump(intent,f,indent=2)
v=intent.get('value',{});cpu=(v.get('CPU-result.json') or {}).get('value',{})
print(json.dumps(dict(remote_outcome=intent['remote_outcome'],exec_request_attempted=intent['exec_request_attempted'],
    authenticated_ready_received=intent['authenticated_ready_received'],transport_rc=p.returncode,
    passed=v.get('passed'),CPU_passed=cpu.get('passed'),phase=cpu.get('phase'),returncode=v.get('returncode'),
    owned_group_reaped=v.get('owned_srun_reaped'),actual_trainable_tensors=cpu.get('actual_trainable_tensors'),
    actual_trainable_elements=cpu.get('actual_trainable_elements'),GPU_cards=0)))
