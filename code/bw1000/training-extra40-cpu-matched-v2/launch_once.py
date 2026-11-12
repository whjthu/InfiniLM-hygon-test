"""One CPU-matched extra40 repair after finite ordinary authenticated READY."""
from pathlib import Path
import base64,hashlib,json,queue,re,shlex,subprocess,sys,threading,uuid
D=Path(__file__).resolve().parent;B=D.parent
tag=sys.argv[1];assert re.fullmatch('[a-z0-9-]{1,30}',tag)
assert not (D/'launch-intent.json').exists(), 'Original mutating repair already attempted;readonly reconciliation only'
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit']
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/training-extra40-cpu-matched-v2.json']))
assert task['allocation_GO'] and task['max_new_sbatch_submissions']==1 and task['actual_new_sbatch_submissions']==0
source=subprocess.check_output(git+['show',claim+':'+task['source']+'/replace_pending_once.py'])
assert hashlib.sha256(source).hexdigest()==task['source_SHA256']['replace_pending_once.py']
request=uuid.uuid4().hex;nonce=uuid.uuid4().hex
payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\n').encode()+source
wire=json.dumps(dict(nonce=nonce,SHA256=hashlib.sha256(payload).hexdigest(),code=base64.b64encode(payload).decode())).encode()+b'\n'
ready_program='''import base64,hashlib,json,signal,sys
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('finite resource repair expired')))
signal.alarm(215)
nonce=NONCE
print(json.dumps(dict(stage='AUTHENTICATED_READY',nonce=nonce)),flush=True)
raw=sys.stdin.buffer.readline(1024**2+1)
assert len(raw)<=1024**2 and raw.endswith(b'\\n')
v=json.loads(raw);assert v['nonce']==nonce
code=base64.b64decode(v['code'],validate=True);assert hashlib.sha256(code).hexdigest()==v['SHA256']
exec(compile(code,'published_extra40_CPU_matched_repair','exec'),dict(__name__='__main__'))
'''.replace('NONCE',repr(nonce))
ns={'__file__':str(D/'reuse.py')};old=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(old.split('REMOTE=',1)[0],'existing_cluster_SSH','exec'),ns)
ssh=list(ns['ssh']);assert ssh[-1]=='/usr/local/python3.12/bin/python3 -B -u -'
ssh[-1]='/usr/local/python3.12/bin/python3 -B -u -c '+shlex.quote(ready_program)
intent=dict(claim=claim,request_id=request,tag=tag,payload_SHA256=hashlib.sha256(payload).hexdigest(),
    exec_request_attempted=False,authenticated_ready_received=False,mutating_payload_transmitted=False,new_service=False,new_poller=False)
with (D/(tag+'-authentication-intent.json')).open('x') as f:json.dump(intent,f,indent=2)
with (D/(tag+'-stderr.private.txt')).open('xb') as err:
    p=subprocess.Popen(ssh,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
    q=queue.Queue();threading.Thread(target=lambda:q.put(p.stdout.readline()),daemon=True).start();out=b''
    try:
        ready=json.loads(q.get(timeout=20));assert ready==dict(stage='AUTHENTICATED_READY',nonce=nonce)
        intent['authenticated_ready_received']=True
        write=dict(**{k:v for k,v in intent.items() if k!='exec_request_attempted'},exec_request_attempted=True,replay_allowed=False)
        with (D/'launch-intent.json').open('x') as f:json.dump(write,f,indent=2)
        intent['exec_request_attempted']=True
        out,_=p.communicate(wire,timeout=195);intent['mutating_payload_transmitted']=True
    except BaseException as error:
        intent['transport_error_type']=type(error).__name__;p.kill();p.wait(timeout=10)
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
final=[v for v in docs if v.get('request_id')==request and ('sbatch_rc' in v or v.get('preserved_existing_running'))]
if final:assert len(final)==1;intent['value']=final[0]
with (D/(tag+'-transport-outcome.json')).open('x') as f:json.dump(intent,f,indent=2)
if intent['exec_request_attempted']:
    with (D/'transport-outcome.json').open('x') as f:json.dump(intent,f,indent=2)
v=intent.get('value',{})
print(json.dumps(dict(remote_outcome=intent['remote_outcome'],exec_request_attempted=intent['exec_request_attempted'],
    authenticated_ready_received=intent['authenticated_ready_received'],transport_rc=p.returncode,
    job_id=v.get('job_id'),sbatch_rc=v.get('sbatch_rc'),preserved_existing_running=v.get('preserved_existing_running'),
    new_job=v.get('new_job'),confirmed_total_nodes=v.get('confirmed_total_nodes'),confirmed_total_DCU=v.get('confirmed_total_DCU'))))
