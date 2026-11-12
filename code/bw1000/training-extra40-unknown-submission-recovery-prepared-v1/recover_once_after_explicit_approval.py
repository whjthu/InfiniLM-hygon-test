"""Prepared recovery client. No network or execution without explicit approval.

Authenticate first and receive a nonce-bound READY before transmitting the
mutating payload. A pre-READY SSH failure then has a positive no-payload record.
"""
from pathlib import Path
import hashlib,json,queue,shlex,subprocess,threading,uuid
D=Path(__file__).resolve().parent;B=D.parent
approval=json.loads((D/'recovery-approved.json').read_bytes())
assert approval['approved_recovery'] is True and approval['scope']=='one_extra40_30day'
assert approval['original_request_id']=='42f2a1e31ffc476d8b03cf27082a79c2'
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
claim=json.loads((D/'recovery-publication/source-publication-receipt.json').read_bytes())['commit']
source_dir='code/bw1000/training-extra40-unknown-submission-recovery-prepared-v1/'
recovery=subprocess.check_output(git+['show',claim+':'+source_dir+'recover_missing_submission_prepared.py'])
assert hashlib.sha256(recovery).hexdigest()==approval['recovery_source_SHA256']
original=subprocess.check_output(git+['show','76cebab37bcb4669695be37b9b8021665ab7a427:code/bw1000/training-extra40-resource-expansion-v1/submit_extra40.py'])
request=uuid.uuid4().hex;nonce=uuid.uuid4().hex
payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nORIGINAL_REQUEST='+repr(approval['original_request_id'])+
    '\nORIGINAL_CLAIM='+repr('76cebab37bcb4669695be37b9b8021665ab7a427')+
    '\nORIGINAL_SOURCE='+repr(original)+'\nORIGINAL_SOURCE_SHA256='+repr(hashlib.sha256(original).hexdigest())+
    '\nEXPLICIT_USER_RECOVERY_APPROVAL=True\n').encode()+recovery
wire=json.dumps(dict(nonce=nonce,SHA256=hashlib.sha256(payload).hexdigest(),code=__import__('base64').b64encode(payload).decode())).encode()+b'\n'
ready_program='''import base64,hashlib,json,signal,sys
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('finite request expired')))
signal.alarm(215)
nonce=NONCE
print(json.dumps(dict(stage='AUTHENTICATED_READY',nonce=nonce)),flush=True)
raw=sys.stdin.buffer.readline(4*1024**2+1)
assert len(raw)<=4*1024**2 and raw.endswith(b'\\n')
v=json.loads(raw);assert v['nonce']==nonce
code=base64.b64decode(v['code'],validate=True);assert hashlib.sha256(code).hexdigest()==v['SHA256']
exec(compile(code,'published_extra40_recovery','exec'),dict(__name__='__main__'))
'''.replace('NONCE',repr(nonce))
ns={'__file__':str(D/'reuse.py')};s=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(s.split('REMOTE=',1)[0],'existing_cluster_SSH','exec'),ns)
ssh=list(ns['ssh']);assert ssh[-1]=='/usr/local/python3.12/bin/python3 -B -u -'
ssh[-1]='/usr/local/python3.12/bin/python3 -B -u -c '+shlex.quote(ready_program)
intent=dict(claim=claim,request_id=request,original_request_id=approval['original_request_id'],payload_SHA256=hashlib.sha256(payload).hexdigest(),
    exec_request_attempted=False,authenticated_ready_received=False,mutating_payload_transmitted=False,new_service=False,new_poller=False)
with (D/'recovery-launch-intent.json').open('x') as f:json.dump(intent,f,indent=2)
with (D/'recovery-stderr.private.txt').open('xb') as err:
    p=subprocess.Popen(ssh,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
    q=queue.Queue()
    threading.Thread(target=lambda:q.put(p.stdout.readline()),daemon=True).start()
    out=b''
    try:
        first=q.get(timeout=20)
        ready=json.loads(first)
        assert ready==dict(stage='AUTHENTICATED_READY',nonce=nonce)
        intent['authenticated_ready_received']=True
        # Any error after this point is UNKNOWN until original namespace/job reconciliation.
        intent['exec_request_attempted']=True
        with (D/'recovery-payload-write-intent.json').open('x') as f:json.dump(intent,f,indent=2)
        out,_=p.communicate(wire,timeout=195)
        intent['mutating_payload_transmitted']=True
    except BaseException as error:
        intent['transport_error_type']=type(error).__name__
        p.kill();p.wait(timeout=10)
        try:out=p.stdout.read()
        except ValueError:pass
    intent['transport_rc']=p.returncode
with (D/'recovery-stdout.private.jsonl').open('xb') as f:f.write(out)
intent['remote_outcome']='UNKNOWN' if intent['exec_request_attempted'] and p.returncode in (None,255) else ('NO_MUTATING_PAYLOAD_SENT' if not intent['exec_request_attempted'] else 'ORIGINAL_RETURNED')
if intent.get('transport_error_type') and intent['exec_request_attempted']:intent['remote_outcome']='UNKNOWN'
docs=[]
for line in out.splitlines():
    try:docs.append(json.loads(line))
    except ValueError:pass
final=[v for v in docs if v.get('request_id')==request and 'sbatch_rc' in v]
if final:assert len(final)==1;intent['value']=final[0]
intent['stdout_SHA256']=hashlib.sha256(out).hexdigest()
with (D/'recovery-transport-outcome.json').open('x') as f:json.dump(intent,f,indent=2)
v=intent.get('value',{})
print(json.dumps(dict(remote_outcome=intent['remote_outcome'],exec_request_attempted=intent['exec_request_attempted'],
    authenticated_ready_received=intent['authenticated_ready_received'],transport_rc=p.returncode,job_id=v.get('job_id'),new_job=v.get('new_job'),total_unique_nodes=v.get('total_unique_nodes'))))
