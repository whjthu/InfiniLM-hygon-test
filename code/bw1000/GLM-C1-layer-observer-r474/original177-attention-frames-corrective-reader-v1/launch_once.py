"""One published read-only original177 consume; no replay on failure/UNKNOWN."""
import argparse,base64,hashlib,json,math,os,re,runpy,subprocess,time,uuid
from pathlib import Path
D=Path(__file__).resolve().parent
G=D.parents[3]
LOCAL=G/'local-only/GLM474-original177-attention-frames-corrective-reader-v1-original'
HELPER=Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA='3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
H=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def save(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
def main(a):
    epoch=time.time();anchor=time.monotonic()
    def remaining(reserve=0):
        left=min(epoch+200-time.time(),200-(time.monotonic()-anchor))-reserve
        need(left>0,'CLIENT200_EXHAUSTED');return left
    need(re.fullmatch('[0-9a-f]{40}',a.publication) and all(re.fullmatch('[0-9a-f]{64}',x) for x in (a.source_manifest_sha256,a.claim_sha256)),'EXACT_PUBLICATION_PINS')
    need(not LOCAL.exists(),'ONCE_SCOPE_EXISTS_NO_REPLAY');LOCAL.mkdir(mode=0o700)
    request=uuid.uuid4().hex
    result=dict(request_id=request,reader_publication=a.publication,epoch=epoch,client_deadline=epoch+200,remote_deadline=epoch+180,attempted=False,remote_state='NOT_ATTEMPTED',replay_allowed=False,Model_qualified=False,GPU_GO=False)
    save(LOCAL/'intent.json',dict(result,source_manifest_SHA256=a.source_manifest_sha256,claim_SHA256=a.claim_sha256))
    stdout=stderr=b'';code=None
    git=['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G)]
    def show(commit,path):return subprocess.check_output(git+['show',commit+':'+path],timeout=min(12,remaining(20)),stderr=subprocess.DEVNULL)
    try:
        relative=D.relative_to(G).as_posix();raw=(D/'SOURCE_FILES.json').read_bytes()
        need(H(raw)==a.source_manifest_sha256 and show(a.publication,relative+'/SOURCE_FILES.json')==raw,'PUBLISHED_SOURCE_MANIFEST')
        mf=json.loads(raw);need({p.name for p in D.iterdir() if p.is_file()}==set(mf)|{'SOURCE_FILES.json'},'EXACT_OWN_INVENTORY')
        blobs={'SOURCE_FILES.json':raw}
        for name,pin in mf.items():
            need(Path(name).name==name,'SAFE_SOURCE_NAME');b=(D/name).read_bytes()
            need(len(b)==pin['bytes'] and H(b)==pin['SHA256'] and show(a.publication,relative+'/'+name)==b,'PUBLISHED_SOURCE_MEMBER');blobs[name]=b
        claim=json.loads(blobs['CLAIM.json']);binding=json.loads(blobs['BINDING.json'])
        need(H(blobs['CLAIM.json'])==a.claim_sha256 and claim['ROOT_GO'] is True and claim['max_invocations']==1 and claim['remote_seconds']==180 and claim['client_seconds']==200 and claim['Model_GO'] is False and claim['GPU_GO'] is False,'EXACT_READONLY_CLAIM')
        dirty=subprocess.check_output(git+['status','--porcelain','--untracked-files=all','--',relative],timeout=min(12,remaining(20)),stderr=subprocess.DEVNULL);need(not dirty,'OWN_SOURCE_DIRTY')
        oldmap=blobs['ORIGINAL_PACKAGE_FILES.json'];need(show(binding['original_publication'],binding['original_source']+'/PACKAGE_FILES.json')==oldmap,'ORIGINAL_PUBLISHED47_MAP')
        for name,digest in json.loads(oldmap).items():need(H(show(binding['original_publication'],binding['original_source']+'/'+name))==digest,'ORIGINAL_PUBLISHED47_MEMBER')
        for local,key in (('ORIGINAL_TASK.json','task_path'),('ORIGINAL_TERMINAL.json','terminal_path')):
            need(show(binding['terminal_task_publication'],binding[key])==blobs[local],'PUBLISHED_ORIGINAL_TASK_TERMINAL')
        need(H(HELPER.read_bytes())==HELPER_SHA,'EXISTING_READONLY_CONNECTION_PIN')
        ssh=runpy.run_path(str(HELPER))['ssh']
        files={n:dict(bytes=len(b),SHA256=H(b),b64=base64.b64encode(b).decode()) for n,b in blobs.items()}
        args=['reader.py','--epoch',str(epoch),'--request',request,'--publication',a.publication,'--source-sha',a.source_manifest_sha256,'--output',binding['new_root']+'/report-private.json']
        prefix=('EPOCH='+repr(epoch)+'\nREQUEST='+repr(request)+'\nPUBLICATION='+repr(a.publication)+'\nSOURCE_SHA='+repr(a.source_manifest_sha256)+'\nCLAIM_SHA='+repr(a.claim_sha256)+'\nNEW_ROOT='+repr(binding['new_root'])+'\nARGS='+repr(args)+'\nFILES='+repr(files)+'\n').encode()
        payload=prefix+blobs['receive_once.py'];need(len(payload)<=262144,'SMALL_SOURCE_PAYLOAD_CAP')
        save(LOCAL/'exec-intent.json',dict(result,attempted=True,payload_bytes=len(payload),payload_SHA256=H(payload),original_request=binding['original_request'],original_step=binding['step']))
        result['attempted']=True
        argv=[ssh[0],'-o','StrictHostKeyChecking=yes',*ssh[1:],'/usr/local/python3.12/bin/python3 -I -S -B -u -']
        done=subprocess.run(argv,input=payload,capture_output=True,timeout=remaining(5));stdout,stderr,code=done.stdout,done.stderr,done.returncode
        need(len(stdout)<=262144 and len(stderr)<=65536,'BOUNDED_TRANSPORT_OUTPUT')
        docs=[]
        for line in stdout.splitlines():
            try:v=json.loads(line)
            except (ValueError,UnicodeError):continue
            if type(v)is dict and v.get('request_id')==request and 'report' in v:docs.append(v)
        result.update(transport_rc=code,remote_state='RETURNED' if code!=255 and len(docs)==1 else 'UNKNOWN')
        if len(docs)==1:
            v=docs[0];body=(json.dumps(v['report'],sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
            need(len(body)<=131072 and len(body)==v['report_bytes'] and H(body)==v['report_SHA256'],'RETURNED_REPORT_HASH')
            with (LOCAL/'report-private.json').open('xb') as f:f.write(body)
            result.update(report_bytes=len(body),report_SHA256=H(body),corrective_audit_completed=v['report']['corrected48_schema_passed'],original_failure_retained=v['report']['prior_rejection_and_original_failure_retained'])
    except subprocess.TimeoutExpired as e:stdout,stderr=e.stdout or b'',e.stderr or b'';result.update(remote_state='UNKNOWN',error_type='TimeoutExpired')
    except BaseException as e:
        result['error_type']=type(e).__name__
        if result['attempted']:result['remote_state']='UNKNOWN'
    (LOCAL/'stdout.private').write_bytes(stdout);(LOCAL/'stderr.private').write_bytes(stderr)
    result.update(stdout_bytes=len(stdout),stdout_SHA256=H(stdout),stderr_bytes=len(stderr),stderr_SHA256=H(stderr),elapsed_s=time.monotonic()-anchor)
    save(LOCAL/'receipt.json',result);print(json.dumps(result,allow_nan=False))
    return 0 if result.get('corrective_audit_completed') is True and result['remote_state']=='RETURNED' else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--source-manifest-sha256',required=True);p.add_argument('--claim-sha256',required=True);raise SystemExit(main(p.parse_args()))
