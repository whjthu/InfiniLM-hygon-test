"""New process: one ordinary SSH consumes original metadata only inside the original300 epoch."""
import argparse,json,math,os,subprocess,time
from pathlib import Path
from prepare_inputs import D,canonical,digest,need,verify_source
from submit_once import SSH

def main(a):
    verify_source();plan=json.loads((D/'PLAN.json').read_bytes());claim=json.loads((D/'CLAIM.json').read_bytes())
    need(claim['READ_GO'] is claim['root_GO'] is True,'PUBLISHED_READER_ADOPTION')
    need(a.run.name==plan['label']+'-original-dispatch' and a.run.parent.name=='local-only','EXACT_LOCAL_RUN')
    def read(name):
        from windows_read_identity import read_stable
        size=(a.run/name).stat().st_size;need(0<size<=262144,'BOUNDED_LOCAL_METADATA');raw,_=read_stable(a.run/name,size,lambda:None,lambda n:None)
        return json.loads(raw)
    intent=read('intent.json');receipt=read('receipt.json');epoch=intent['epoch'];need(math.isfinite(epoch) and intent['deadline']==epoch+300,'SAME_ORIGINAL300')
    for k in ['request_id','publication','epoch','deadline','source_manifest_SHA256','claim_SHA256','namespace','planned']:need(intent[k]==receipt[k],'PRODUCER_ORIGINAL_CONTEXT')
    need(intent['namespace']==plan['namespace'] and intent['planned']==plan['planned'] and intent['source_manifest_SHA256']==digest((D/'SOURCE_FILES.json').read_bytes()) and intent['claim_SHA256']==digest((D/'CLAIM.json').read_bytes()),'EXACT_PUBLISHED_READER_CONTEXT')
    folder=a.run/'consumer';folder.mkdir(mode=0o700)
    child=None;attempted=False;reaped=False;forced=False;out=b'';err=b'';error=None;remote=None;start=time.monotonic();mono_end=start+max(0,epoch+295-time.time())
    def left():
        v=min(epoch+295-time.time(),mono_end-time.monotonic());need(v>0,'ORIGINAL_READER295_END');return v
    try:
        need(receipt['exec_request_attempted'] is True,'NO_REMOTE_NAMESPACE_CREATED_NO_READER_SSH')
        need(left()>3,'INSUFFICIENT_READER_REAP_RESERVE')
        script=plan['namespace']+'/src/read_terminal.py';output=plan['namespace']+'/independent-terminal-private.json';origin=time.time()
        args=[script,'--scope',plan['namespace'],'--output',output,'--request-id',intent['request_id'],'--publication',intent['publication'],'--source-manifest-sha256',intent['source_manifest_SHA256'],'--claim-sha256',intent['claim_SHA256'],'--epoch',str(epoch),'--reader-origin',str(origin)]
        code='import sys,runpy;sys.path.insert(0,'+repr(plan['namespace']+'/src')+');sys.argv='+repr(args)+';runpy.run_path('+repr(script)+',run_name="__main__")'
        # Shell quoting, not JSON encoding; no arbitrary user text or payload is sent.
        quoted="'"+code.replace("'","'\"'\"'")+"'"
        argv=SSH[:-1]+['/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -c '+quoted]
        need(len(subprocess.list2cmdline(argv))<30000,'SHORT_READER_COMMANDLINE')
        attempt=dict(intent,reader_origin=origin,reader_end=min(epoch+295,origin+35),ordinary_SSH_attempts=1,metadata_only=True,remote_output=output,replay_allowed=False)
        with (folder/'attempt.json').open('xb') as f:f.write(canonical(attempt));f.flush();os.fsync(f.fileno())
        attempted=True;child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        out,err=child.communicate(timeout=min(35,left()))
        need(len(out)+len(err)<=1024**2,'BOUNDED_READER_OUTPUT')
        if child.returncode==0:remote=json.loads(out.decode().splitlines()[-1])
        else:raise ValueError('ORIGINAL_READER_NONZERO')
    except BaseException as e:error=dict(type=type(e).__name__,message_SHA256=digest(str(e).encode()))
    finally:
        if child is not None:
            if child.poll() is None:
                forced=True
                try:child.kill()
                except ProcessLookupError:pass
            try:out,err=child.communicate(timeout=max(.001,min(3,epoch+298-time.time())))
            except subprocess.TimeoutExpired:pass
            reaped=child.poll() is not None
        else:reaped=True
    (folder/'stdout.private').write_bytes(out);(folder/'stderr.private').write_bytes(err)
    producer_ok=receipt['state']=='RETURNED' and receipt['returncode']==0 and receipt['held_reaped'] is True and receipt['error'] is None and receipt['timeout'] is False and receipt['remote'] and receipt['remote']['passed'] is True
    passed=bool(producer_ok and not error and reaped and not forced and remote and remote['metadata_verified'] is True and remote['native_preprocessing_qualified'] is True)
    result=dict(intent,passed=passed,independent_metadata_verified=bool(remote and remote.get('metadata_verified')),native_preprocessing_qualified=bool(passed),producer_passed=bool(producer_ok),producer_failure_retained=not bool(producer_ok),reader=remote,reader_error=error,actual_reader_SSH_attempts=int(attempted),held_reader_reaped=reaped,forced_reader_kill=forced,reader_returncode=child.returncode if child else None,reader_stdout=dict(bytes=len(out),SHA256=digest(out)),reader_stderr=dict(bytes=len(err),SHA256=digest(err)),finished_epoch=time.time(),whole_scope_seconds=time.time()-epoch,GPU_GO=False,training_GO=False,Model_GO=False,Dataset_GO=False,replay_allowed=False)
    with (a.run/'independent-result.json').open('xb') as f:f.write(canonical(result));f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(passed=passed,metadata_verified=result['independent_metadata_verified'],native_preprocessing_qualified=passed,reader_SSH_attempts=int(attempted))));return 0 if passed else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);raise SystemExit(main(p.parse_args()))
