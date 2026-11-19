"""One held ordinary Hygon SSH after fixed original3PASS, single900s through independent closure."""
import argparse,base64,hashlib,json,re,runpy,subprocess,sys,time,uuid
from pathlib import Path
from common import P,DEP,canonical,save,need
from dependency_gate import collect_original_pass
D=Path(__file__).resolve().parent;G=D.parents[2]
HELPER=Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA='3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'

def sha(b):return hashlib.sha256(b).hexdigest()

def main(a):
    epoch=time.time();mono=time.monotonic();context=dict(publication=a.publication,request_id=uuid.uuid4().hex,epoch=epoch,deadline=epoch+900,source_manifest_SHA256=a.source_manifest_sha256,dependency_snapshot_SHA256=None)
    def left(offset):
        v=min(epoch+offset-time.time(),mono+offset-time.monotonic());need(v>0,'ORIGINAL900_END');return v
    from verify_package import verify
    checked=verify(D);need(checked['root_GO'] is checked['CPU_GO'] is checked['assembly_GO'] is checked['READ_GO'] is True,'SOURCE_BLOCKED_NOT_EXECUTION_READY_ROOT_ADOPTION_REQUIRED');need(re.fullmatch('[0-9a-f]{40}',a.publication),'EXACT_NEW_PUBLICATION')
    mf=(D/'SOURCE_FILES.json').read_bytes();need(sha(mf)==a.source_manifest_sha256,'EXACT_NEW_MF')
    def git(*args):return subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G),*args],timeout=min(12,left(60)))
    rel=D.relative_to(G).as_posix();need(git('remote','get-url','origin').decode().strip()=='git@github.com:whjthu/InfiniLM-hygon-test.git','ORDINARY_ORIGIN');need(not git('diff','--cached','--name-only').strip(),'EMPTY_INDEX');need(git('rev-parse','HEAD').decode().strip()==a.publication and git('cat-file','blob',a.publication+':'+rel+'/SOURCE_FILES.json')==mf,'PUBLISHED_NEW_HEAD_AND_SOURCE');need(git('ls-remote','origin','refs/heads/bw1000-experiments').decode().split()==[a.publication,'refs/heads/bw1000-experiments'],'ORDINARY_PUBLICATION');need(not git('status','--porcelain=v1','--untracked-files=all','--',rel).strip(),'OWN_SOURCE_CLEAN')
    local=G/'local-only'/(P['label']+'-original-dispatch');local.mkdir(mode=0o700);save(local/'preflight-intent.json',dict(context,planned_files=17,planned_read_bytes=P['total_read_bytes'],planned_write_bytes=P['total_write_bytes'],original_request=DEP['original_context']['request_id'],replay_allowed=False))
    child=None;stdout=stderr=b'';host=None;failure=None;timed_out=False;forced=False;reaped=False;attempted=False;snapshot=None
    try:
        # Mandatory before helper import, any new SSH/install/Slurm or asset body. Failed original scope cannot satisfy it.
        snapshot=collect_original_pass(G);context['dependency_snapshot_SHA256']=sha(canonical(snapshot));save(local/'upstream-snapshot.json',snapshot);save(local/'intent.json',context);left(60)
        package={n:base64.b64encode((D/n).read_bytes()).decode() for n in [*json.loads(mf)['files'],'SOURCE_FILES.json']};prefix='FILES='+repr(package)+'\nCONTEXT='+repr(context)+'\nSNAPSHOT='+repr(snapshot)+'\n';body=prefix.encode()+(D/'install_once.py').read_bytes();need(len(body)<=262144,'BOUNDED_SOURCE_AND_METADATA_ONLY_BODY')
        need(sha(HELPER.read_bytes())==HELPER_SHA,'QUALIFIED_ORDINARY_HYGON_PROVIDER');ssh=runpy.run_path(str(HELPER))['ssh'];argv=[ssh[0],'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12',*ssh[1:],'/usr/local/python3.12/bin/python3 -I -S -B -'];need(len(subprocess.list2cmdline(argv))<30000,'SHORT_WINDOWS_COMMANDLINE')
        save(local/'SSH-attempt.json',dict(context,exec_request_attempted=True,body_bytes=len(body),body_SHA256=sha(body),remote_state='UNKNOWN',max_invocations=1));attempted=True;child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE);save(local/'held-SSH.json',dict(context,PID=child.pid,held_unreaped_child=True));stdout,stderr=child.communicate(input=body,timeout=left(P['budget']['client_end']))
    except subprocess.TimeoutExpired as e:timed_out=True;failure=dict(type=type(e).__name__);stdout=e.output or b'';stderr=e.stderr or b''
    except BaseException as e:failure=dict(type=type(e).__name__,reason=sha(str(e).encode()))
    finally:
        if child is not None:
            if child.poll() is None:
                forced=True
                try:child.kill()
                except ProcessLookupError:pass
            try:stdout,stderr=child.communicate(timeout=max(.001,min(4,epoch+P['budget']['PC_cleanup']-time.time())))
            except subprocess.TimeoutExpired:pass
            reaped=child.poll() is not None
    need(len(stdout)+len(stderr)<=1048576,'BOUNDED_NEW_METADATA_TRANSPORT');(local/'stdout.private').write_bytes(stdout);(local/'stderr.private').write_bytes(stderr)
    state='RETURNED' if attempted and reaped and not forced and not timed_out and child.returncode not in(None,255) else ('UNKNOWN' if attempted else 'NOT_STARTED_BLOCKED_PREREQUISITE')
    if state=='RETURNED':
        try:host=json.loads(stdout);need(all(host[k]==v for k,v in context.items()),'NEW_HOST_CONTEXT');save(local/'host-terminal.json',host)
        except (ValueError,KeyError,TypeError) as e:
            if failure is None:failure=dict(type=type(e).__name__,reason=sha(str(e).encode()))
    passed=bool(failure is None and child and child.returncode==0 and host and host['passed'] is True and state=='RETURNED')
    result=dict(context,passed=passed,failure=failure,exec_request_attempted=attempted,upstream_snapshot_passed=snapshot is not None,execution_state=state,transport=dict(returncode=child.returncode if child else None,forced_local_SSH_kill=forced,timed_out=timed_out,held_SSH_reaped=reaped),finished_epoch=time.time(),whole_seconds=time.monotonic()-mono,planned_files=17,planned_read_bytes=P['total_read_bytes'],planned_write_bytes=P['total_write_bytes'],PC_payload_files=0,old_partial_accesses=0,initial8_accesses=0,GPU_GO=False,Model_GO=False,training_GO=False,replay_allowed=False)
    save(local/'outcome.json',result);independent=None;consumer_error=None
    # Prewired for either original producer terminal or blocked/failed prerequisite, never upgrades a failed producer.
    try:
        intent=local/('intent.json' if (local/'intent.json').exists() else 'preflight-intent.json');argv=[sys.executable,'-B',str(D/'read_result.py'),'--run',str(local),'--context',str(intent),'--output',str(local/'independent-result.json')];save(local/'consumer-intent.json',dict(context,command=argv,max_invocations=1));r=subprocess.run(argv,capture_output=True,timeout=min(35,left(P['budget']['consumer_closure'])));need(len(r.stdout)+len(r.stderr)<=262144,'CONSUMER_LOG_BOUND');(local/'consumer.stdout.private').write_bytes(r.stdout);(local/'consumer.stderr.private').write_bytes(r.stderr)
        if (local/'independent-result.json').is_file():independent=json.loads((local/'independent-result.json').read_bytes())
        need(r.returncode==0 and independent and independent['independent_metadata_verified'] is True,'INDEPENDENT_NEW_SCOPE_CLOSURE')
    except BaseException as e:consumer_error=dict(type=type(e).__name__,reason=sha(str(e).encode()))
    accepted=bool(passed and independent and independent.get('independent_metadata_verified') and consumer_error is None);final=dict(context,passed=accepted,producer_passed=passed,independent_metadata_verified=bool(independent and independent.get('independent_metadata_verified')),consumer_error=consumer_error,execution_state=state,verified_files=17 if accepted else 0,verified_read_bytes=P['total_read_bytes'] if accepted else 0,verified_write_bytes=P['total_write_bytes'] if accepted else 0,whole_scope_seconds=time.monotonic()-mono,all26_qualified=False,GPU_GO=False,Model_GO=False,training_GO=False,replay_allowed=False)
    save(local/'closure.json',final);print(json.dumps(final));return 0 if accepted else (2 if not attempted else 1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--source-manifest-sha256',required=True);raise SystemExit(main(p.parse_args()))
