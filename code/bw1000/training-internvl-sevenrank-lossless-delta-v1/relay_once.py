"""One two-held-SSH binary bridge. PC stores only bounded source/receipt metadata."""
import argparse,base64,hashlib,json,queue,re,runpy,shlex,subprocess,threading,time,uuid,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;G=D.parents[2]
HELPER=Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA='3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
BOOTSTRAP=("import sys,os\n"
 "def exact(n):\n b=bytearray()\n while len(b)<n:\n  x=os.read(0,n-len(b))\n  if not x: raise EOFError('short script')\n  b.extend(x)\n return bytes(b)\n"
 "n=int.from_bytes(exact(8),'little')\nif not 0<n<=262144: raise ValueError('script bound')\nexec(compile(exact(n),'published-internvl-sevenrank-delta','exec'))\n")

def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,v):
    raw=(json.dumps(v,sort_keys=True,allow_nan=False)+'\n').encode()
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def main(a):
    epoch=time.time();mono=time.monotonic();request=uuid.uuid4().hex
    plan=json.loads((D/'PLAN.json').read_bytes());local=G/'local-only'/(plan['label']+'-original-dispatch');local.mkdir(mode=0o700)
    context=dict(publication=a.publication,request_id=request,epoch=epoch,deadline=epoch+900,source_manifest_SHA256=a.source_manifest_sha256)
    save(local/'intent.json',dict(context,planned_source_files=8,planned_output_files=7,patch_wire_cap_bytes=29360128,replay_allowed=False))
    def left(offset=885):
        v=min(epoch+offset-time.time(),mono+offset-time.monotonic())
        if v<=0:raise TimeoutError('ORIGINAL900_CLIENT_DEADLINE')
        return v
    need=lambda b,w: None if b else (_ for _ in ()).throw(ValueError(w))
    need(re.fullmatch('[0-9a-f]{40}',a.publication),'EXACT_PUBLICATION')
    from verify_package import verify
    checked=verify(D);need(checked['root_GO'] is checked['provision_GO'] is checked['READ_GO'] is True,'ROOT_ADOPTION_AND_PREWIRED_CONSUMER_REQUIRED')
    raw=(D/'SOURCE_FILES.json').read_bytes();need(sha(raw)==a.source_manifest_sha256,'EXACT_MF')
    def git(*args):return subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G),*args],timeout=min(12,left()))
    need(git('remote','get-url','origin').decode().strip()=='git@github.com:whjthu/InfiniLM-hygon-test.git','EXACT_ORDINARY_ORIGIN')
    need(not git('diff','--cached','--name-only').strip(),'EMPTY_INDEX')
    rel=D.relative_to(G).as_posix();need(git('cat-file','blob',a.publication+':'+rel+'/SOURCE_FILES.json')==raw,'ORDINARY_PUBLISHED_SOURCE_MF')
    need(git('rev-parse','HEAD').decode().strip()==a.publication,'PUBLISHED_HEAD')
    need(git('ls-remote','origin','refs/heads/bw1000-experiments').decode().split()==[a.publication,'refs/heads/bw1000-experiments'],'PUBLISHED_ORDINARY_REMOTE')
    need(not git('status','--porcelain=v1','--untracked-files=all','--',rel).strip(),'OWN_SOURCE_DIRTY')
    base_terminal=plan['base_provenance']['public_terminal']
    published_base=git('cat-file','blob',a.publication+':'+base_terminal['path'])
    need(len(published_base)==base_terminal['bytes'] and sha(published_base)==base_terminal['SHA256'],'PUBLISHED_ORIGINAL362_BASE_TERMINAL')
    package={name:base64.b64encode((D/name).read_bytes()).decode() for name in [*json.loads(raw)['files'],'SOURCE_FILES.json']}
    need(sha(HELPER.read_bytes())==HELPER_SHA,'QUALIFIED_ORDINARY_TRANSPORT_PROVIDER');base=runpy.run_path(str(HELPER))['ssh']
    need(time.time()<epoch+25,'PREPARATION_WITHIN_ORIGINAL_EPOCH')
    children={};readers=[];forced=set();errors=[];expired=threading.Event();ready={};terminals={};forwarded=0;members=[];patch_plan=None
    io=dict(source_read_calls=0,source_read_blocked_seconds=0.,target_write_calls=0,target_write_blocked_seconds=0.,relay_hash_calls=0,relay_hash_seconds=0.,logical_source_read_bytes=0,logical_target_write_bytes=0,bootstrap_write_bytes=0,source_control_read_bytes=0,receiver_control_read_bytes=0,source_control_write_bytes=0,receiver_control_write_bytes=0)
    def kill_owned():
        for child in list(children.values()):
            if child.poll() is None:forced.add(child.pid);child.kill()
    def timeout():expired.set();kill_owned()
    timer=threading.Timer(left(),timeout);timer.daemon=True;timer.start()
    def readline(child,cap=65536,wait_cap=60):
        q=queue.Queue()
        def take():
            try:q.put(('ok',child.stdout.readline(cap+1)))
            except BaseException as e:q.put(('error',type(e).__name__))
        t=threading.Thread(target=take,daemon=True);t.start();readers.append(t)
        kind,b=q.get(timeout=min(wait_cap,left()));need(kind=='ok' and 0<len(b)<=cap,'BOUNDED_ORIGINAL_CONTROL');mode=next(k for k,v in children.items() if v is child);io[mode+'_control_read_bytes']+=len(b);return json.loads(b)
    def start(mode):
        prefix='FILES='+repr(package)+'\nCONTEXT='+repr(context)+'\nROLE='+repr(mode)+'\n'
        code=prefix.encode()+(D/'install_endpoint.py').read_bytes();need(0<len(code)<=262144,'BOUNDED_SOURCE_BOOTSTRAP')
        if mode=='receiver':argv=[base[0],'-o','StrictHostKeyChecking=yes',*base[1:],'/usr/local/python3.12/bin/python3 -I -S -B -c '+shlex.quote(BOOTSTRAP)]
        else:argv=['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12','-F','C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config','a100-27','/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -c '+shlex.quote(BOOTSTRAP)]
        need(len(subprocess.list2cmdline(argv))<30000,'WINDOWS_CREATEPROCESS_BOUND')
        save(local/(mode+'-attempt.json'),dict(context,exec_request_attempted=True,remote_outcome='UNKNOWN',script_bytes=len(code),script_SHA256=sha(code),framing_header_bytes=8,GO_control_bytes=3,Windows_commandline_chars=len(subprocess.list2cmdline(argv)),reference_bootstrap_base64_bytes=0))
        child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE);children[mode]=child;save(local/(mode+'-held-ssh.json'),dict(context,PID=child.pid,held_unreaped_child=True))
        def drain():
            n=0;h=hashlib.sha256()
            try:
                with (local/(mode+'-stderr.private')).open('xb') as f:
                    while True:
                        b=child.stderr.read(65536)
                        if not b:break
                        n+=len(b);need(n<=262144,'STDERR_BOUND');h.update(b);f.write(b)
            except BaseException as e:errors.append(dict(mode=mode,type=type(e).__name__));kill_owned()
            finally:save(local/(mode+'-stderr-pin.json'),dict(bytes=n,SHA256=h.hexdigest()))
        t=threading.Thread(target=drain,daemon=True);t.start();readers.append(t)
        packet=len(code).to_bytes(8,'little')+code;child.stdin.write(packet);child.stdin.flush();io['bootstrap_write_bytes']+=len(packet);return child
    failure=None
    try:
        target=start('receiver');source=start('source') # Source stays held before initial FD/body reads.
        for mode in ('receiver','source'):
            r=readline(children[mode]);need(r['stage']=='READY' and r['mode']==mode and all(r[k]==v for k,v in context.items()),'EXACT_HELD_READY_CONTEXT')
            need(r['expected_source_files']==8 and r['expected_output_files']==7 and r['max_patch_wire_bytes']==29360128 and r['owner']['UID']==(1001 if mode=='source' else 154459),'EXACT_READY_SCOPE')
            if mode=='receiver':need(r['owner']['host']==plan['target_node'] and r['allocated_CPU']==4 and r['worker_CPU']==2 and len(r['CPU_affinity'])==2 and r['GRES']==0,'COMPUTE_NOT_LOGIN_RECEIVER')
            save(local/(mode+'-ready.json'),r);ready[mode]=r
        source.stdin.write(b'GO\n');source.stdin.flush();io['source_control_write_bytes']+=3;source.stdin.close()
        save(local/'release.json',dict(context,both_ready_before_source_payload=True,source_precompute_release_epoch=time.time(),payload_deadline=epoch+800))
        from delta_codec import validate_plan
        patch_plan=validate_plan(readline(source,wait_cap=left(800)));save(local/'patch-plan.json',dict(context,patch_plan=patch_plan))
        packet=b'GO\n'+(json.dumps(patch_plan,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        target.stdin.write(packet);target.stdin.flush();io['receiver_control_write_bytes']+=len(packet)
        last_bytes=0;last_time=time.monotonic()
        for row in patch_plan['frames']:
            offset=0;h=hashlib.sha256()
            while offset<row['bytes']:
                left();mark=time.monotonic();b=source.stdout.read1(min(plan['chunk_bytes'],row['bytes']-offset));io['source_read_blocked_seconds']+=time.monotonic()-mark;io['source_read_calls']+=1;need(b,'SOURCE_PAYLOAD_EOF');io['logical_source_read_bytes']+=len(b)
                mark=time.monotonic();h.update(b);io['relay_hash_seconds']+=time.monotonic()-mark;io['relay_hash_calls']+=1
                sent=0
                while sent<len(b):
                    mark=time.monotonic();n=os.write(target.stdin.fileno(),b[sent:]);io['target_write_blocked_seconds']+=time.monotonic()-mark;io['target_write_calls']+=1;need(n>0,'TARGET_WRITE_SHORT');sent+=n;io['logical_target_write_bytes']+=n
                offset+=len(b);forwarded+=len(b)
                if forwarded-last_bytes>=plan['progress_bytes'] or time.monotonic()-last_time>=plan['progress_seconds']:
                    progress=dict(context,logical_forwarded_bytes=forwarded,member=row['name'],member_offset=offset,io=io,observed_epoch=time.time(),PC_large_payload_file_bytes=0,whole_MiB_s=forwarded/1048576/max(.000001,time.monotonic()-mono),interval_MiB_s=(forwarded-last_bytes)/1048576/max(.000001,time.monotonic()-last_time))
                    temp=local/'progress.tmp';temp.write_text(json.dumps(progress),encoding='utf8');temp.replace(local/'progress.json');last_bytes=forwarded;last_time=time.monotonic()
            need(h.hexdigest()==row['SHA256'],'RELAY_COMPLETE_FRAME_SHA');members.append(dict(row))
        terminals['source']=readline(source);source.wait(timeout=left());need(source.returncode==0 and terminals['source']['passed'] is True,'SOURCE_ORIGINAL_NORMAL_TERMINAL')
        save(local/'source-terminal.json',terminals['source']);target.stdin.close()
        terminals['receiver']=readline(target,wait_cap=left());target.wait(timeout=left());save(local/'target-terminal.json',terminals['receiver'])
        need(target.returncode==0 and terminals['receiver']['passed'] is True,'TARGET_ORIGINAL_NORMAL_TERMINAL')
    except BaseException as e:failure=dict(type=type(e).__name__,reason_SHA256=sha(str(e).encode()))
    finally:
        for mode,child in children.items():
            if child.poll() is None:forced.add(child.pid);child.kill()
            try:child.wait(timeout=max(.01,min(4,epoch+895-time.time())))
            except subprocess.TimeoutExpired:pass
        timer.cancel()
        for t in readers:t.join(timeout=max(.01,min(2,epoch+895-time.time())))
    transport={mode:dict(returncode=c.returncode,forced_local_SSH_kill=c.pid in forced,remote_identity_acknowledged=mode in ready,state='RETURNED' if mode in ready and c.returncode not in(None,255) and c.pid not in forced and not expired.is_set() else 'UNKNOWN') for mode,c in children.items()}
    passed=bool(failure is None and not errors and not expired.is_set() and patch_plan is not None and forwarded==patch_plan['total_wire_bytes'] and len(members)==7 and set(transport)=={'source','receiver'} and all(v['state']=='RETURNED' and v['returncode']==0 for v in transport.values()))
    result=dict(context,passed=passed,failure=failure,stream_errors=errors,transport=transport,relay_verified_frames=members,patch_plan=patch_plan,logical_forwarded_bytes=forwarded,io=io,root_wall_seconds=time.monotonic()-mono,PC_large_payload_file_bytes=0,root_max_payload_chunk_bytes=1048576,physical_wire_and_IO_bytes='UNKNOWN',concurrent_phase_times_not_added_as_exclusive_network=True,old14_accesses=0,old_partial_accesses=0,hardlinks=0,rank00_RNG_inherited=False,GPU_GO=False,Model_GO=False,training_GO=False,all26_assets_qualified=False,replay_allowed=False,finished_epoch=time.time())
    save(local/'outcome.json',result)
    # Concrete independent process entry, already wired before any provision attempt.
    # Its metadata-only result is distinct from the original producer/transport outcome.
    independent=None;consumer_error=None
    if passed:
        argv=[sys.executable,'-B',str(D/'read_result.py'),'--run',str(local),'--context',str(local/'intent.json'),'--output',str(local/'independent-result.json')]
        save(local/'independent-consumer-intent.json',dict(context,command=argv,max_invocations=1,asset_body_reads=0))
        try:
            r=subprocess.run(argv,capture_output=True,timeout=min(30,left(895)))
            need(len(r.stdout)+len(r.stderr)<=262144,'LOCAL_CONSUMER_LOG_BOUND')
            (local/'consumer.stdout.private').write_bytes(r.stdout);(local/'consumer.stderr.private').write_bytes(r.stderr)
            need(r.returncode==0,'INDEPENDENT_CONSUMER_RETURN');independent=json.loads((local/'independent-result.json').read_bytes());need(independent['independent_metadata_verified'] is True,'INDEPENDENT_METADATA_CLOSURE')
        except BaseException as e:consumer_error=dict(type=type(e).__name__,reason_SHA256=sha(str(e).encode()))
    accepted=bool(passed and independent and independent.get('independent_metadata_verified') is True and consumer_error is None)
    final=dict(context,producer_passed=passed,independent_metadata_verified=bool(independent and independent.get('independent_metadata_verified')),passed=accepted,consumer_error=consumer_error,whole_scope_seconds=time.monotonic()-mono,verified_files=7 if accepted else 0,verified_original_bytes=1223753293 if accepted else 0,verified_patch_wire_bytes=forwarded if accepted else 0,all26_assets_qualified=False,rank00_RNG_inherited=False,GPU_GO=False,Model_GO=False,training_GO=False,replay_allowed=False)
    save(local/'closure.json',final);print(json.dumps(final));return 0 if accepted else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--source-manifest-sha256',required=True);raise SystemExit(main(p.parse_args()))
