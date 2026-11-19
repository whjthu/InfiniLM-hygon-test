"""One original24h coordinator, six held SSH, three fixed bounded compressed file streams."""
import argparse,base64,hashlib,json,os,queue,re,runpy,shlex,subprocess,sys,threading,time,uuid
from pathlib import Path
from common import P,canonical,save,need,base_context,CONTEXT_KEYS
from protocol import unpack
D=Path(__file__).resolve().parent;G=D.parents[2]
HELPER=Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA='3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
BOOTSTRAP=("import os\n"
 "def exact(n):\n b=bytearray()\n while len(b)<n:\n  x=os.read(0,n-len(b))\n  if not x: raise EOFError('short script')\n  b.extend(x)\n return bytes(b)\n"
 "n=int.from_bytes(exact(8),'little')\nif not 0<n<=262144: raise ValueError('script bound')\nexec(compile(exact(n),'published-internvl-three-shard-zlib','exec'))\n")

def sha(b):return hashlib.sha256(b).hexdigest()

def main(a):
    epoch=time.time();mono=time.monotonic();request=uuid.uuid4().hex
    context=dict(publication=a.publication,request_id=request,epoch=epoch,deadline=epoch+86400,source_manifest_SHA256=a.source_manifest_sha256,namespace_nonce=uuid.uuid4().hex)
    local=G/'local-only'/(P['label']+'-original-dispatch');local.mkdir(mode=0o700);save(local/'intent.json',dict(context,planned_files=3,planned_blocks=11590,planned_raw_bytes=P['raw_bytes'],wire_cap_bytes=P['wire_cap_bytes'],replay_allowed=False))
    def left(offset=None):
        offset=P['budget']['client_SSH_end'] if offset is None else offset;v=min(epoch+offset-time.time(),mono+offset-time.monotonic())
        if v<=0:raise TimeoutError('ORIGINAL86400_CLIENT_DEADLINE')
        return v
    from verify_package import verify
    checked=verify(D);need(checked['root_GO'] is checked['provision_GO'] is checked['READ_GO'] is True,'ROOT_ADOPTION_PREWIRED_CONSUMER_REQUIRED');need(re.fullmatch('[0-9a-f]{40}',a.publication),'EXACT_PUBLICATION')
    mf=(D/'SOURCE_FILES.json').read_bytes();need(sha(mf)==a.source_manifest_sha256,'EXACT_MF')
    def git(*args):return subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G),*args],timeout=min(12,left(P['budget']['prepare_end'])))
    need(git('remote','get-url','origin').decode().strip()=='git@github.com:whjthu/InfiniLM-hygon-test.git','ORDINARY_ORIGIN');need(not git('diff','--cached','--name-only').strip(),'EMPTY_INDEX');rel=D.relative_to(G).as_posix()
    need(git('cat-file','blob',a.publication+':'+rel+'/SOURCE_FILES.json')==mf and git('rev-parse','HEAD').decode().strip()==a.publication,'PUBLISHED_SOURCE_MF_HEAD')
    need(git('ls-remote','origin','refs/heads/bw1000-experiments').decode().split()==[a.publication,'refs/heads/bw1000-experiments'],'ORDINARY_PUBLISHED_REMOTE');need(not git('status','--porcelain=v1','--untracked-files=all','--',rel).strip(),'OWN_SOURCE_CLEAN')
    package={name:base64.b64encode((D/name).read_bytes()).decode() for name in [*json.loads(mf)['files'],'SOURCE_FILES.json']}
    need(sha(HELPER.read_bytes())==HELPER_SHA,'QUALIFIED_ORDINARY_SSH_PROVIDER');base=runpy.run_path(str(HELPER))['ssh'];left(P['budget']['prepare_end'])
    children={};contexts={};ready={};forced=set();readers=[];workers=[];abort=threading.Event();expired=threading.Event();failures=[];arms={};lock=threading.Lock()
    for row in P['files']:
        fid=row['file_id'];arm=local/('shard%02d'%fid);arm.mkdir();ctx=dict(context,file_id=fid);contexts[fid]=ctx;save(arm/'intent.json',dict(ctx,planned_files=1,planned_blocks=row['blocks'],planned_raw_bytes=row['bytes'],replay_allowed=False))
        arms[fid]=dict(passed=False,raw_forwarded_bytes=0,wire_forwarded_bytes=0,blocks_forwarded=0,wire_SHA256=None,io=dict(source_read_bytes=0,target_write_bytes=0,source_read_calls=0,target_write_calls=0,source_read_blocked_seconds=0.,target_write_blocked_seconds=0.,compressed_hash_bytes=0,compressed_hash_seconds=0.,wire_hash_bytes=0,wire_hash_seconds=0.,bootstrap_write_bytes=0,source_control_read_bytes=0,receiver_control_read_bytes=0,source_control_write_bytes=0,receiver_control_write_bytes=0,progress_persist_seconds=0.,progress_persist_calls=0),failure=None)
    def kill_owned():
        abort.set()
        for child in list(children.values()):
            if child.poll() is None:
                with lock:forced.add(child.pid)
                try:child.kill()
                except ProcessLookupError:pass
    def timeout():expired.set();kill_owned()
    timer=threading.Timer(left(),timeout);timer.daemon=True;timer.start()
    def control_line(fid,mode,wait):
        child=children[(fid,mode)];q=queue.Queue()
        def take():
            try:q.put(('ok',child.stdout.readline(65537)))
            except BaseException as e:q.put(('error',type(e).__name__))
        t=threading.Thread(target=take,daemon=True);t.start();readers.append(t);kind,raw=q.get(timeout=min(wait,left()));need(kind=='ok' and 0<len(raw)<=65536,'BOUNDED_ORIGINAL_CONTROL');arms[fid]['io'][mode+'_control_read_bytes']+=len(raw);return json.loads(raw)
    def start(row,mode):
        fid=row['file_id'];ctx=contexts[fid];arm=local/('shard%02d'%fid);prefix='FILES='+repr(package)+'\nCONTEXT='+repr(ctx)+'\nROLE='+repr(mode)+'\n';code=prefix.encode()+(D/'install_endpoint.py').read_bytes();need(0<len(code)<=262144,'SMALL_STDIN_SOURCE_SCRIPT')
        if mode=='receiver':argv=[base[0],'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12',*base[1:],'/usr/local/python3.12/bin/python3 -I -S -B -c '+shlex.quote(BOOTSTRAP)]
        else:argv=['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12','-F','C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config','a100-27','/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -c '+shlex.quote(BOOTSTRAP)]
        chars=len(subprocess.list2cmdline(argv));need(chars<30000,'WINDOWS_CREATEPROCESS_BOUND');save(arm/(mode+'-attempt.json'),dict(ctx,exec_request_attempted=True,remote_state='UNKNOWN',script_bytes=len(code),script_SHA256=sha(code),framing_header_bytes=8,commandline_chars=chars,replay_allowed=False))
        child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE);children[(fid,mode)]=child;save(arm/(mode+'-held-SSH.json'),dict(ctx,PID=child.pid,held_unreaped_child=True))
        def stderr_drain():
            total=0;stored=0;h=hashlib.sha256()
            try:
                with (arm/(mode+'-stderr.private')).open('xb') as out:
                    while True:
                        b=child.stderr.read(65536)
                        if not b:break
                        total+=len(b);need(total<=262144,'BOUNDED_STDERR');h.update(b);out.write(b);stored+=len(b)
            except BaseException as e:
                with lock:failures.append(dict(file_id=fid,mode=mode,type=type(e).__name__))
                kill_owned()
            finally:save(arm/(mode+'-stderr-pin.json'),dict(bytes=stored,observed_stream_bytes=total,SHA256=h.hexdigest()))
        t=threading.Thread(target=stderr_drain,daemon=True);t.start();readers.append(t);packet=len(code).to_bytes(8,'little')+code;need(child.stdin.write(packet)==len(packet),'BOOTSTRAP_SHORT_WRITE');child.stdin.flush();arms[fid]['io']['bootstrap_write_bytes']+=len(packet)
    def receive_ready(row,mode):
        fid=row['file_id'];r=control_line(fid,mode,left(P['budget']['all_install_admission_end']));need(r['stage']=='READY' and r['mode']==mode and all(r[k]==v for k,v in contexts[fid].items()),'ORIGINAL_READY_CONTEXT')
        need(r['body_read_before_GO'] is False and r['planned_blocks']==row['blocks'] and r['planned_raw_bytes']==row['bytes'] and r['group_files']==3 and r['worker_CPU']==2,'FIXED_READY_DENOMINATOR')
        if mode=='source':need(r['owner']['UID']==1001 and r['CPU_affinity']==row['source_CPU_affinity'] and not set(r['CPU_affinity'])&set(range(10)),'SOURCE_FIXED10_TO15_AVOID_EVAL2_TO9')
        else:
            need(r['owner']['UID']==154459 and r['owner']['host']==row['node'] and r['allocated_CPU']==4 and r['worker_CPU']==2 and len(r['CPU_affinity'])==2 and not set(r['CPU_affinity'])&{0,1} and r['GRES']==0,'ACTUAL_FIXED_COMPUTE_READY')
            adm=r['fresh_admission'];need(adm['parent']['lease_end_epoch']>=epoch+86400 and adm['actual_end_epoch']<=epoch+85020 and adm['actual_limit_seconds']==adm['requested_seconds'],'FRESH_LEASE_AND_ACTUAL_STEP_END')
        save(local/('shard%02d'%fid)/(mode+'-ready.json'),r);ready[(fid,mode)]=r
    def forward_file(row):
        fid=row['file_id'];arm=local/('shard%02d'%fid);s=children[(fid,'source')];t=children[(fid,'receiver')];aresult=arms[fid];io=aresult['io'];h=hashlib.sha256();last=0;last_t=time.monotonic()
        def exact(n):
            out=bytearray()
            while len(out)<n:
                left();need(not abort.is_set(),'GROUP_ABORT');mark=time.monotonic();b=s.stdout.read1(min(P['chunk_bytes'],n-len(out)));io['source_read_blocked_seconds']+=time.monotonic()-mark;io['source_read_calls']+=1;io['source_read_bytes']+=len(b);need(b,'COMPRESSED_SOURCE_SHORT_EOF');out.extend(b)
            return bytes(out)
        def put(b):
            position=0
            while position<len(b):
                left();need(not abort.is_set(),'GROUP_ABORT');mark=time.monotonic();n=os.write(t.stdin.fileno(),memoryview(b)[position:position+P['chunk_bytes']]);io['target_write_blocked_seconds']+=time.monotonic()-mark;io['target_write_calls']+=1;need(n>0,'TARGET_STDIN_SHORT_WRITE');position+=n;io['target_write_bytes']+=n
        try:
            for index in range(row['blocks']):
                header=exact(128);meta=unpack(header,row,index);comp=exact(meta['compressed_bytes']);mark=time.monotonic();digest=hashlib.sha256(comp).digest();io['compressed_hash_seconds']+=time.monotonic()-mark;io['compressed_hash_bytes']+=len(comp);need(digest==meta['compressed_digest'],'RELAY_COMPRESSED_BLOCK_SHA')
                for b in (header,comp):
                    mark=time.monotonic();h.update(b);io['wire_hash_seconds']+=time.monotonic()-mark;io['wire_hash_bytes']+=len(b);put(b)
                aresult['wire_forwarded_bytes']+=128+len(comp);aresult['raw_forwarded_bytes']+=meta['raw_bytes'];aresult['blocks_forwarded']+=1
                if time.monotonic()-last_t>=P['progress_seconds'] or aresult['wire_forwarded_bytes']-last>=P['progress_bytes']:
                    mark=time.monotonic()
                    try:save(arm/'progress.json',dict(contexts[fid],observed_epoch=time.time(),state=aresult,PC_payload_file_bytes=0),replace=True)
                    finally:io['progress_persist_seconds']+=time.monotonic()-mark;io['progress_persist_calls']+=1
                    last_t=time.monotonic();last=aresult['wire_forwarded_bytes']
            aresult['wire_SHA256']=h.hexdigest();src=control_line(fid,'source',left());s.wait(timeout=left());save(arm/'source-terminal.json',src)
            need(s.returncode==0 and src['passed'] is True and src['raw_SHA256']==row['SHA256'] and src['wire_SHA256']==h.hexdigest() and src['wire_processed_bytes']==aresult['wire_forwarded_bytes'],'SOURCE_NORMAL_FULL_WHOLE_SHA_AND_WIRE')
            # EOF is released only after source original wholeSHA/stability/normal termination.
            t.stdin.close();host=control_line(fid,'receiver',left());t.wait(timeout=left());save(arm/'target-terminal.json',host);target=host.get('endpoint')
            need(t.returncode==0 and host['passed'] is True and target['passed'] is True and target['raw_SHA256']==row['SHA256'] and target['wire_SHA256']==h.hexdigest() and target['wire_processed_bytes']==aresult['wire_forwarded_bytes'],'COMPUTE_NORMAL_FULL_WHOLE_SHA_AND_WIRE')
            aresult['passed']=True
        except BaseException as e:aresult['failure']=dict(type=type(e).__name__,reason_SHA256=sha(str(e).encode()));kill_owned()
        finally:
            aresult['wire_SHA256']=h.hexdigest();aresult['finished_epoch']=time.time();save(arm/'relay-outcome.json',dict(contexts[fid],**aresult,replay_allowed=False))
    preflight_failure=None
    try:
        first=P['files'][0];need(first['file_id']==2,'FIRST_RECEIVER_FIXED2');start(first,'receiver');receive_ready(first,'receiver')
        for row in P['files'][1:]:start(row,'receiver')
        for row in P['files']:start(row,'source')
        for row in P['files'][1:]:receive_ready(row,'receiver')
        for row in P['files']:receive_ready(row,'source')
        need(set(ready)=={(r['file_id'],mode) for r in P['files'] for mode in ['source','receiver']},'ALL_SIX_READY_BEFORE_ANY_BODY_GO')
        srcs=[ready[(r['file_id'],'source')] for r in P['files']];need(len({x['owner']['host'] for x in srcs})==len({x['owner']['boot_id'] for x in srcs})==1,'SAME_SOURCE_HOST_BOOT')
        need(set().union(*(set(x['CPU_affinity']) for x in srcs))==set(range(10,16)) and sum(len(x['CPU_affinity']) for x in srcs)==6,'SOURCE6CPU_PAIRWISE_DISJOINT_FIXED_MASKS')
        marks=[ready[(r['file_id'],'receiver')]['common_marker'] for r in P['files']];need(all(m['owner_UID']==154459 and m['stable_FD_path'] is True and m['stat']==m['stat_after'] and all(m['observer'][k]==ready[(r['file_id'],'receiver')]['owner'][k] for k in ['UID','host','boot_id','PID','startticks']) for r,m in zip(P['files'],marks)),'EACH_COMPUTE_MARKER_OBSERVER_STABLE_OWNER');need(len({(m['SHA256'],m['bytes'],m['owner_UID']) for m in marks})==1,'SAME_UNPREDICTABLE_CONTEXT_MARKER_OWNER')
        left(P['budget']['all_install_admission_end']);save(local/'release.json',dict(context,all_six_READY_before_any_body_GO=True,source_masks=[x['CPU_affinity'] for x in srcs],shared_marker=marks[0],release_epoch=time.time()))
        for row in P['files']:
            fid=row['file_id'];save(local/('shard%02d'%fid)/'release.json',dict(contexts[fid],all_six_READY_before_any_body_GO=True,source_masks=[x['CPU_affinity'] for x in srcs],release_epoch=time.time()))
        for mode in ['receiver','source']:
            for row in P['files']:
                fid=row['file_id'];child=children[(fid,mode)];need(child.stdin.write(b'GO\n')==3,'EXACT_GO3');child.stdin.flush();arms[fid]['io'][mode+'_control_write_bytes']+=3
                if mode=='source':child.stdin.close()
        for row in P['files']:
            t=threading.Thread(target=forward_file,args=(row,),daemon=True);t.start();workers.append(t)
        for t in workers:t.join(timeout=left())
        need(not any(t.is_alive() for t in workers),'BOUNDED_FILE_WORKERS_CLOSED')
    except BaseException as e:preflight_failure=dict(type=type(e).__name__,reason_SHA256=sha(str(e).encode()));kill_owned()
    finally:
        for child in children.values():
            if child.poll() is None:
                forced.add(child.pid)
                try:child.kill()
                except ProcessLookupError:pass
            try:child.wait(timeout=max(.001,min(4,epoch+P['budget']['PC_cleanup']-time.time())))
            except subprocess.TimeoutExpired:pass
        timer.cancel()
        for t in workers+readers:t.join(timeout=max(.001,min(2,epoch+P['budget']['PC_cleanup']-time.time())))
    transport={str(fid)+'-'+mode:dict(returncode=c.returncode,forced_local_SSH_kill=c.pid in forced,held_SSH_reaped=c.returncode is not None,state='RETURNED' if (fid,mode) in ready and c.returncode not in(None,255) and c.pid not in forced and not expired.is_set() else 'UNKNOWN') for (fid,mode),c in children.items()}
    passed=bool(preflight_failure is None and not failures and not expired.is_set() and len(transport)==6 and all(a['passed'] for a in arms.values()) and all(v['state']=='RETURNED' and v['returncode']==0 for v in transport.values()))
    result=dict(context,passed=passed,preflight_failure=preflight_failure,stream_errors=failures,transport=transport,arms={str(k):v for k,v in arms.items()},planned_files=3,completed_files=sum(a['passed'] for a in arms.values()),planned_blocks=11590,blocks_forwarded=sum(a['blocks_forwarded'] for a in arms.values()),raw_forwarded_bytes=sum(a['raw_forwarded_bytes'] for a in arms.values()),wire_forwarded_bytes=sum(a['wire_forwarded_bytes'] for a in arms.values()),finished_epoch=time.time(),root_wall_seconds=time.monotonic()-mono,PC_payload_file_bytes=0,physical_wire_bytes='UNKNOWN',old14_accesses=0,initial362_364_accesses=0,old_partial_accesses=0,assembly_actions=0,GPU_GO=False,Model_GO=False,training_GO=False,replay_allowed=False)
    save(local/'outcome.json',result)
    independent=None;consumer_error=None
    # Concrete new process is wired for producer PASS or FAIL. Failed/unknown producer stays separate and cannot be upgraded.
    try:
        left(P['budget']['consumer_closure']);argv=[sys.executable,'-B',str(D/'read_result.py'),'--run',str(local),'--context',str(local/'intent.json'),'--output',str(local/'independent-result.json')];save(local/'independent-consumer-intent.json',dict(context,command=argv,max_invocations=1,asset_body_reads=0,fresh_exact_step_queries_max=3))
        r=subprocess.run(argv,capture_output=True,timeout=min(100,left(P['budget']['consumer_closure'])));need(len(r.stdout)+len(r.stderr)<=262144,'INDEPENDENT_CONSUMER_LOG_BOUND');(local/'consumer.stdout.private').write_bytes(r.stdout);(local/'consumer.stderr.private').write_bytes(r.stderr)
        if (local/'independent-result.json').is_file():independent=json.loads((local/'independent-result.json').read_bytes())
        need(r.returncode==0 and independent and independent['independent_metadata_verified'] is True,'INDEPENDENT_ORIGINAL_CLOSURE')
    except BaseException as e:consumer_error=dict(type=type(e).__name__,reason_SHA256=sha(str(e).encode()))
    accepted=bool(passed and independent and independent.get('independent_metadata_verified') and consumer_error is None)
    final=dict(context,passed=accepted,producer_passed=passed,independent_metadata_verified=bool(independent and independent.get('independent_metadata_verified')),consumer_error=consumer_error,planned_files=3,verified_files=3 if accepted else 0,producer_completed_files=result['completed_files'],planned_blocks=11590,producer_blocks_forwarded=result['blocks_forwarded'],verified_raw_bytes=P['raw_bytes'] if accepted else 0,producer_wire_bytes=result['wire_forwarded_bytes'],whole_scope_seconds=time.monotonic()-mono,old14_accesses=0,initial362_364_accesses=0,assembly_actions=0,GPU_GO=False,Model_GO=False,training_GO=False,replay_allowed=False)
    save(local/'closure.json',final);print(json.dumps(final));return 0 if accepted else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--source-manifest-sha256',required=True);raise SystemExit(main(p.parse_args()))
