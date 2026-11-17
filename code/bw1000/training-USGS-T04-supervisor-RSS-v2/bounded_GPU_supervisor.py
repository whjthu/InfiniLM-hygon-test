"""One prospective A100 T04 native arm. Shared framework owns all training."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import socket
import stat
import subprocess
import sys
import time
import traceback
from deadline_contract import schedule
from held_child import cleanup_held
from owned_processes import snapshot, refresh, cleanup

D = Path(__file__).resolve().parent
CAP = 2 * 1024**2
BOOTSTRAP = "import os,sys; token=sys.stdin.buffer.readline(4); assert token==b'GO\\n'; os.execv(sys.executable,[sys.executable]+sys.argv[1:])"


def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, 'DUPLICATE_JSON_KEY')
            value[key] = item
        return value
    def bad(_):
        raise ValueError('NONFINITE_JSON')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def stable(path, cap=CAP, expected=None):
    path = Path(path)
    before = path.stat()
    need(not path.is_symlink() and stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'BOUNDED_REGULAR_FILE')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno()); raw = stream.read(cap + 1); after = os.fstat(stream.fileno())
    key = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    need(key(before) == key(opened) == key(after) == key(path.stat()) and len(raw) == before.st_size, 'STABLE_FILE')
    pin = dict(path=str(path), bytes=len(raw), SHA256=sha(raw))
    if expected is not None:
        need(pin['bytes'] == expected['bytes'] and pin['SHA256'] == expected['SHA256'], 'FILE_PIN')
    return raw, pin


def save(root, name, value, *, replace=False):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    need(len(raw) <= CAP, 'RECORD_BOUND')
    path = root / name
    target = root / (name + '.tmp') if replace else path
    with target.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    if replace:
        target.replace(path)
    fd = os.open(str(root), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(path), bytes=len(raw), SHA256=sha(raw))


def own_identity(request):
    row = snapshot().get(os.getpid())
    need(row is not None, 'SUPERVISOR_IDENTITY')
    return dict(PID=os.getpid(), startticks=row['startticks'], uid=os.getuid(), host=socket.gethostname(),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), scope_request_id=request)


def source_gate(manifest_sha, claim_sha):
    raw, _ = stable(D / 'SOURCE_FILES.json', 256 * 1024)
    need(sha(raw) == manifest_sha, 'PUBLISHED_SOURCE_MANIFEST')
    files = decode(raw)['files']
    need(set(files) == {'bounded_GPU_supervisor.py','deadline_contract.py','held_child.py','owned_processes.py',
                        'CLAIM.json','SOURCE_PROOF.json','README.md','check_supervisor_CPU.py','check_rss_CPU.py'}, 'SOURCE_MEMBERS')
    for name, pin in files.items():
        stable(D / name, CAP, pin)
    raw, _ = stable(D / 'CLAIM.json')
    need(sha(raw) == claim_sha, 'PUBLISHED_CLAIM')
    claim = decode(raw)
    need(claim['root_GO'] is True and claim['GPU_GO'] is True and claim['max_submissions'] == 1
         and claim['case'] == 'T04.sft' and claim['whole_seconds'] == 900, 'PUBLISHED_ACTUAL_ADOPTION_REQUIRED')
    return files


def template_gate(c, args):
    need(c['request_id'] == args.request_id and c['epoch'] == args.epoch and c['source_publication'] == args.publication,
         'TEMPLATE_ORIGINAL_CONTEXT')
    need(c['mode'] in ('fresh25','resume20to25') and c['arm'] in ('domain','generic')
         and c['case'] == 'T04.sft' and c['root_GO'] is True and c['GPU_GO'] is True
         and c['device'] == 'cuda' and c['formal17'] is False, 'ACTUAL_ARM_ADOPTION')
    need(c['output'] == str(args.scope / 'witness') and c['train_output'] == str(args.scope / 'train'), 'OWN_OUTPUT')
    need(c['deadline'] == args.epoch + 820 and c['maximum_seconds'] == 820, 'ONE_NATIVE_ORIGIN')
    need(c['world_size'] == 8 and c['max_steps'] == 25 and c['physical_rows'] == 800, 'FIXED_DENOMINATOR')
    need(len(args.request_id) == 32 and set(args.request_id) <= set('0123456789abcdef')
         and len(args.publication) == 40 and set(args.publication) <= set('0123456789abcdef'), 'CONTEXT_ENCODING')


def verify_framework(c, root):
    raw, _ = stable(c['framework_manifest']['path'], 256 * 1024, c['framework_manifest'])
    files = decode(raw)['files']
    need(len(files) == c['framework_members'] and len(files) >= 400, 'COMPLETE_MERGED_FRAMEWORK')
    for name, pin in files.items():
        rel = PurePosixPath(name)
        need(not rel.is_absolute() and '..' not in rel.parts, 'FRAMEWORK_PATH')
        stable(root / name, 4 * 1024**2, pin)
    need({'tools/hygon_training_recipe.py','tools/usgs_domain_launch.py','vlm/runtime/usgs_domain_contract.py',
          'vlm/runtime/usgs_domain_witness.py','vlm/plugins/usgs_domain_training.py'} <= set(files), 'NEW_DOMAIN_MEMBERS')


def verify_model(c, left):
    raw, pin = stable(c['model_admission']['path'], CAP, c['model_admission'])
    model = decode(raw); root = Path(model['model_root']); verified = []
    need(root.is_absolute() and 0 < len(model['files']) <= 32, 'MODEL_INVENTORY')
    for name, expected in sorted(model['files'].items()):
        left(c)
        rel = PurePosixPath(name)
        need(not rel.is_absolute() and '..' not in rel.parts, 'MODEL_MEMBER_PATH')
        path = root / name
        if name.endswith('.safetensors'):
            current = path.stat()
            need(not path.is_symlink() and stat.S_ISREG(current.st_mode), 'ORIGINAL_WEIGHT_REGULAR')
            need(dict(bytes=current.st_size, mtime_ns=current.st_mtime_ns, inode=current.st_ino, device=current.st_dev)
                 == model['weight_stats'][name] and current.st_size == expected['bytes'], 'ORIGINAL_WHOLE_WEIGHT_STAT_CONTINUITY')
            need(len(expected['SHA256']) == 64, 'ORIGINAL_WHOLE_WEIGHT_PIN')
        else:
            stable(path, 16 * 1024**2, expected)
        verified.append(name)
    return dict(model_inventory=pin, members=verified, whole_weight_SHA_reused_from_published_inventory=True,
                weight_payload_read_by_supervisor=False)


def rank_result(row, rank, c):
    micros, updates = (100,25) if c['mode'] == 'fresh25' else (20,5)
    need(row['rank'] == rank and row['mode'] == c['mode'] and row['arm'] == c['arm'] and row['pair_id'] == c['pair_id'], 'RANK_CONTEXT')
    need(row['completed_microbatches'] == micros and row['updates'] == updates and row['completed_step'] == 25, 'RANK_DENOMINATOR')
    need(row['formal17'] is False and row['performance_qualified'] is False and row['effect_qualified'] is False, 'RANK_SCOPE')
    need(row['recovery_exact'] is (None if c['mode'] == 'fresh25' else True), 'RANK_RECOVERY')
    return {k: row[k] for k in ['rank','mode','arm','pair_id','completed_microbatches','updates','completed_step','recovery_exact']}


def entry_stages(rows):
    stages=[row['stage'] for row in rows]
    need(stages==['planned','owned-cleanup-contract','fresh-admission','runtime','running','exited']
         and rows[-1]['returncode']==0,'ENTRY_STAGES')
    actual=rows[1]['actual']
    need(set(actual)=={'held_session_identity_verified','cleanup_helpers_verified','resource_release_permitted'}
         and actual['held_session_identity_verified'] is True and actual['cleanup_helpers_verified'] is True
         and actual['resource_release_permitted'] is False,'ENTRY_OWNED_CLEANUP_CONTRACT')
    return stages


RSS_SOFT_BYTES = 256 * 1024**3


def affinity_contract(cpus):
    cpus=sorted(cpus)
    need(2<=len(cpus)<=64 and len(set(cpus))==len(cpus) and all(type(c)is int and c>=2 for c in cpus),
         'INHERITED_AFFINITY_EXCLUDES_0_1_BOUND')
    return cpus


def rss_record(saved, before=None, status=None, after=None, *, classification, value=None):
    def phase(row):
        if row is None:
            return None
        return {key:row.get(key) for key in ('PID','startticks','state')}
    return dict(PID=saved['PID'],startticks=saved['startticks'],before=phase(before),
                status=status,after=phase(after),classification=classification,RSS_bytes=value)


def checked_owned_rss(saved, before, status_reader, after_reader):
    # No status read for a PID that no longer belongs to this registry identity.
    match=lambda row: row['PID']==saved['PID'] and row['startticks']==saved['startticks']
    if not match(before):
        return rss_record(saved,before,classification='IDENTITY_CHANGED_BEFORE')
    if before['state'] in ('Z','X'):
        return rss_record(saved,before,classification='EXITED_BEFORE')
    raw=status_reader()
    lines=raw.splitlines()
    states=[line for line in lines if line.startswith(b'State:')]
    rows=[line for line in lines if line.startswith(b'VmRSS:')]
    status_state=None
    if len(states)==1:
        parts=states[0].split()
        if len(parts)>=2 and len(parts[1])==1 and parts[1] in (b'R',b'S',b'D',b'Z',b'T',b't',b'X',b'I',b'P',b'K',b'W'):
            status_state=parts[1].decode('ascii')
    status=dict(state=status_state,State_row_count=len(states),VmRSS_row_count=len(rows),
                bytes=len(raw),SHA256=sha(raw),State_rows_SHA256=sha(b'\n'.join(states)),VmRSS_rows_SHA256=sha(b'\n'.join(rows)))
    after=after_reader()
    if not match(after):
        return rss_record(saved,before,status,after,classification='IDENTITY_CHANGED_AFTER')
    # A present malformed metric is never silently accepted, even during exit.
    if len(rows)>1:
        return rss_record(saved,before,status,after,classification='MALFORMED_VMRSS_DUPLICATE')
    value=None
    if rows:
        parts=rows[0].split()
        if len(parts)!=3 or parts[1].isdigit() is False or parts[2]!=b'kB':
            return rss_record(saved,before,status,after,classification='MALFORMED_VMRSS_VALUE_UNIT')
        value=int(parts[1])*1024
    terminal=after['state'] in ('Z','X') or status_state in ('Z','X')
    if terminal:
        # Do not manufacture a 0-byte measurement for an exited process.
        return rss_record(saved,before,status,after,classification='EXITED_DURING_SAMPLE')
    if status_state is None:
        return rss_record(saved,before,status,after,classification='MALFORMED_STATUS_STATE')
    if not rows:
        return rss_record(saved,before,status,after,classification='LIVE_MISSING_RSS')
    return rss_record(saved,before,status,after,classification='MEASURED',value=value)


def read_owned_rss(saved):
    directory=None
    before=None
    try:
        directory=os.open('/proc/'+str(saved['PID']),os.O_RDONLY|os.O_DIRECTORY)
        need(os.fstat(directory).st_uid==os.getuid(),'OWNED_PROC_UID')
        def read(name,cap):
            fd=os.open(name,os.O_RDONLY,dir_fd=directory)
            try:
                with os.fdopen(fd,'rb',closefd=False) as stream:
                    raw=stream.read(cap+1)
                need(len(raw)<=cap,'OWNED_PROC_RECORD_BOUND')
                return raw
            finally:
                os.close(fd)
        def identity():
            raw=read('stat',8192).decode('ascii')
            fields=raw.rsplit(') ',1)[1].split()
            return dict(PID=int(raw.split(' ',1)[0]),startticks=int(fields[19]),state=fields[0])
        before=identity()
        return checked_owned_rss(saved,before,lambda:read('status',65536),identity)
    except (FileNotFoundError,ProcessLookupError):
        return rss_record(saved,before,classification='ABSENT_DURING_SAMPLE')
    finally:
        if directory is not None:
            os.close(directory)


def sample_group_rss(registry,resources,reader=read_owned_rss):
    total=0;sampled=0;unavailable=0;unknown=0;failures=[];counts={}
    pending=resources.setdefault('RSS_missing_pending',{})
    diagnostics=resources.setdefault('RSS_diagnostics',[])
    seen=resources.setdefault('RSS_missing_seen',{})
    for saved in registry.values():
        row=reader(saved)
        need(row['PID']==saved['PID'] and row['startticks']==saved['startticks'],'RSS_READER_OWNER')
        key=str(saved['PID'])+':'+str(saved['startticks'])
        kind=row['classification']
        if kind=='LIVE_MISSING_RSS':
            unknown+=1
            if key in seen:
                kind='LIVE_MISSING_RSS_PERSISTENT' if key in pending else 'LIVE_MISSING_RSS_ALLOWANCE_USED'
                row['classification']=kind
                failures.append(kind)
            else:
                need(len(seen)<1024,'RSS_PENDING_OWNER_BOUND')
                pending[key]=dict(first_sample=resources['samples']+1,confirmations_allowed=1)
                seen[key]=resources['samples']+1
        elif kind=='MEASURED':
            pending.pop(key,None)
            value=row['RSS_bytes'];need(type(value)is int and value>=0,'OWNED_RSS_VALUE')
            total+=value;sampled+=1
        elif kind in ('EXITED_BEFORE','EXITED_DURING_SAMPLE','ABSENT_DURING_SAMPLE','IDENTITY_CHANGED_BEFORE','IDENTITY_CHANGED_AFTER'):
            pending.pop(key,None);unavailable+=1
        else:
            unknown+=1;failures.append(kind)
        counts[kind]=counts.get(kind,0)+1
        if kind!='MEASURED':
            diagnostics.append(dict(sample=resources['samples']+1,**row))
            if len(diagnostics)>128:
                del diagnostics[:-128]
            resources['RSS_diagnostic_events']=resources.get('RSS_diagnostic_events',0)+1
            if kind in failures:
                resources['RSS_last_failure']=dict(sample=resources['samples']+1,**row)
    resources['samples']+=1
    resources['sampled_group_peak_RSS_bytes']=max(resources['sampled_group_peak_RSS_bytes'],total)
    resources['RSS_peak_is_observed_lower_bound']=True
    if unknown:
        resources['RSS_unknown_coverage_samples']=resources.get('RSS_unknown_coverage_samples',0)+1
    resources['last_sample']=dict(group_RSS_bytes=None if unknown else total,known_group_RSS_bytes=total,
        coverage_complete=unknown==0,unknown_live_processes=unknown,matched_processes=sampled,
        exited_absent_or_identity_changed=unavailable,classifications=counts,observed_monotonic=time.monotonic())
    resources['soft_limit_exceeded'] |= total>RSS_SOFT_BYTES
    need(not failures,'OWNED_RSS_SCHEMA_OR_PERSISTENT_MISSING')
    return total<=RSS_SOFT_BYTES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['template','framework','scope','a100-launch','swift','models','plugins']:
        p.add_argument('--' + name, required=True, type=Path)
    for name in ['template-sha256','source-manifest-sha256','claim-sha256','request-id','publication']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--epoch', required=True, type=float)
    p.add_argument('--master-port', required=True, type=int)
    a = p.parse_args()
    clock = schedule(a.epoch, time.time())
    # Epoch supplied by root includes original upload/publication gate; never begins here.
    files = source_gate(a.source_manifest_sha256, a.claim_sha256)
    raw, template_pin = stable(a.template)
    need(sha(raw) == a.template_sha256, 'PUBLISHED_ACTUAL_TEMPLATE')
    c = decode(raw); template_gate(c, a)
    root = a.scope
    need(all(path.is_absolute() for path in [root,a.framework,a.template,a.a100_launch,a.swift,a.models,a.plugins]), 'ABSOLUTE_PATHS')
    need(root.parent.is_dir() and root.parent.resolve() == root.parent and not root.is_symlink() and not root.exists(), 'EXCLUSIVE_SCOPE_NO_REPLAY')
    need(time.time() < clock['native_deadline'] - 60, 'ADMISSION_RESERVE')
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    control = dict(request_id=a.request_id, publication=a.publication, epoch=a.epoch, deadline=clock['deadline'],
                   source_manifest_SHA256=a.source_manifest_sha256, claim_SHA256=a.claim_sha256,
                   exec_request_attempted=False, entry_released=False)
    registry = {}; child = None; anchor = None; failure = None; post = None; domain = None; entry = None
    timed_out = False; receipts = {}; receipt_errors = {}; clean = {'owned_processes_remaining':None,'termination_needed':False}
    def expired(*_):
        raise TimeoutError('ORIGINAL_SUPERVISOR_EPOCH_OR_SIGNAL')
    signal.signal(signal.SIGALRM, expired); signal.signal(signal.SIGTERM, expired); signal.signal(signal.SIGINT, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.01, a.epoch + 890 - time.time()))
    resources=dict(inherited_affinity=None,samples=0,sampled_group_peak_RSS_bytes=0,
                   soft_limit_bytes=RSS_SOFT_BYTES,soft_limit_exceeded=False,hard_limit=False,
                   method='Sum measured VmRSS for owned PID/startticks; live missing is unknown and one subsequent poll only; exited unsampled; shared pages may double count; not cgroup peak',last_sample=None)
    phase = 'admission'
    try:
        save(root, 'control-state.json', control)
        save(root, 'install-intent.json', dict(template=template_pin, **control))
        resources['inherited_affinity']=affinity_contract(os.sched_getaffinity(0))
        save(root,'resource-samples.json',resources)
        verify_framework(c, a.framework)
        sys.path.insert(0, str(a.framework)); sys.path.insert(0, str(a.framework / 'tools'))
        from vlm.runtime import usgs_domain_contract as domain
        import usgs_domain_launch as entry
        domain.check_config(c); domain.load_inputs(c); domain.verify_provider_files(c)
        save(root, 'assets-admission.json', verify_model(c, domain.remaining))
        c['supervisor'] = own_identity(a.request_id)
        source_pins = {name: dict(path=str(D/name), **files[name]) for name in ['bounded_GPU_supervisor.py','held_child.py','owned_processes.py']}
        env = os.environ.copy()
        need(not any(k in env for k in ['RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK']), 'UNRANKED_SUPERVISOR')
        need(not any(k in env for k in ['USGS_T04_DOMAIN_CONFIG','USGS_T04_DOMAIN_CONFIG_SHA256','USGS_T04_DOMAIN_OPTIN']), 'NO_INHERITED_DOMAIN_CONFIG')
        env['PYTHONPATH'] = str(a.framework) + os.pathsep + env.get('PYTHONPATH','')
        config_path = root / 'domain-config.json'
        # Bootstrap waits for the exact config/owner files before exec, retaining its process identity.
        command_prefix = [sys.executable, '-c', BOOTSTRAP, str(a.framework/'tools/hygon_training_recipe.py'),'T04.sft']
        command_tail = ['--usgs-domain-config',str(config_path),'--usgs-domain-config-sha256','PENDING',
            '--a100-launch',str(a.a100_launch),'--swift',str(a.swift),'--models',str(a.models),'--plugins',str(a.plugins),
            '--master-port',str(a.master_port),'--receipt',str(root/'common-entry-receipt.jsonl'),'--run']
        # Config hash is unknown until PID exists. A second stdin line carries only that hash.
        bootstrap = BOOTSTRAP.replace("os.execv(sys.executable,[sys.executable]+sys.argv[1:])",
            "digest=sys.stdin.buffer.readline(66).strip().decode('ascii'); assert len(digest)==64 and set(digest)<=set('0123456789abcdef'); args=sys.argv[1:]; args[args.index('--usgs-domain-config-sha256')+1]=digest; os.execv(sys.executable,[sys.executable]+args)")
        command_prefix[2] = bootstrap
        phase = 'owned_entry_start'
        control['exec_request_attempted'] = True
        save(root,'control-state.json',control,replace=True)
        with (root/'native.stdout').open('xb') as out, (root/'native.stderr').open('xb') as err:
            child = subprocess.Popen(command_prefix+command_tail, env=env, stdin=subprocess.PIPE,
                                     stdout=out, stderr=err, start_new_session=True)
            anchor = dict(PID=child.pid,PGID=child.pid,start_new_session=True,request_id=a.request_id)
            save(root,'held-child-anchor.json',anchor)
            row = snapshot().get(child.pid)
            need(row is not None and row['PPID']==os.getpid() and row['PGID']==child.pid, 'OWNED_CHILD_IDENTITY')
            registry[child.pid] = {k: row[k] for k in ['PID','PPID','PGID','startticks','state']}
            owned = dict(**anchor, startticks=row['startticks'], uid=os.getuid(), host=c['supervisor']['host'], boot_id=c['supervisor']['boot_id'])
            c['owned_cleanup'] = dict(supervisor_source=source_pins['bounded_GPU_supervisor.py'],
                held_child=source_pins['held_child.py'],owned_processes=source_pins['owned_processes.py'],
                anchor=owned,scope_request_id=a.request_id)
            save(root,'child-owner.json',dict(supervisor=c['supervisor'],child=owned,request_id=a.request_id,epoch=a.epoch))
            config_pin = save(root,'domain-config.json',c)
            save(root,'exec-intent.json',dict(**control,mode=c['mode'],arm=c['arm'],config=config_pin,
                 child=owned,supervisor=c['supervisor'],same_framework_training=True))
            domain.remaining(c)
            child.stdin.write(('GO\n'+config_pin['SHA256']+'\n').encode()); child.stdin.flush(); child.stdin.close()
            control['entry_released'] = True; save(root,'control-state.json',control,replace=True)
            phase = 'native_common_entry'
            while child.poll() is None:
                refresh(registry,root); save(root,'owned-registry.json',registry,replace=True)
                try:
                    below=sample_group_rss(registry,resources)
                finally:
                    save(root,'resource-samples.json',resources,replace=True)
                need(below,'OWNED_GROUP_SAMPLED_RSS_SOFT_LIMIT')
                need(all((root/n).stat().st_size<=16*1024**2 for n in ['native.stdout','native.stderr']), 'NATIVE_LOG_BOUND')
                left = clock['supervisor_stop']-time.time()
                if left<=0:
                    timed_out=True; break
                try:
                    child.wait(timeout=min(1.0,left))
                except subprocess.TimeoutExpired:
                    pass
    except BaseException as error:
        failure = dict(phase=phase,error_type=type(error).__name__,message_SHA256=sha(str(error).encode()))
        try:
            private = dict(phase=phase,error_type=type(error).__name__,message=str(error)[:4096],trace=traceback.format_exc()[-4096:])
            failure['private_file'] = save(root,'failure-private.json',private)
        except BaseException:
            failure['private_file'] = None
    finally:
        # Do not let a second external TERM interrupt the held-Popen cleanup reserve.
        signal.signal(signal.SIGTERM,signal.SIG_IGN); signal.signal(signal.SIGINT,signal.SIG_IGN)
        try:
            held = cleanup_held(child,anchor,a.epoch+880)
            clean['held_child'] = held
        except BaseException as error:
            clean['held_child'] = dict(reaped=False,error_type=type(error).__name__)
        try:
            tree = cleanup(registry,root,grace=max(0,min(10,(a.epoch+880-time.time())/2)))
            clean.update(tree)
        except BaseException as error:
            clean['tree_error_type']=type(error).__name__
        clean['termination_needed'] = clean['termination_needed'] or clean['held_child'].get('termination_needed',False)
        if entry is not None and domain is not None:
            try:
                post_config=dict(c,deadline=clock['deadline'],maximum_seconds=900)
                post=entry.fresh_gpu_admission(post_config,a.master_port,domain.remaining)
                save(root,'after.json',post)
            except BaseException as error:
                post=dict(state='POST_UNKNOWN',error_type=type(error).__name__)
        for rank in range(8):
            try:
                raw,pin=stable(root/'witness'/('rank-%02d'%rank)/'receipt.json')
                receipts[str(rank)]=dict(file=pin,**rank_result(decode(raw),rank,c))
            except BaseException as error:
                receipt_errors[str(rank)]=dict(error_type=type(error).__name__)
        entry_ok=False
        try:
            raw,pin=stable(root/'common-entry-receipt.jsonl')
            rows=[decode(line) for line in raw.splitlines()]
            stages=entry_stages(rows)
            entry_ok=True
            entry_receipt=dict(file=pin,stages=stages,returncode=0)
        except BaseException as error:
            entry_receipt=dict(error_type=type(error).__name__)
        known_ownership=bool(child is not None or control['exec_request_attempted'] is False)
        clean_exit=bool(known_ownership and clean['held_child'].get('reaped') is True and clean['owned_processes_remaining']==0
                        and post and post.get('active_compute_processes')==0 and post.get('master_port_bindable') is True)
        passed=bool(child is not None and child.returncode==0 and failure is None and not timed_out
                    and not clean['termination_needed'] and clean_exit and entry_ok and len(receipts)==8 and not receipt_errors)
        outcome=dict(request_id=a.request_id,source_publication=a.publication,**clock,mode=c['mode'],arm=c['arm'],pair_id=c['pair_id'],
            native_returncode=child.returncode if child else None,control=control,failure=failure,timed_out=timed_out,resources=resources,
            entry_receipt=entry_receipt,rank_receipts=receipts,rank_receipt_errors=receipt_errors,
            planned_ranks=8,planned_microbatches_per_rank=100 if c['mode']=='fresh25' else 20,
            planned_updates_per_rank=25 if c['mode']=='fresh25' else 5,cleanup=clean,post=post,
            release_permitted=clean_exit,engineering_completed=passed,
            actual_GPU_training_started=None if not entry_ok else True,
            formal17=False,performance_qualified=False,effect_qualified=False,finished_epoch=time.time())
        save(root,'outcome.json',outcome)
        signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(dict(request_id=a.request_id,engineering_completed=passed,release_permitted=clean_exit,
                         native_returncode=outcome['native_returncode'],rank_receipts=len(receipts),formal17=False)))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
