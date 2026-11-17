"""One original-owner trace-tail observation; stdlib, no container/device query."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import socket
import stat
import subprocess
import sys
import time

D = Path(__file__).resolve().parent
H = lambda b: hashlib.sha256(b).hexdigest()
LABEL = 'GLM188-live-trace-tail-v1'
NODES = ['f13r1n05', 'f13r1n07']
COLUMNS = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'
PYTHON = '/usr/local/python3.12/bin/python3'


def need(v, why):
    if not v:
        raise ValueError(why)


def save(p, q):
    tmp = p.with_name(p.name+'.tmp')
    with tmp.open('x') as f:
        json.dump(q, f, allow_nan=False, sort_keys=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def parse_tail(raw, offset, now):
    lines = raw.split(b'\n')[:-1]
    if offset and lines:
        lines = lines[1:]
    need(bool(lines), 'NO_COMPLETE_BOUNDED_EVENT')
    event = json.loads(lines[-1]); phase = event['phase']
    need(phase in ('forward.enter','forward.return','request.enter','request.return',
                   'barrier.enter','barrier.return','destroy.enter','destroy.return'), 'PHASE_NOT_WHITELISTED')
    stamp = event['monotonic']
    need(type(stamp) in (int,float) and math.isfinite(stamp) and 0 <= stamp <= now, 'LOCAL_EVENT_CLOCK')
    progress = event.get('progress')
    if phase in ('forward.enter','forward.return'):
        need(isinstance(progress,dict), 'FORWARD_PROGRESS_MISSING')
    if progress is not None:
        need(set(progress) == {'ordinal','generated','forward_attempts','forward_completed'}
             and all(type(v) is int and v >= 0 for v in progress.values()), 'COUNTER_SCHEMA')
    return dict(state='OBSERVED', lastphase=phase, progress=progress, event_monotonic=stamp,
                same_host_boot_event_age_s=now-stamp,
                more_forward_than_old_snapshot=bool(progress and progress['forward_completed'] > 12017))


def contract(resolved=True):
    c = json.loads((D/'CLAIM.json').read_bytes())
    q = json.loads(((D.parent/'resolved-identities.json') if resolved else (D/'IDENTITIES.json')).read_bytes())
    need(c['ROOT_GO'] is True and c['CPU_GO'] is True and c['GPU_GO'] is False
         and c['max_invocations'] == 1 and c['label'] == LABEL, 'ROOT_ADMISSION')
    if not resolved:
        return q
    need(q['identity_complete'] is True and q['original_step'] == '1103606.188'
         and set(q['node_boot_ids']) == set(NODES) and len(q['owners']) == 16, 'IDENTITIES_MISSING')
    for r, x in enumerate(q['owners']):
        need(x['rank'] == r and x['node'] == NODES[r//8] and type(x['UID']) is int
             and type(x['PID']) is int and x['PID'] > 0 and type(x['startticks']) is int
             and x['startticks'] > 0 and re.fullmatch('[0-9a-f-]{36}', q['node_boot_ids'][x['node']]), 'OWNER_IDENTITY')
    return q


def node(epoch, request):
    q = contract(); host = socket.gethostname().split('.')[0]
    need(host in NODES and os.environ.get('SLURM_JOB_ID') == '1103606', 'ORIGINAL_NODE_PARENT')
    left = epoch+20-time.time(); need(0 < left <= 20, 'NODE_ORIGINAL_EPOCH')
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124)); signal.setitimer(signal.ITIMER_REAL, left)
    resource.setrlimit(resource.RLIMIT_AS, (256*1024**2,)*2)
    used = 0

    def read(p, cap):
        nonlocal used
        need(used+cap <= 16384, 'NODE_READ_CAP')
        with open(p, 'rb', buffering=0) as f:
            raw = f.read(cap)
        used += len(raw)
        need(len(raw) < cap, 'METADATA_TRUNCATED')
        return raw

    boot = read('/proc/sys/kernel/random/boot_id', 64).decode().strip()
    need(boot == q['node_boot_ids'][host], 'ORIGINAL_BOOT_CHANGED')
    rows = []
    for original in q['owners']:
        if original['node'] != host:
            continue
        x = dict(original, state='UNKNOWN', tail_bytes=0)
        rows.append(x)
        try:
            pid = original['PID']; proc = Path('/proc')/str(pid)

            def identity():
                need(proc.stat().st_uid == original['UID'] == os.getuid(), 'UID_CHANGED')
                fields = read(proc/'stat', 512).decode().rsplit(')', 1)[1].split()
                need(int(fields[19]) == original['startticks'] and fields[0] not in ('Z','X'), 'ORIGINAL_NOT_LIVE')
                return fields[0]

            x['process_state_before'] = identity()
            p = proc/'root/local'/('r470-pipe-owner-%02d-%d.jsonl' % (original['rank'], pid))
            fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                before = os.fstat(fd)
                need(stat.S_ISREG(before.st_mode) and before.st_uid == original['UID'], 'TRACE_REGULAR_OWNER')
                offset = max(0, before.st_size-896)
                need(used+896 <= 16384, 'NODE_READ_CAP')
                raw = os.pread(fd, min(896, before.st_size), offset); used += len(raw)
                after = os.fstat(fd)
                need((before.st_dev,before.st_ino) == (after.st_dev,after.st_ino)
                     and after.st_size >= before.st_size and len(raw) == before.st_size-offset, 'TRACE_CHANGED_OR_SHORT')
            finally:
                os.close(fd)
            x['process_state_after'] = identity()
            x.update(tail_bytes=len(raw), tail_SHA256=H(raw), offset=offset,
                     size_before=before.st_size, size_after=after.st_size,
                     inode=before.st_ino, device=before.st_dev)
            x.update(parse_tail(raw, offset, time.monotonic()))
        except (OSError, ValueError, KeyError, IndexError, UnicodeError) as e:
            x['error_type'] = type(e).__name__  # No raw token, path, exception message or event.
    result = dict(request=request, node=host, boot_id=boot, original_step=q['original_step'],
                  planned=8, rows=rows, raw_original_bytes=used, GPU_queries=0,
                  diagnostic_only=True, hung_or_normal_verdict='UNKNOWN')
    save(D.parent/('node-'+host+'.json'), result)


def controller(epoch, request, publication):
    q = contract(False); end = time.monotonic()+max(0, epoch+28-time.time())
    need(0 < epoch+30-time.time() <= 30, 'CONTROL_ORIGINAL_EPOCH')
    out = D.parent; attempted = False; held = None
    result = dict(request=request, publication=publication, epoch=epoch, deadline=epoch+30,
                  original_step=q['original_step'], planned_owners=16, exec_request_attempted=False,
                  state='UNKNOWN', original_execution_state='NOT_DECIDED_BY_THIS_DIAGNOSTIC',
                  GPU_queries=0, Model_launches=0, replay_allowed=False)

    def left(reserve=0):
        n = end-time.monotonic()-reserve
        need(n > 0, 'READ_EPOCH_EXHAUSTED')
        return n

    def cleanup(child):
        if child is None:
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if child.poll() is None:
                try: os.killpg(child.pid, sig)
                except ProcessLookupError: pass
            try:
                child.wait(timeout=min(1, max(.01,end-time.monotonic())))
                return
            except subprocess.TimeoutExpired:
                pass

    def query(argv, cap=262144):
        child = None
        try:
            child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            raw, err = child.communicate(timeout=min(3,left(2)))
            need(child.returncode == 0 and len(raw) <= cap and len(err) <= 65536, 'ACCOUNTING_COMMAND')
            return raw.decode()
        finally:
            cleanup(child)

    def account():
        text = query(['sacct','-n','-P','-j','1103606','--format='+COLUMNS])
        rows = [s.split('|') for s in text.splitlines() if s]
        return [r for r in rows if len(r)>=7 and r[1] == LABEL]

    save(out/'control-state.json', result)
    try:
        def expired(*_):
            raise TimeoutError('CONTROL_WORK_DEADLINE')
        signal.signal(signal.SIGALRM, expired)
        signal.signal(signal.SIGTERM, expired)
        signal.signal(signal.SIGHUP, expired)
        signal.setitimer(signal.ITIMER_REAL, max(.001, epoch+20-time.time()))
        parent = query(['scontrol','show','job','1103606','-o'])
        fields = dict(s.split('=',1) for s in parent.split() if '=' in s)
        uid = re.fullmatch(r'[^()]+\(([0-9]+)\)', fields.get('UserId',''))
        need(uid and int(uid[1]) == os.getuid() and fields['JobState']=='RUNNING', 'PARENT_OWNER_STATE')
        hosts = query(['scontrol','show','hostnames',fields['NodeList']]).splitlines()
        need(set(NODES) <= set(hosts), 'PARENT_NODES')
        need(not account(), 'EXACT_LABEL_ALREADY_USED')
        step = query(['scontrol','show','step','1103606.188','-o'])
        sf = dict(s.split('=',1) for s in step.split() if '=' in s)
        need(sf.get('StepId') == '1103606.188' and sf.get('State') == 'RUNNING', 'ORIGINAL_STEP_NOT_RUNNING')
        # Exact original small records, same one invocation. No owner bodies or directory scan.
        original = Path(q['original_output'])
        evidence = {}; owners = []; boots = {}; manifests = set()
        for i, host in enumerate(NODES):
            values = []
            for rel, cap in [('worker-%d.json'%i,65536), ('pre/'+host+'.json',16384)]:
                p = original/rel
                fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    before = os.fstat(fd)
                    need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
                         and 0 < before.st_size <= cap, 'ORIGINAL_SMALL_BOUND')
                    b = os.read(fd, before.st_size); after = os.fstat(fd)
                    need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
                         == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
                         and len(b)==before.st_size, 'ORIGINAL_SMALL_CHANGED')
                finally:
                    os.close(fd)
                values.append(json.loads(b)); evidence[rel] = dict(bytes=len(b),SHA256=H(b))
            worker, pre = values
            need(worker['node']==host and worker['node_index']==i and worker['UID']==os.getuid()
                 and worker['actual_GPU']==q['original_step'] and worker['gpu_epoch']==q['original_epoch']
                 and worker['completed'] is False and len(worker['owners'])==8, 'ORIGINAL_WORKER_BINDING')
            ident = worker['identity']
            need(ident['label']==original.name and ident['step']==q['original_step']
                 and ident['epoch']==q['original_epoch'] and ident['arm']=='socket'
                 and ident['source_manifest_SHA256']==q['original_source_manifest_SHA256']
                 and ident['Native_SHA256']==q['original_Native_SHA256'], 'ORIGINAL_WORKER_SOURCE')
            manifests.add(ident['run_manifest_SHA256'])
            need(pre['node']==host and pre['step']=='1103606.187', 'ORIGINAL_PRE_BINDING')
            boots[host] = pre['boot_id']
            for j, item in enumerate(worker['owners']):
                need(item['global_rank']==item['rank']==i*8+j, 'ORIGINAL_OWNER_RANK')
                owners.append(dict(rank=i*8+j,node=host,UID=worker['UID'],PID=item['PID'],startticks=int(item['startticks'])))
        need(len(manifests)==1 and re.fullmatch('[0-9a-f]{64}',next(iter(manifests))), 'ORIGINAL_MANIFEST_CONSISTENCY')
        q.update(identity_complete=True,owners=owners,node_boot_ids=boots,
                 original_small_evidence=evidence,original_GPU_manifest_SHA256=next(iter(manifests)),
                 UID_source='Original worker UID inherited by Popen child; live procUID independently required.')
        save(out/'resolved-identities.json', q); contract()
        result['identity_evidence'] = evidence
        need(epoch+20-time.time() >= 5, 'NO_NODE_BUDGET')
        argv = ['srun','--jobid=1103606','--job-name='+LABEL,'--exact','--immediate=5',
                '--nodes=2','--ntasks=2','--ntasks-per-node=1','--distribution=block:block',
                '--nodelist='+','.join(NODES),'--cpus-per-task=1','--mem=512M','--gres=none',
                '--cpu-bind=none','--kill-on-bad-exit=1','--time=1',PYTHON,'-I','-S','-B',
                str(D/'read_tail.py'),'node',str(epoch),request]
        result['exec_request_attempted'] = attempted = True
        save(out/'control-state.json', result)  # fsync before the only srun Popen.
        with (out/'step.stdout.private').open('xb') as stdout, (out/'step.stderr.private').open('xb') as stderr:
            held = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
            st = Path('/proc/%d/stat'%held.pid).read_text().rsplit(')',1)[1].split()
            save(out/'owned-srun.json', dict(PID=held.pid,startticks=int(st[19]),host=socket.gethostname(),
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),request=request))
            held.wait(timeout=min(left(8), max(.01,epoch+20-time.time())))
        result['srun_returncode'] = held.returncode
    except BaseException as e:
        result['error_type'] = type(e).__name__
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
        signal.setitimer(signal.ITIMER_REAL, max(.001,end-time.monotonic()))
        cleanup(held)
        result['owned_srun_reaped'] = held is None or held.poll() is not None
        if attempted:
            try:
                rows = account(); need(len(rows) <= 1, 'DUPLICATE_ACCOUNTING_HANDLE')
                if rows and rows[0][2] not in ('RUNNING','PENDING','COMPLETING'):
                    result.update(state='TERMINAL', accounting=rows[0])
            except BaseException as e:
                result['accounting_error_type'] = type(e).__name__
        else:
            result['state'] = 'NOT_SUBMITTED'
        result['nodes'] = {}
        for host in NODES:
            p = out/('node-'+host+'.json')
            try:
                if p.is_file() and p.stat().st_size <= 32768:
                    b = p.read_bytes(); value = json.loads(b)
                    need(value['request'] == request and value['node'] == host
                         and isinstance(value['rows'],list) and len(value['rows']) == 8
                         and all(isinstance(x,dict) for x in value['rows']), 'NODE_RESULT_BINDING')
                    first = NODES.index(host)*8
                    need([x.get('rank') for x in value['rows']] == list(range(first,first+8))
                         and all(x.get('state') in ('UNKNOWN','OBSERVED') for x in value['rows']), 'NODE_RESULT_RANKS')
                    result['nodes'][host] = dict(SHA256=H(b),value=value)
            except (OSError,ValueError,KeyError,TypeError) as e:
                result.setdefault('node_result_errors',{})[host] = type(e).__name__
        all_rows = {x['rank']:x for v in result['nodes'].values() for x in v['value']['rows']}
        result['owners'] = [all_rows.get(r,dict(rank=r,node=NODES[r//8],state='UNKNOWN',
                                  error_type='NODE_RESULT_MISSING_OR_REJECTED')) for r in range(16)]
        result['observed_owners'] = sum(x['state']=='OBSERVED' for x in result['owners'])
        result['finished_epoch'] = time.time()
        save(out/'outcome.json', result)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    if sys.argv[1] == 'node':
        node(float(sys.argv[2]), sys.argv[3])
    else:
        controller(float(sys.argv[2]), sys.argv[3], sys.argv[4])
