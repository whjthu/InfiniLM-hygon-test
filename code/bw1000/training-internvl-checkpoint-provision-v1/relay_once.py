"""One receiver-first ordinary-SSH byte bridge; no local weight file or large buffer."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import queue
import re
import runpy
import shlex
import struct
import subprocess
import threading
import time
import uuid

D = Path(__file__).resolve().parent
G = D.parents[2]
HELPER = Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA = '3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
H = lambda raw: hashlib.sha256(raw).hexdigest()
BOOTSTRAP = ("import sys\n"
             "def exact(n):\n"
             " b=bytearray()\n"
             " while len(b)<n:\n"
             "  x=sys.stdin.buffer.read(n-len(b))\n"
             "  if not x: raise EOFError('short endpoint script')\n"
             "  b.extend(x)\n"
             " return bytes(b)\n"
             "n=int.from_bytes(exact(8),'little')\n"
             "if not 0<n<=262144: raise ValueError('script bound')\n"
             "exec(compile(exact(n),'published-internvl-provision','exec'))\n")


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def left(deadline, monotonic_end, reserve=0):
    remaining = min(deadline-time.time(), monotonic_end-time.monotonic())-reserve
    if remaining <= 0:
        raise TimeoutError('original six-hour relay deadline exhausted')
    return remaining


def envelope(script):
    if not 0 < len(script) <= 262144:
        raise ValueError('endpoint script size differs')
    return struct.pack('<Q', len(script))+script


def transport_state(*, ready, forced, returncode, expired):
    # Killing our SSH client never establishes the remote execution outcome.
    return 'RETURNED' if ready and not forced and not expired and returncode not in (None, 255) else 'UNKNOWN'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publication', required=True)
    parser.add_argument('--source-manifest-sha', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', args.publication)
    mfraw = (D/'SOURCE_FILES.json').read_bytes()
    assert H(mfraw) == args.source_manifest_sha
    git = ['git', '-c', 'safe.directory='+G.as_posix(), '-C', str(G)]
    published = subprocess.check_output(git+['show', args.publication+':'+D.relative_to(G).as_posix()+'/SOURCE_FILES.json'], timeout=12)
    assert published == mfraw
    blobs = {}
    for name, pin in json.loads(mfraw)['files'].items():
        raw = (D/name).read_bytes()
        assert len(raw) == pin['bytes'] and H(raw) == pin['SHA256']
        blobs[name] = raw
    plan, claim = json.loads(blobs['PLAN.json']), json.loads(blobs['CLAIM.json'])
    assert claim['provision_GO'] is True and claim['max_invocations'] == 1 and claim['case'] == 'T03.sft'
    assert claim['GPU_GO'] is False and claim['whole_seconds'] == 21600
    for name,pin in claim['published_dependencies'].items():
        dep = subprocess.check_output(git+['show',args.publication+':'+name],timeout=12)
        assert len(dep) == pin['bytes'] and H(dep) == pin['SHA256']
    reference = (G/plan['local_reference']['path']).read_bytes()
    assert len(reference) == plan['local_reference']['bytes'] == 25357 and H(reference) == plan['local_reference']['SHA256']
    assert H(HELPER.read_bytes()) == HELPER_SHA
    base = runpy.run_path(str(HELPER))['ssh']
    local = G/'local-only'/plan['label']
    local.mkdir(mode=0o700)  # fixed scope, no repeated or renamed submission
    epoch, monotonic_start = time.time(), time.monotonic()
    context = dict(publication=args.publication, request_id=uuid.uuid4().hex, epoch=epoch,
                   deadline=epoch+21600, source_manifest_SHA256=args.source_manifest_sha)
    deadline, monotonic_end = context['deadline'], monotonic_start+21600
    save(local/'intent.json', dict(context, byte_budget=plan['stream_payload_bytes'],
                                  file_denominator=26, receiver_first=True, replay_allowed=False))
    children, readers, errors, ready_rows = {}, [], [], {}
    forced_local_kills = set()
    expired = threading.Event()
    transferred = 0

    def kill_owned():
        for child in list(children.values()):
            if child.poll() is None:
                forced_local_kills.add(child.pid)
                child.kill()  # Only our unreaped local SSH child, never a remote guessed PID.

    def expire():
        expired.set()
        kill_owned()

    timer = threading.Timer(left(deadline, monotonic_end, 5), expire)
    timer.daemon = True
    timer.start()

    def start(mode):
        prefix = ('MODE='+repr(mode)+'\nPLAN='+repr(plan)+'\nCONTEXT='+repr(context)+'\n')
        prefix += 'REFERENCE_B64='+repr(base64.b64encode(reference).decode() if mode == 'source' else '')+'\n'
        code = prefix.encode()+blobs['protocol.py']
        if mode == 'receiver':
            argv = [base[0], '-o', 'StrictHostKeyChecking=yes', *base[1:],
                    '/usr/local/python3.12/bin/python3 -I -S -B -c '+shlex.quote(BOOTSTRAP)]
        else:
            config = G.__class__('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config')
            argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=12',
                    '-F', str(config), 'a100-27', '/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -c '+shlex.quote(BOOTSTRAP)]
        assert len(subprocess.list2cmdline(argv)) < 30000
        child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        children[mode] = child
        save(local/(mode+'-owned-ssh.json'), dict(pid=child.pid, owned_unreaped_child=True, deadline=deadline))

        def drain():
            size, digest = 0, hashlib.sha256()
            try:
                with (local/(mode+'-stderr.private')).open('xb') as stream:
                    while True:
                        block = child.stderr.read(65536)
                        if not block:
                            break
                        size += len(block)
                        if size > 262144:
                            raise ValueError('bounded SSH stderr exceeded')
                        digest.update(block); stream.write(block)
            except BaseException as error:
                errors.append(type(error).__name__)
                kill_owned()
            finally:
                save(local/(mode+'-stderr-pin.json'), dict(bytes=size, SHA256=digest.hexdigest()))
        reader = threading.Thread(target=drain, daemon=True); reader.start(); readers.append(reader)
        child.stdin.write(envelope(code)); child.stdin.flush()
        if mode == 'source':
            child.stdin.close()
        received = queue.Queue()
        threading.Thread(target=lambda: received.put(child.stdout.readline(8193)), daemon=True).start()
        raw = received.get(timeout=min(45, left(deadline, monotonic_end, 10)))
        assert len(raw) <= 8192
        ready = json.loads(raw)
        assert ready['stage'] == 'READY' and ready['mode'] == mode
        assert all(ready[key] == value for key,value in context.items())
        assert ready['expected_payload_bytes'] == plan['stream_payload_bytes'] and ready['expected_files'] == 26
        assert len(ready['CPU_affinity']) == 2 and ready['AS_limit_bytes'] == 512*1024**2
        assert ready['nice'] == 10 and ready['uid'] == {'source':1001, 'receiver':154459}[mode]
        assert type(ready['pid']) is type(ready['startticks']) is int and ready['pid'] > 0 and ready['startticks'] > 0
        assert ready['boot_id'] and isinstance(ready['host'], str) and ready['host']
        save(local/(mode+'-remote-owned.json'), ready)
        ready_rows[mode] = ready
        return child

    outcome = dict(context, passed=False, replay_allowed=False, source='NOT_DISPATCHED', receiver='NOT_DISPATCHED',
                   planned_files=26, planned_payload_bytes=plan['stream_payload_bytes'], GPU_GO=False, Model_GO=False)
    try:
        receiver = start('receiver')
        outcome['receiver'] = 'READY'
        source = start('source')  # Receiver namespace/space/identity ack precedes all source data reads.
        outcome['source'] = 'READY'
        last_checkpoint = 0
        while True:
            left(deadline, monotonic_end, 5)
            block = source.stdout.read1(plan['chunk_bytes'])
            if not block:
                break
            if transferred+len(block) > plan['stream_payload_bytes']:
                raise ValueError('source stream exceeds exact file budget')
            if receiver.stdin.write(block) != len(block):
                raise IOError('short local bridge write')
            transferred += len(block)
            if transferred-last_checkpoint >= plan['progress_bytes']:
                value = dict(context, logical_forwarded_bytes=transferred, expected_bytes=plan['stream_payload_bytes'],
                             actual_root_weight_files=0, observed_epoch=time.time())
                temp = local/'progress.tmp'
                with temp.open('w', encoding='utf-8') as stream: json.dump(value, stream)
                temp.replace(local/'progress.json')
                last_checkpoint = transferred
        source.wait(timeout=left(deadline, monotonic_end, 5))
        receiver.stdin.close()
        final = queue.Queue()
        threading.Thread(target=lambda: final.put(receiver.stdout.readline(32769)), daemon=True).start()
        raw = final.get(timeout=min(45, left(deadline, monotonic_end, 5)))
        assert len(raw) <= 32768
        target = json.loads(raw)
        receiver.wait(timeout=left(deadline, monotonic_end, 5))
        assert all(target[key] == value for key,value in context.items())
        save(local/'receiver-terminal.json', target)
        outcome.update(source='RETURNED' if source.returncode != 255 else 'UNKNOWN',
                       receiver='RETURNED' if receiver.returncode != 255 else 'UNKNOWN',
                       source_returncode=source.returncode, receiver_returncode=receiver.returncode,
                       passed=bool(source.returncode == receiver.returncode == 0 and target['passed'] is True
                                   and target['completed_files'] == 26 and transferred == plan['stream_payload_bytes']))
    except BaseException as error:
        outcome['error_type'] = type(error).__name__
    finally:
        for mode, child in children.items():
            if child.poll() is None:
                forced_local_kills.add(child.pid)
                child.kill()
                outcome[mode] = 'UNKNOWN'
            forced = child.pid in forced_local_kills
            try:
                child.wait(timeout=max(.01, min(4, deadline-time.time())))
            except subprocess.TimeoutExpired:
                outcome[mode] = 'UNKNOWN'
            outcome[mode] = transport_state(ready=mode in ready_rows, forced=forced,
                                             returncode=child.returncode, expired=expired.is_set())
            outcome[mode+'_transport'] = dict(returncode=child.returncode, forced_local_SSH_kill=forced,
                                              remote_owned_identity_acknowledged=mode in ready_rows,
                                              remote_outcome='UNKNOWN_PENDING_ORIGINAL_RECEIPT_RECONCILE')
        timer.cancel()
        for reader in readers:
            reader.join(timeout=max(.01, min(2, deadline-time.time())))
    outcome.update(logical_forwarded_bytes=transferred, actual_root_weight_files=0,
                   root_elapsed_seconds=time.monotonic()-monotonic_start, root_max_chunk_bytes=plan['chunk_bytes'],
                   deadline_expired=expired.is_set(), stream_errors=errors,
                   physical_wire_and_filesystem_IO='UNKNOWN_NOT_MEASURED',
                   independent_source_and_target_artifact_audit_required=True)
    outcome['passed'] = bool(outcome['passed'] and not errors and not expired.is_set())
    save(local/'outcome.json', outcome)
    print(json.dumps(outcome, allow_nan=False))
    return 0 if outcome['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
