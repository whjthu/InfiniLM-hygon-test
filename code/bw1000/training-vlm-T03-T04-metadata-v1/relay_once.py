"""Once-only bounded in-memory SSH relay, dispatched only by root after publication."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import struct
import threading
import time
import uuid

D = Path(__file__).resolve().parent
G = D.parents[2]
HELPER = Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA = '3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
BOOTSTRAP = ("import sys\n"
             "def exact(n):\n"
             " b=bytearray()\n"
             " while len(b)<n:\n"
             "  x=sys.stdin.buffer.read(n-len(b))\n"
             "  if not x: raise EOFError('short source envelope')\n"
             "  b.extend(x)\n"
             " return bytes(b)\n"
             "n=int.from_bytes(exact(8),'little')\n"
             "if not 0<n<=131072: raise ValueError('source envelope limit')\n"
             "exec(compile(exact(n),'published-metadata-endpoint','exec'))\n")


def frame(script, suffix=b''):
    if not 0 < len(script) <= 128*1024:
        raise ValueError('endpoint source envelope too large')
    return struct.pack('<Q', len(script)) + script + suffix


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, sort_keys=True, allow_nan=False)
        f.write('\n')


def remaining(deadline, mono_end):
    value = min(deadline-time.time(), mono_end-time.monotonic())
    if not math.isfinite(value) or value <= 0:
        raise TimeoutError('original relay epoch exhausted')
    return value


def bounded_call(argv, payload, maximum, deadline, mono_end, local, tag):
    # Readers are capped while the process runs; no unbounded communicate buffers.
    # Reserve5s inside the original300s for local SSH reap; remote deadline never moves.
    active_budget = remaining(deadline, mono_end) - 5
    if active_budget <= 0:
        raise TimeoutError('no original epoch cleanup reserve')
    if len(subprocess.list2cmdline(argv)) >= 30000:
        raise ValueError('SSH argv exceeds Windows command-line bound')
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    save(local/(tag+'-owned-ssh.json'), dict(pid=proc.pid, epoch=time.time(), owned_unreaped_child=True,
                                          deadline=deadline, argv_not_recorded=True))
    buffers, errors = {}, []
    deadline_hit = threading.Event()

    def expire():
        deadline_hit.set()
        if proc.poll() is None:
            proc.kill()

    timer = threading.Timer(active_budget, expire)
    timer.daemon = True
    timer.start()

    def read(name, stream, cap):
        data = bytearray()
        try:
            while True:
                block = stream.read(65536)
                if not block:
                    break
                if len(data)+len(block) > cap:
                    raise ValueError(name+' exceeds relay cap')
                data.extend(block)
            buffers[name] = bytes(data)
        except BaseException as error:
            errors.append(type(error).__name__)
            if proc.poll() is None:
                proc.kill()
        finally:
            stream.close()

    def write():
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except (BrokenPipeError, OSError) as error:
            errors.append(type(error).__name__)

    workers = [threading.Thread(target=read, args=('stdout', proc.stdout, maximum), daemon=True),
               threading.Thread(target=read, args=('stderr', proc.stderr, 512*1024), daemon=True),
               threading.Thread(target=write, daemon=True)]
    for worker in workers:
        worker.start()
    timed_out = False
    try:
        proc.wait(timeout=max(.001, remaining(deadline, mono_end)-5))
        for worker in workers:
            worker.join(timeout=max(.001, remaining(deadline, mono_end)-5))
        if any(worker.is_alive() for worker in workers):
            raise TimeoutError('relay stream did not finish')
    except (subprocess.TimeoutExpired, TimeoutError):
        timed_out = True
        if proc.poll() is None:
            proc.kill()
        # Only reap our already-exited/killed SSH child; no remote fallback or retry.
        proc.wait(timeout=remaining(deadline, mono_end))
    finally:
        timer.cancel()
    timed_out = timed_out or deadline_hit.is_set()
    out, err = buffers.get('stdout', b''), buffers.get('stderr', b'')
    receipt = dict(returncode=proc.returncode, timed_out=timed_out, stream_errors=errors,
                   stdout_bytes=len(out), stdout_SHA256=sha(out), stderr_bytes=len(err), stderr_SHA256=sha(err),
                   remote_outcome='UNKNOWN' if timed_out or proc.returncode == 255 or errors else 'RETURNED',
                   replay_allowed=False)
    save(local/(tag+'-transport.json'), receipt)
    # stderr retained locally, bounded, never printed; compressed source stdout never lands locally.
    (local/(tag+'-stderr.private')).write_bytes(err)
    return receipt, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publication', required=True)
    parser.add_argument('--source-manifest-sha', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.publication):
        raise ValueError('exact publication required')
    mf_raw = (D/'SOURCE_FILES.json').read_bytes()
    if sha(mf_raw) != args.source_manifest_sha:
        raise ValueError('published source manifest differs')
    mf = json.loads(mf_raw)
    relative_mf = D.relative_to(G).as_posix() + '/SOURCE_FILES.json'
    published_mf = subprocess.check_output(['git', '-c', 'safe.directory='+G.as_posix(), '-C', str(G),
                                            'show', args.publication+':'+relative_mf], timeout=12)
    if published_mf != mf_raw:
        raise ValueError('publication does not contain this exact source manifest')
    blobs = {}
    for name, pin in mf['files'].items():
        raw = (D/name).read_bytes()
        if len(raw) != pin['bytes'] or sha(raw) != pin['SHA256']:
            raise ValueError('published source differs')
        blobs[name] = raw
    plan = json.loads(blobs['PLAN.json'])
    claim = json.loads(blobs['CLAIM.json'])
    if not (claim['metadata_stage_GO'] is True and claim['max_invocations'] == 1 and
            claim['GPU_GO'] is False and claim['Model_GO'] is False and claim['epoch_seconds'] == 300):
        raise ValueError('published metadata claim does not authorize this scope')
    if sha(HELPER.read_bytes()) != HELPER_SHA:
        raise ValueError('existing Hygon helper differs')
    epoch, mono = time.time(), time.monotonic()
    context = dict(publication=args.publication, request_id=uuid.uuid4().hex, epoch=epoch,
                   deadline=epoch+300, source_manifest_SHA256=args.source_manifest_sha)
    deadline, mono_end = context['deadline'], mono+300
    local = G/'local-only'/plan['label']
    local.mkdir(mode=0o700)  # fixed scope; no renamed replay or multiple requests
    save(local/'intent.json', dict(context, metadata_only=True, replay_allowed=False))
    value = dict(context, passed=False, source='NOT_DISPATCHED', receiver='NOT_DISPATCHED',
                 GPU_GO=False, Model_GO=False, full_weight_identity=False, replay_allowed=False)
    try:
        prefix = 'PLAN='+repr(plan)+'\nCONTEXT='+repr(context)+'\n'
        source_code = prefix + blobs['common.py'].decode() + '\n' + blobs['source.py'].decode()
        a = Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
        source_argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                       '-o', 'StrictHostKeyChecking=yes', '-F', str(a/'ssh_config'), 'a100-27',
                       '/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -c '+shlex.quote(BOOTSTRAP)]
        remaining(deadline, mono_end)
        source, compressed = bounded_call(source_argv, frame(source_code.encode('utf-8')), plan['max_compressed_bytes'],
                                          deadline, mono_end, local, 'source')
        value['source'] = source['remote_outcome']
        if source['returncode'] != 0 or source['remote_outcome'] != 'RETURNED' or not compressed:
            raise ValueError('source did not return complete bundle; reconcile original namespace')
        remaining(deadline, mono_end)
        target_argv = runpy.run_path(str(HELPER))['ssh'].copy()
        target_code = prefix + blobs['common.py'].decode() + '\n' + blobs['receive.py'].decode()
        code_bytes = target_code.encode('utf-8')
        # Short argv also works under Windows' command-line limit. The same stdin
        # buffer then supplies only the compressed bundle to the real receiver.
        target_argv = [target_argv[0], '-o', 'StrictHostKeyChecking=yes', *target_argv[1:],
                       '/usr/local/python3.12/bin/python3 -I -S -B -c '+shlex.quote(BOOTSTRAP)]
        envelope = frame(code_bytes, compressed)
        receiver, stdout = bounded_call(target_argv, envelope, 32768, deadline, mono_end, local, 'receiver')
        del envelope
        del compressed
        value['receiver'] = receiver['remote_outcome']
        if receiver['remote_outcome'] != 'RETURNED':
            raise ValueError('receiver execution unknown; reconcile original namespace')
        result = json.loads(stdout)
        if any(result.get(k) != v for k, v in context.items()):
            raise ValueError('receiver context differs')
        save(local/'receiver-summary.json', result)
        value['passed'] = receiver['returncode'] == 0 and result['passed'] is True
        value['receiver_summary'] = result
    except BaseException as error:
        value['error_type'] = type(error).__name__
    value['finished_epoch'] = time.time()
    save(local/'outcome.json', value)
    print(json.dumps({k:v for k,v in value.items() if k != 'receiver_summary'}, allow_nan=False))
    return 0 if value['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
