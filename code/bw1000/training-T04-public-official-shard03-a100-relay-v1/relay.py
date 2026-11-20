"""Run a frozen source-to-target relay with no model data written on this computer."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import threading
import time

from endpoint import CHUNK, CHECKPOINT_SIZE, COMMIT, frame, validate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HELPER = Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA = '3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
A100_CONFIG = 'C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config'
LOCAL = REPO / 'local-only' / HERE.name


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, obj):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, allow_nan=False)
        stream.write('\n')


def open_published(publication):
    git = ['git', '-c', 'safe.directory='+REPO.as_posix()]
    relative = HERE.relative_to(REPO).as_posix()

    def read_git(*args):
        return subprocess.check_output([*git, *args], cwd=REPO, timeout=12)

    manifest = (HERE / 'SOURCE_FILES.json').read_bytes()
    validate(read_git('cat-file', 'blob', publication+':'+relative+'/SOURCE_FILES.json') == manifest,
             'PUBLISHED_SOURCE_MANIFEST_MISMATCH')
    validate(not read_git('status', '--porcelain', '--', relative).strip(), 'SOURCE_MODIFIED')
    for name, pin in json.loads(manifest)['files'].items():
        raw = (HERE / name).read_bytes()
        validate((len(raw), sha(raw)) == (pin['bytes'], pin['SHA256']), 'SOURCE_FILE_PIN')
    plan = json.loads((HERE / 'PLAN.json').read_bytes())
    claim = json.loads((HERE / 'CLAIM.json').read_bytes())
    validate(claim['request'] == plan['request'] and claim['target'] == plan['target']
             and claim['transfer_GO'] is True and claim['max_transfer_invocations'] == 1
             and claim['replay_allowed'] is False, 'EXACT_TRANSFER_SCOPE')
    validate(all(isinstance(plan['source_stat'][key], str) and re.fullmatch('[0-9]{19}', plan['source_stat'][key])
                 for key in ('mtime_ns', 'ctime_ns')), 'LOSSLESS_NANOSECOND_IDENTITY')
    ref = plan['official_evidence']
    raw = read_git('cat-file', 'blob', publication+':'+ref['path'])
    validate(sha(raw) == ref['SHA256'], 'OFFICIAL_EVIDENCE_PIN')
    public = json.loads(raw)
    item = public['public_metadata']['shards'][plan['file']]
    validate(public['official_revision'] == plan['revision']
             and item['bytes'] == plan['bytes'] == plan['source_stat']['size']
             and item['lfs_sha256'] == plan['SHA256'] and item['sha256_match'] and item['bytes_match'],
             'EXACT_PUBLIC_AND_A100_MATCH')
    validate(plan['file'] == 'model-00003-of-00004.safetensors' and plan['bytes'] == 4952432880
             and plan['timeout_seconds'] == 21600 and Path(plan['target']).name == HERE.name
             and plan['rate_bytes_per_second'] == 2*CHUNK, 'PREDECLARED_SOURCE_LIMITS')
    return plan


def command(plan, binding, role, budget):
    validate(sha(HELPER.read_bytes()) == HELPER_SHA, 'SSH_HELPER_CHANGED')
    if role == 'sender':
        ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'ConnectTimeout=12', '-F', A100_CONFIG, 'a100-26']
        interpreter = '/home/whj/nvidia-baseline/env-vlm/bin/python'
    else:
        argv = runpy.run_path(str(HELPER))['ssh']
        ssh = [argv[0], '-o', 'StrictHostKeyChecking=yes', *argv[1:]]
        interpreter = '/usr/local/python3.12/bin/python3'
    code = (HERE / 'endpoint.py').read_text(encoding='utf-8')
    code += '\nraise SystemExit(endpoint('+repr(plan)+','+repr(binding)+','+repr(role)+','+str(budget)+'))\n'
    args = [*ssh, interpreter+' -I -S -B -u -c '+shlex.quote(code)]
    validate(len(subprocess.list2cmdline(args)) < 30000, 'BOUNDED_SSH_ARGV')
    return args


def expected_receipt(path, binding, role):
    raw = path.read_bytes()
    validate(len(raw) <= 32768, 'RECEIPT_SIZE')
    rows = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get('request') == binding['request']:
            rows.append(row)
    validate(len(rows) == 1 and all(rows[0][k] == v for k,v in binding.items())
             and rows[0]['role'] == role and rows[0]['passed'], 'ENDPOINT_RECEIPT')
    return rows[0], sha(raw)


def transfer(plan, publication):
    LOCAL.mkdir(mode=0o700)
    epoch = time.time()
    started = time.monotonic()
    budget = plan['timeout_seconds']
    binding = {k:plan[k] for k in ('request','file','bytes','SHA256')}
    binding.update(publication=publication, epoch=epoch, deadline=epoch+budget)
    save(LOCAL/'intent.json', binding)
    children = []
    lock = threading.Lock()

    def remaining():
        return min(binding['deadline']-time.time(), started+budget-time.monotonic())

    def stop():
        with lock:
            for child in reversed(children):
                if child.poll() is None:
                    try:
                        child.kill()
                    except ProcessLookupError:
                        pass

    total = len(frame(binding)) + plan['bytes']
    timer = threading.Timer(max(0, remaining()), stop)
    timer.daemon = True
    sender_log = LOCAL/'sender.stderr.private'
    receiver_log = LOCAL/'receiver.stdout.private'
    copied = 0
    report = 0
    result = dict(binding, passed=False, local_weight_bytes_written=0)
    try:
        receiver_cmd = command(plan, binding, 'receiver', budget)
        sender_cmd = command(plan, binding, 'sender', budget)
        with sender_log.open('xb') as se, receiver_log.open('xb') as ro, (LOCAL/'receiver.stderr.private').open('xb') as re:
            timer.start()
            with lock:
                receiver = subprocess.Popen(receiver_cmd, stdin=subprocess.PIPE, stdout=ro, stderr=re, cwd=REPO)
                children.append(receiver)
                sender = subprocess.Popen(sender_cmd, stdout=subprocess.PIPE, stderr=se, cwd=REPO)
                children.append(sender)
            save(LOCAL/'transports.json', dict(sender_pid=sender.pid, receiver_pid=receiver.pid, request=plan['request']))
            while True:
                validate(remaining() > 12, 'RELAY_CLOSING_RESERVE')
                chunk = sender.stdout.read(CHUNK)
                if not chunk:
                    break
                copied += len(chunk)
                validate(copied <= total and receiver.stdin.write(chunk) == len(chunk), 'BOUND_PIPE_TRANSFER')
                delay = max(0, started+copied/plan['rate_bytes_per_second']-time.monotonic())
                validate(delay+12 < remaining(), 'RELAY_PACE_EXHAUSTS_RESERVE')
                time.sleep(delay)
                if copied - report >= CHECKPOINT_SIZE:
                    report = copied
                    stage = LOCAL/'progress.tmp'
                    save(stage, dict(request=plan['request'], wire_bytes=copied, observed_epoch=time.time()))
                    stage.replace(LOCAL/'progress.json')
            validate(copied == total and sender.wait(timeout=max(.1,remaining())) == 0, 'SOURCE_TERMINAL')
            _, result['sender_receipt_SHA256'] = expected_receipt(sender_log,binding,'sender')
            validate(receiver.stdin.write(COMMIT) == len(COMMIT), 'SHORT_COMMIT_FRAME')
            receiver.stdin.close()
            validate(receiver.wait(timeout=max(.1,remaining())) == 0, 'TARGET_TERMINAL')
            row, result['receiver_receipt_SHA256'] = expected_receipt(receiver_log,binding,'receiver')
            validate(row['whole_SHA256'] == plan['SHA256'] and row['durable_bytes'] == plan['bytes'], 'TARGET_PROOF')
            result['passed'] = True
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        result.update(error=type(exc).__name__,
                      failure_code=str(exc) if isinstance(exc, ValueError) else None)
    finally:
        timer.cancel()
        stop()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                result['local_cleanup_pending'] = True
        result.update(wire_bytes=copied, transport_rcs=[c.returncode for c in children],
                      elapsed_seconds=time.monotonic()-started)
        save(LOCAL/'outcome.json', result)
    print(json.dumps(result,allow_nan=False),flush=True)
    return int(not result['passed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('publication')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    plan = open_published(args.publication)
    if not args.audit:
        return transfer(plan,args.publication)
    binding = json.loads((LOCAL/'intent.json').read_bytes())
    validate(binding['publication'] == args.publication and binding['request'] == plan['request'], 'ORIGINAL_REQUEST')
    r = subprocess.run(command(plan,binding,'audit',480), capture_output=True, text=True, timeout=490)
    validate(r.returncode == 0 and 1 <= len(r.stdout) <= 16384, 'INDEPENDENT_AUDIT_RECEIPT')
    value = json.loads(r.stdout)
    validate(value['verified'] and value['bytes']==plan['bytes'] and value['SHA256']==plan['SHA256'], 'INDEPENDENT_WHOLE_SHA')
    save(LOCAL/'independent.json',value)
    print(json.dumps(value,allow_nan=False),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
