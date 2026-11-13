"""Independently verify the published VAE after the original relay succeeds."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess

TASK = 'coordination/tasks/training-hunyuanvideo-vae-a100-identity-v1.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stable_read(path, expected_bytes, limit=None):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), 'Expected a regular file')
    if expected_bytes is not None:
        require(before.st_size == expected_bytes, 'File size differs')
    if limit is not None:
        require(0 < before.st_size <= limit, 'Receipt size differs')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        opened = os.fstat(f.fileno())
        if limit is None:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
            data = None
        else:
            data = f.read(limit + 1)
            require(len(data) == before.st_size, 'Receipt size changed')
            digest = sha(data)
        after = os.fstat(f.fileno())
    final = path.lstat()
    def identity(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    require(identity(before) == identity(opened) == identity(after) == identity(final),
            'File changed during verification')
    return digest, data


def audit(plan, transfer):
    root = Path(plan['target_root'])
    scope = root / 'a100-verified-vae-suffix-v1'
    require(scope.is_dir() and not scope.is_symlink(), 'Original receiver scope differs')
    intent_sha, intent_raw = stable_read(scope / 'intent.json', None, 32768)
    terminal_sha, terminal_raw = stable_read(scope / 'outcome.json', None, 32768)
    identity = {k: transfer[k] for k in ('claim', 'request_id', 'epoch')}
    intent = json.loads(intent_raw)
    pid = intent.get('pid')
    require(type(pid) is int and pid > 0 and intent.get('passed') is False and
            intent == dict(identity, role='receiver', passed=False, deadline=transfer['deadline'], pid=pid),
            'Wrong original receiver intent')
    terminal = json.loads(terminal_raw)
    name = plan['name']
    require(terminal.get('passed') is True and terminal == dict(
        identity, role='receiver', passed=True, suffix_bytes=transfer['suffix_bytes'],
        files={name: plan['file']}), 'Wrong original receiver terminal')
    target = root / 'vae'
    require(target.is_dir() and not target.is_symlink() and not os.path.lexists(root / 'vae.incomplete'),
            'VAE publication incomplete')
    require({p.name for p in target.iterdir()} == {'config.json', name}, 'Published VAE file set differs')
    config, _ = stable_read(target / 'config.json', 462)
    require(config == plan['config_sha256'], 'VAE config hash differs')
    weight, _ = stable_read(target / name, plan['file']['bytes'])
    require(weight == plan['file']['SHA256'], 'VAE whole-file hash differs')
    return dict(passed=True, intent_SHA256=intent_sha, terminal_SHA256=terminal_sha,
                config_SHA256=config, weight_SHA256=weight, bytes=plan['file']['bytes'], GPU_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('commit')
    parser.add_argument('receipts', type=Path)
    parser.add_argument('connection', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    require(len(args.commit) == 40 and all(c in '0123456789abcdef' for c in args.commit),
            'Expected a published commit SHA')
    git = ['git', '-c', 'safe.directory=' + Path.cwd().as_posix()]
    def blob(commit, path):
        return subprocess.check_output(git + ['show', commit + ':' + path], timeout=15)
    task = json.loads(blob(args.commit, TASK))
    program = Path(__file__).read_bytes()
    require(program == blob(args.commit, task['source']), 'Local verifier differs from published source')
    plan_raw = blob(task['transfer']['claim'], task['plan']['path'])
    require(sha(plan_raw) == task['plan']['SHA256'], 'Published transfer plan differs')
    plan = json.loads(plan_raw)
    old = blob(args.commit, task['receipt_reader']['path'])
    require(sha(old) == task['receipt_reader']['SHA256'], 'Published receipt reader differs')
    ns = {'__name__': 'published_receipt_reader'}
    exec(compile(old, 'published_receipt_reader', 'exec'), ns)
    proofs = ns['check_local_receipts'](args.receipts, task['transfer'], {plan['name']: plan['file']})
    require(sha(args.connection.read_bytes()) == task['connection_SHA256'], 'Connection helper differs')
    spec = importlib.util.spec_from_file_location('connection', args.connection)
    connection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connection)
    seconds = task['max_seconds']
    require(type(seconds) is int and 0 < seconds <= 300, 'Invalid deadline')
    body = ("import signal\n"
            "def expired(*_): raise TimeoutError('VAE verification deadline')\n"
            "signal.signal(signal.SIGALRM, expired)\n"
            f"signal.alarm({seconds})\n"
            f"namespace = {{'__name__': 'published_vae_identity'}}\n"
            f"exec(compile({program!r}, 'published_vae_identity', 'exec'), namespace)\n"
            f"print(namespace['json'].dumps(namespace['audit']({plan!r}, {task['transfer']!r})))\n").encode()
    remote = '/usr/local/python3.12/bin/python3 -B -c ' + shlex.quote(
        'import sys;exec(compile(sys.stdin.buffer.read(),"vae_identity","exec"))')
    args.output.mkdir(mode=0o700)
    with (args.output / 'intent.json').open('x', encoding='utf8') as f:
        json.dump(dict(commit=args.commit, request_id=task['transfer']['request_id'],
                       payload_SHA256=sha(body), **proofs), f)
    result = dict(passed=False, remote_outcome='UNKNOWN')
    try:
        run = subprocess.run(connection.ssh + [remote], input=body, capture_output=True, timeout=seconds + 25)
        result.update(rc=run.returncode, stderr_SHA256=sha(run.stderr))
        if run.returncode == 0:
            value = json.loads(run.stdout)
            require(value['passed'] is True and value['weight_SHA256'] == plan['file']['SHA256']
                    and value['config_SHA256'] == plan['config_sha256'] and value['GPU_calls'] == 0
                    and value['bytes'] == plan['file']['bytes'], 'Independent result differs')
            result.update(passed=True, remote_outcome='VERIFIED', result=value)
        elif run.returncode == 1 and any(x in run.stderr for x in (b'ValueError:', b'TimeoutError:')):
            result['remote_outcome'] = 'VERIFICATION_FAILED'
    except (ValueError, KeyError, OSError, subprocess.TimeoutExpired) as error:
        result['error_type'] = type(error).__name__
    with (args.output / 'outcome.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
