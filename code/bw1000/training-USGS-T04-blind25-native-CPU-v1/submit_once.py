"""One ordinary A100 SSH invocation; no retries, fallback, secret reads or polling."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import uuid

D = Path(__file__).resolve().parent
G = D.parents[2]
LABEL = D.name
SSH = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=yes',
       '-F', 'C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config', 'a100-27',
       '/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -u -']


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args, budget):
    left = budget()
    r = subprocess.run(['git', '--no-optional-locks', '-c', 'safe.directory=' + str(G), *args], cwd=G,
        capture_output=True, timeout=min(15, left), check=False)
    need(r.returncode == 0, 'ORDINARY_GIT_READ_FAILED')
    return r.stdout


def main(a):
    epoch = time.time(); monotonic_deadline = time.monotonic() + 295
    def budget():
        left = min(epoch + 295-time.time(), monotonic_deadline-time.monotonic())
        need(left > 0, 'ORIGINAL300_CLIENT_BUDGET_EXHAUSTED'); return left
    need(a.root_go and re.fullmatch('[0-9a-f]{40}', a.publication), 'EXPLICIT_ROOT_PUBLICATION_REQUIRED')
    from verify_package import verify
    check = verify(budget=budget); need(check['root_GO'] is True, 'PACKAGE_NOT_ADOPTED')
    need(git('remote', 'get-url', 'origin', budget=budget).decode().strip() == 'git@github.com:whjthu/InfiniLM-hygon-test.git', 'EXPECTED_ORIGIN')
    need(git('rev-parse', 'HEAD', budget=budget).decode().strip() == a.publication, 'EXACT_PUBLISHED_HEAD')
    remote = git('ls-remote', 'origin', 'refs/heads/bw1000-experiments', budget=budget).decode().strip().split()
    need(remote == [a.publication, 'refs/heads/bw1000-experiments'], 'ORDINARY_REMOTE_MUST_EQUAL_HEAD')
    git('diff', '--cached', '--quiet', budget=budget)
    raw = (D / 'SOURCE_FILES.json').read_bytes(); mf = json.loads(raw)
    rel = D.relative_to(G).as_posix()
    package = {}
    for name in [*mf['files'], 'SOURCE_FILES.json']:
        budget()
        body = (D / name).read_bytes()
        need(git('show', a.publication + ':' + rel + '/' + name, budget=budget) == body, 'UNPUBLISHED_OR_DIRTY_SOURCE_BYTES')
        package[name] = base64.b64encode(body).decode('ascii')
    claim = (D / 'CLAIM.json').read_bytes()
    out = G / 'local-only' / (LABEL + '-posix-path-corrected-dispatch')
    out.mkdir(mode=0o700, exist_ok=False)  # Even an UNKNOWN attempt can never be invoked again.
    budget(); request = uuid.uuid4().hex
    from prepare_inputs import prepare
    plan = json.loads((D/'PLAN.json').read_bytes())
    adapter = out/'adapter'
    data_binding = prepare(Path(plan['original_dataset_root']), adapter, Path(plan['namespace'])/'data', budget)
    data = {}
    for name in [*plan['media'], 'blind-swift.jsonl', 'data-binding-private.json']:
        budget(); data[name] = base64.b64encode((adapter/name).read_bytes()).decode('ascii')
    payload = dict(case='T04.sft', data=data, request_id=request, publication=a.publication, epoch=epoch, source_manifest_SHA256=sha(raw),
                   claim_SHA256=sha(claim), package=package)
    packed = base64.b64encode(json.dumps(payload, separators=(',', ':')).encode()).decode('ascii')
    body = ("INSTALL_PAYLOAD_B64 = '" + packed + "'\n").encode() + (D / 'install_and_run.py').read_bytes()
    (out / 'body.py').write_bytes(body)
    intent = dict(request_id=request, publication=a.publication, epoch=epoch, deadline=epoch+300,
        source_manifest_SHA256=sha(raw), claim_SHA256=sha(claim), body_bytes=len(body), body_SHA256=sha(body),
        max_invocations=1, exec_request_attempted=False, remote_outcome='NOT_INVOKED', replay_allowed=False)
    (out / 'intent.json').write_text(json.dumps(intent, indent=2)+'\n', encoding='utf8')
    stdout = b''; stderr = b''; rc = None; error = None
    try:
        left = budget()
        intent['exec_request_attempted'] = True
        (out / 'attempt.json').write_text(json.dumps(dict(request_id=request, epoch=epoch, exec_request_attempted=True))+'\n', encoding='utf8')
        r = subprocess.run(SSH, input=body, capture_output=True, timeout=left, check=False)
        stdout, stderr, rc = r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = exc.stdout or b'', exc.stderr or b''; error = type(exc).__name__
    except BaseException as exc:
        error = type(exc).__name__
    (out / 'stdout.private').write_bytes(stdout); (out / 'stderr.private').write_bytes(stderr)
    remote = None
    if rc is not None:
        try:
            candidate = json.loads(stdout)
            if candidate.get('request_id') == request:
                remote = candidate
        except (ValueError, TypeError):
            pass
    result = dict(**intent, rc=rc, stdout_bytes=len(stdout), stdout_SHA256=sha(stdout),
        stderr_bytes=len(stderr), stderr_SHA256=sha(stderr), error_type=error, finished_epoch=time.time(),
        engineering_completed=remote.get('passed') if remote else None,
        release_permitted=(remote.get('passed') is True) if remote else None)
    result['remote_outcome'] = 'KNOWN_TERMINAL' if remote else 'UNKNOWN'
    (out / 'receipt.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k: result[k] for k in ['request_id','rc','remote_outcome','engineering_completed','release_permitted','replay_allowed']}))
    return 0 if remote and remote.get('passed') is True else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--publication', required=True); p.add_argument('--root-go', action='store_true')
    raise SystemExit(main(p.parse_args()))
