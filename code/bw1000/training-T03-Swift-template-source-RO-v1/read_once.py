"""Root-published single existing-A100 SSH read. Third-party bodies stay ignored."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid

D = Path(__file__).resolve().parent
G = D.parents[2]
RUN = G / 'local-only/training-T03-Swift-template-source-RO-v1-original'


def need(ok, why):
    if not ok: raise ValueError(why)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def write(name, value):
    with (RUN / name).open('xb') as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2) + '\n').encode())


def main(a):
    epoch = time.time(); deadline = epoch + 60
    need(a.root_go and re.fullmatch('[0-9a-f]{40}', a.publication), 'root publication GO required')
    scope = json.loads((D / 'SCOPE.json').read_bytes()); need(scope['root_GO'] is True, 'source RO scope not adopted')
    def git(*args):
        need(time.time() < deadline - 5, 'source gate exhausted original60epoch')
        return subprocess.check_output(['git', '-c', 'safe.directory=' + str(G), *args], cwd=G,
                                       timeout=max(.01, min(5, deadline - time.time() - 3)))
    relative = D.relative_to(G).as_posix()
    need(not git('status', '--porcelain', '--untracked-files=all', '--', relative).strip(), 'own source dirty/unpublished')
    need(not git('diff', '--cached', '--name-only').strip(), 'index not empty; serialize with root Git')
    mfraw = (D / 'SOURCE_FILES.json').read_bytes(); mf = json.loads(mfraw)['files']
    need(git('show', a.publication + ':' + relative + '/SOURCE_FILES.json') == mfraw, 'published manifest differs')
    for name, pin in mf.items():
        raw = (D / name).read_bytes()
        need(len(raw) == pin['bytes'] and sha(raw) == pin['SHA256'] and git('show', a.publication + ':' + relative + '/' + name) == raw,
             'published source differs')
    prior = scope['prior_terminal_failure']; raw = git('show', a.publication + ':' + prior['path'])
    need(len(raw) == prior['bytes'] and sha(raw) == prior['SHA256'], 'published original failure differs')
    fail = json.loads(raw)
    need(fail['request_id'] == '1e0a5eb9fac94bcd8a5f2387908a3545' and fail['worker_returncode'] == 1
         and fail['actual_completed_train'] == fail['actual_completed_dev'] == 0 and fail['independent_postclean'] is True,
         'original failure is not closed')
    RUN.mkdir(mode=0o700)  # Exclusive original scope; unknown never replays.
    request = uuid.uuid4().hex
    context = dict(request_id=request, epoch=epoch, deadline=deadline, publication=a.publication, files=scope['files'])
    body = ('READ_CONTEXT=' + repr(context) + '\n').encode() + (D / 'remote_read.py').read_bytes()
    ast.parse(body)
    with (RUN / 'body.py').open('xb') as stream: stream.write(body)
    write('intent.json', dict(**context, body_SHA256=sha(body), source_manifest_SHA256=sha(mfraw), source_files=2,
                              source_bytes=43908, GPU=0, read_only=True, replay_allowed=False))
    command = shlex.join(['/home/whj/nvidia-baseline/env-vlm/bin/python', '-I', '-S', '-B', '-'])
    argv = ['ssh', '-F', 'C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config',
            '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=8', 'a100-27', command]
    forced = False
    try:
        result = subprocess.run(argv, input=body, capture_output=True, timeout=max(.01, deadline - time.time()))
        out, err, rc = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        forced = True; out, err, rc = error.stdout or b'', error.stderr or b'', None
    for name, raw in [('stdout.private', out), ('stderr.private', err)]:
        with (RUN / name).open('xb') as stream: stream.write(raw)
    outcome = dict(request_id=request, epoch=epoch, deadline=deadline, finished=time.time(), returncode=rc,
                   forced_local_SSH_kill=forced, remote_outcome='UNKNOWN' if forced or rc == 255 else 'RETURNED',
                   stdout_bytes=len(out), stdout_SHA256=sha(out), stderr_bytes=len(err), stderr_SHA256=sha(err),
                   source_files_saved=0, passed=False, GPU=0, provider_import=False, native_template_run=False)
    try:
        if rc == 0 and not forced:
            need(len(out) <= 96 * 1024, 'response cap exceeded'); value = json.loads(out)
            need(value['request_id'] == request and value['epoch'] == epoch and value['deadline'] == deadline
                 and value['passed'] is True and value['logical_source_bytes'] == 43908 and len(value['files']) == 2,
                 'source response context/denominator differs')
            pins = {p['name']: p for p in scope['files']}; need(set(pins) == {r['name'] for r in value['files']}, 'response source names differ')
            target = RUN / 'source-files'; target.mkdir(mode=0o700)
            for row in value['files']:
                pin = pins[row['name']]; raw = base64.b64decode(row['source_base64'], validate=True)
                need(row['canonical_path'] == pin['canonical_path'] and len(raw) == row['bytes'] == pin['bytes']
                     and sha(raw) == row['SHA256'] == pin['SHA256'], 'preserved source differs')
                with (target / row['name']).open('xb') as stream: stream.write(raw)
                outcome['source_files_saved'] += 1
            outcome['passed'] = True
    finally:
        write('outcome.json', outcome)
    print(json.dumps({k: outcome[k] for k in ['request_id', 'returncode', 'remote_outcome', 'passed', 'source_files_saved', 'stdout_bytes', 'stderr_bytes', 'GPU']}))
    return 0 if outcome['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--publication', required=True); p.add_argument('--root-go', action='store_true')
    raise SystemExit(main(p.parse_args()))
