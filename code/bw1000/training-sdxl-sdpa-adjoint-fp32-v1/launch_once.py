"""Stage and start one published package using the coordinator's authorized transport."""

import base64
import json
import os
from pathlib import Path
import re
import subprocess
import time

from render_target import HERE, LABEL, REQUEST, digest, render


REPO = HERE.parents[2]
LOCAL = REPO/'local-only'/f'{LABEL}-original-dispatch'
ORIGIN = 'git@github.com:whjthu/InfiniLM-hygon-test.git'
BRANCH = 'refs/heads/bw1000-experiments'
TARGET = '/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/' + LABEL + '-' + REQUEST


REMOTE = '''import base64, hashlib, json, os, pathlib, subprocess, sys, time
epoch = time.time()
root = pathlib.Path(payload['target'])
root.mkdir(mode=0o700)
for name, b64 in payload['members'].items():
    assert pathlib.Path(name).name == name
    with (root/name).open('xb') as handle:
        handle.write(base64.b64decode(b64, validate=True))
        handle.flush()
        os.fsync(handle.fileno())
manifest = (root/'SOURCE_FILES.json').read_bytes()
assert hashlib.sha256(manifest).hexdigest() == payload['manifest_SHA256']
files = json.loads(manifest)['files']
assert set(files)|{'SOURCE_FILES.json'} == set(payload['members'])
for name,pin in files.items():
    raw = (root/name).read_bytes()
    assert pin == dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())
with (root/'controller.log').open('xb') as output:
    child = subprocess.Popen([sys.executable, '-B', str(root/'campaign.py'), str(epoch), payload['request']],
                             stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {k: payload[k] for k in ('request', 'publication', 'manifest_SHA256')}
receipt.update(epoch=epoch, controller_PID=child.pid)
try:
    fields = pathlib.Path('/proc',str(child.pid),'stat').read_text().rsplit(') ',1)[1].split()
    receipt['controller_startticks'] = int(fields[19])
except FileNotFoundError:
    receipt['controller_startticks'] = None
with (root/'launcher-owner.json').open('x', encoding='utf-8') as output:
    json.dump(receipt, output, allow_nan=False)
    output.flush()
    os.fsync(output.fileno())
print(json.dumps(receipt), flush=True)
'''


def bootstrap(publication, bundle):
    assert re.fullmatch('[0-9a-f]{40}', publication)
    manifest = digest(bundle['SOURCE_FILES.json'])
    payload = dict(request=REQUEST, publication=publication, target=TARGET, manifest_SHA256=manifest,
                   members={name: base64.b64encode(raw).decode('ascii') for name, raw in bundle.items()})
    script = (f'payload = {payload!r}\n' + REMOTE).encode()
    assert len(script) <= 128*1024
    return script, manifest


def published_files(publication, receipt_name='TARGET_BUILD_FILES.json'):
    def git(*args):
        return subprocess.check_output(['git','-c','safe.directory='+REPO.as_posix(),*args],
                                       cwd=REPO, timeout=25, stdin=subprocess.DEVNULL)

    assert git('remote', 'get-url', 'origin').decode().strip() == ORIGIN
    assert git('ls-remote', 'origin', BRANCH).decode().split() == [publication, BRANCH]
    relative = HERE.relative_to(REPO).as_posix()
    raw = (HERE/receipt_name).read_bytes()
    assert git('cat-file', 'blob', f'{publication}:{relative}/{receipt_name}') == raw
    receipt = json.loads(raw)
    for name, pin in receipt['files'].items():
        assert Path(name).name == name
        data = (HERE/name).read_bytes()
        assert pin == dict(bytes=len(data), SHA256=digest(data)), name
        assert git('cat-file', 'blob', f'{publication}:{relative}/{name}') == data, name
    return receipt['generated_target_manifest_SHA256']


def save(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def launch(publication, request, transport, *, local=LOCAL, check_publication=published_files):
    """The coordinator holds the original node and passes the existing transport."""
    assert request == REQUEST and callable(transport) and re.fullmatch('[0-9a-f]{40}', publication)
    script, manifest = bootstrap(publication, render())
    assert check_publication(publication) == manifest, 'published generated package differs'
    local.mkdir(mode=0o700, exist_ok=False)
    intent = dict(request=request, publication=publication, client_epoch=time.time(), target=TARGET,
                  body_SHA256=digest(script), manifest_SHA256=manifest, replay_allowed=False)
    save(local/'intent.json', intent)
    result = dict(request=request, remote_outcome='UNKNOWN', returned=None, replay_allowed=False)
    try:
        with (local/'stdout.private').open('xb') as out, (local/'stderr.private').open('xb') as err:
            execution = transport(script, out, err)
        result['returned'] = execution.returncode
    except (OSError, subprocess.SubprocessError) as error:
        result['transport_error'] = type(error).__name__
    finally:
        save(local/'transport.json', result)
    return result
