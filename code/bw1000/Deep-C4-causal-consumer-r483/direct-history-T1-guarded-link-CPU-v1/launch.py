"""Send the frozen zero-GPU link packet through the approved existing transport."""

import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[3]
LABEL = 'DeepC4-T1-history-guarded-probe-link-v1'
REQUEST = 'ae85c97293ae410483e55d94383a6c48'
CLAIM = 'coordination/tasks/deep-c4-direct-history-guarded-link-cpu-v1.json'
REMOTE_BRANCH = 'refs/heads/bw1000-experiments'


def load_packet(publication, query):
    if not re.fullmatch('[0-9a-f]{40}', publication):
        raise ValueError('expected the ordinary published commit')
    if query('ls-remote', '--exit-code', 'origin', REMOTE_BRANCH).decode().split() != [publication, REMOTE_BRANCH]:
        raise ValueError('ordinary experiment head differs')
    manifest_file = PACKAGE / 'SOURCE_FILES.json'
    raw = manifest_file.read_bytes()
    manifest_link = manifest_file.relative_to(REPO).as_posix()
    if query('cat-file', 'blob', publication + ':' + manifest_link) != raw:
        raise ValueError('published manifest differs')
    mf = sha256(raw).hexdigest()
    claim = json.loads(query('cat-file', 'blob', publication + ':' + CLAIM))
    placement = dict(parent='1103606', node='f13r2n01', cpus=4, memory_MiB=8192, GPU_cards=0,
                     original_deadline_seconds=300)
    if (claim['label'], claim['request'], claim['manifest_SHA256'], claim['placement']) != (
        LABEL, REQUEST, mf, placement,
    ) or claim['CPU_GO'] is not True or claim['GPU_GO'] is not False:
        raise ValueError('unexpected published CPU placement or scope')
    files = {'SOURCE_FILES.json': raw}
    for path, pin in json.loads(raw)['files'].items():
        data = query('cat-file', 'blob', publication + ':' + pin['path'])
        if data != (REPO / pin['path']).read_bytes() or sha256(data).hexdigest() != pin['SHA256']:
            raise ValueError('published source bytes differ: ' + path)
        files[path] = data
    payload = dict(request=REQUEST, publication=publication, manifest_SHA256=mf,
                   files={path: base64.b64encode(data).decode('ascii') for path, data in files.items()})
    body = ('PAYLOAD = ' + repr(payload) + '\n').encode('ascii') + files['receive.py']
    if len(body) > 128000:
        raise ValueError('source packet exceeds the bounded transport size')
    return body, mf


def main(publication, transport):
    git = ['git', '--no-optional-locks', '-c', 'safe.directory=' + REPO.as_posix(), '-C', str(REPO)]

    def query(*args):
        return subprocess.check_output([*git, *args], timeout=15)

    body, mf = load_packet(publication, query)
    work = REPO / 'local-only' / LABEL
    work.mkdir(mode=0o700)
    identity = dict(request=REQUEST, publication=publication, manifest_SHA256=mf)
    (work / 'intent.json').write_text(json.dumps(dict(identity, body_SHA256=sha256(body).hexdigest(),
                                                      replay_allowed=False)) + '\n', encoding='utf-8')
    outcome = dict(identity, remote_outcome='UNKNOWN', rc=None)
    try:
        with (work / 'stdout.private').open('xb') as out, (work / 'stderr.private').open('xb') as err:
            result = transport(body, out, err)
        outcome['rc'] = result.returncode
        if result.returncode == 0:
            receipt = json.loads((work / 'stdout.private').read_bytes())
            if any(receipt.get(key) != value for key, value in identity.items()) or receipt.get('GPU') is not False:
                raise ValueError('target launch receipt identity differs')
            outcome.update(remote_outcome='OWNED_SRUN_STARTED', receipt=receipt)
    except subprocess.TimeoutExpired:
        pass
    finally:
        (work / 'transport.json').write_text(json.dumps(outcome) + '\n', encoding='utf-8')
    print(json.dumps(outcome), flush=True)
    return int(outcome['remote_outcome'] != 'OWNED_SRUN_STARTED')
