"""Prepare the published original GLM component sources for an authenticated controller."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[3]
AREA = 'code/bw1000/GLM-C1-layer-observer-r474'
CONTROL = 'tools/control_glm_i128_component.py'
INDEX = 'coordination/tuning/glm-i128-component-controller-files-v1.json'
INDEX_SHA = '3e222de5c1310839ef7a9533626ac21911c2413b06267d700fb0b72c8a3bea3e'
AUDITOR = f'{AREA}/routed-i128-cpu-artifact-audit-v1/audit_once.py'
AUDITOR_SHA = '5db40da457f7b1449b30764bb9c224a5cd4f50824555b2fa866294aedca5df46'
LINK_LABEL = 'GLM-I128-five-isolated-providers-CPU-v1'
GPU_LABEL = 'GLM-I128-four-routed-variants-one-device-v1'


def published_blob(publication, name, pin=None):
    local = (REPO / name).read_bytes()
    args = ['git', '--no-optional-locks', '-c', 'safe.directory=' + REPO.as_posix(),
            '-C', str(REPO), 'cat-file', 'blob', publication + ':' + name]
    if (pin is not None and sha256(local).hexdigest() != pin) or subprocess.check_output(args, timeout=10) != local:
        raise ValueError(f'published source pin differs: {name}')
    return local


def depot(publication):
    index = json.loads(published_blob(publication, INDEX, INDEX_SHA))
    files = {name: published_blob(publication, name, pin) for name, pin in index.items()}
    audit = published_blob(publication, AUDITOR, AUDITOR_SHA)
    files[AUDITOR] = audit
    namespace = {'__file__': str(REPO / AUDITOR), '__name__': 'i128_scope'}
    exec(compile(audit, AUDITOR, 'exec'), namespace)
    for arm, (_, _, mf_sha, claim_sha) in namespace['BRANCHES'].items():
        prefix = f'{AREA}/routed-i128-{arm}-rows-CPU-v1'
        name = prefix + '/SOURCE_FILES.json'
        members = files[name] = published_blob(publication, name, mf_sha)
        for member, expected in json.loads(members)['files_SHA256'].items():
            path = prefix + '/' + member
            files[path] = published_blob(publication, path, expected)
        name = f'coordination/tasks/glm-i128-{arm}-rows-gfx936-cpu-v1.json'
        files[name] = published_blob(publication, name, claim_sha)
    caller = f'{AREA}/routed-i128-one-gpu-caller-v1'
    for member, expected in json.loads(files[caller + '/SOURCE_FILES.json'])['files_SHA256'].items():
        path = caller + '/' + member
        files[path] = published_blob(publication, path, expected)
    name = f'{AREA}/attention-detail-CPU-v1/BUILD_PLAN.json'
    files[name] = published_blob(publication, name, '7bb0a4ebc72612b00e4498ac0542dd258dacb853c1d151aad311637bd2365fca')
    source = files[CONTROL]
    for previous, current in (
        (b'    print(json.dumps(info), flush=True)', b"    link.write(run / (phase + '-device.json'), info)"),
        (b"small_json(run/'pre.log')", b"small_json(run/'pre-device.json')"),
        (b"small_json(run/'post.log')", b"small_json(run/'post-device.json')"),
    ):
        if source.count(previous) != 1:
            raise ValueError('original structured snapshot boundary differs')
        source = source.replace(previous, current)
    compile(source, CONTROL, 'exec')
    files[CONTROL] = source
    index[CONTROL] = sha256(source).hexdigest()
    files[INDEX] = (json.dumps(index, sort_keys=True, allow_nan=False) + '\n').encode()
    return files


def packet(publication, mode, claim_path, original_audits=None, link_sha=None):
    if mode not in ('link', 'component'):
        raise ValueError('scope must be link or component')
    files = depot(publication)
    raw = published_blob(publication, claim_path)
    claim = json.loads(raw)
    pins = {name: sha256(data).hexdigest() for name, data in files.items()}
    if claim['depot'] != pins or claim['mode'] != mode:
        raise ValueError('the installed source set is outside this claim')
    package = PACKAGE.relative_to(REPO).as_posix()
    source = published_blob(publication, package + '/receive.py')
    payload = dict(publication=publication, mode=mode, claim=raw, files=files,
                   original_audits=original_audits, link_sha=link_sha)
    return ('PAYLOAD=' + repr(payload) + '\n').encode() + source
