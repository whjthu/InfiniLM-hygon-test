"""Verify the ordinary published packet, then send it through the supplied existing transport."""
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import uuid


SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[3]
LABEL = 'GLM-I128-gate-rows-gfx936-CPU-v1'


def verify(publication):
    if re.fullmatch('[0-9a-f]{40}', publication) is None:
        raise ValueError('EXACT_PUBLICATION')
    git = ['git','--no-optional-locks','-c','safe.directory='+REPO.as_posix(),'-C',str(REPO)]
    query = lambda *a: subprocess.check_output([*git,*a],timeout=15)
    if query('ls-remote','--exit-code','origin','refs/heads/bw1000-experiments').decode().split() != [
            publication,'refs/heads/bw1000-experiments']:
        raise ValueError('ACTUAL_PUBLISHED_HEAD')
    location = SOURCE.relative_to(REPO).as_posix()
    raw = (SOURCE/'SOURCE_FILES.json').read_bytes()
    files = {'SOURCE_FILES.json': raw}
    if query('show',publication+':'+location+'/SOURCE_FILES.json') != raw:
        raise ValueError('PUBLISHED_SOURCE_MANIFEST')
    for name, pin in json.loads(raw)['files_SHA256'].items():
        if Path(name).name != name:
            raise ValueError('MANIFEST_MEMBER')
        data = (SOURCE/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != pin or query('show',publication+':'+location+'/'+name) != data:
            raise ValueError('PUBLISHED_SOURCE_DRIFT')
        files[name] = data
    claim_path = 'coordination/tasks/glm-i128-gate-rows-gfx936-cpu-v1.json'
    claim = json.loads(query('show',publication+':'+claim_path))
    mf = hashlib.sha256(raw).hexdigest()
    if claim['label'] != LABEL or claim['manifest_SHA256'] != mf or not claim['CPU_GO'] or claim['GPU_GO']:
        raise ValueError('PUBLISHED_RESOURCE_CLAIM')
    return files, mf


def main(publication, transport):
    files, mf = verify(publication)
    request = uuid.uuid4().hex
    payload = dict(publication=publication,request=request,manifest_SHA256=mf,
                   files={name:base64.b64encode(raw).decode('ascii') for name,raw in files.items()})
    body = ('PAYLOAD = '+repr(payload)+'\n').encode('ascii') + files['receive.py']
    if len(body) > 160000:
        raise ValueError('EXACT_PAYLOAD_BOUND')
    work = REPO/'local-only'/LABEL
    work.mkdir(mode=0o700)
    (work/'intent.json').write_text(json.dumps(dict(request=request,publication=publication,manifest_SHA256=mf,
        body_SHA256=hashlib.sha256(body).hexdigest(),replay_allowed=False))+'\n')
    outcome = dict(request=request,remote_outcome='UNKNOWN')
    try:
        with (work/'stdout.private').open('xb') as out,(work/'stderr.private').open('xb') as err:
            result = transport(body,out,err)
        outcome['rc'] = result.returncode
        if result.returncode == 0:
            receipt = json.loads((work/'stdout.private').read_bytes())
            if (receipt['request'],receipt['publication'],receipt['manifest_SHA256']) != (request,publication,mf):
                raise ValueError('REMOTE_RECEIPT_IDENTITY')
            outcome.update(remote_outcome='OWNED_SRUN_STARTED',receipt=receipt)
    except subprocess.TimeoutExpired:
        outcome['rc'] = None
    finally:
        (work/'transport.json').write_text(json.dumps(outcome)+'\n')
    print(json.dumps(outcome),flush=True)
    return int(outcome['remote_outcome'] != 'OWNED_SRUN_STARTED')
