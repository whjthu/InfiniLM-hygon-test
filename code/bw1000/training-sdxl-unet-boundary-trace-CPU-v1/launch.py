"""Deploy the exact published probe once over the existing SSH connection."""
import base64
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time


SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[2]
LOCAL = REPO/'local-only'/SOURCE.name
HELPER = Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA = '3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
REMOTE_PARENT = '/public/home/aclsylqidf/hygon-inference-refactor-20261014/results'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(publication, request):
    manifest = (SOURCE/'SOURCE_FILES.json').read_bytes()
    relative = SOURCE.relative_to(REPO).as_posix()
    base = ['git','-c','safe.directory='+str(REPO)]

    def git(*args):
        return subprocess.check_output([*base,*args],cwd=REPO,timeout=10)

    claim = json.loads((SOURCE/'CLAIM.json').read_bytes())
    if re.fullmatch('[a-f0-9]{32}',request) is None or request != claim['request'] or not claim['CPU_GO']:
        raise RuntimeError('EXACT_RUN_CLAIM')
    if git('cat-file','blob',publication+':'+relative+'/SOURCE_FILES.json') != manifest:
        raise RuntimeError('EXACT_PUBLISHED_MANIFEST')
    if git('status','--short','--',relative).strip() or sha(HELPER.read_bytes()) != HELPER_SHA:
        raise RuntimeError('FROZEN_SOURCE_AND_CONNECTION_HELPER')
    files = {}
    for name, pin in json.loads(manifest)['files'].items():
        raw = (SOURCE/name).read_bytes()
        if len(raw) != pin['bytes'] or sha(raw) != pin['SHA256']:
            raise RuntimeError('EXACT_SOURCE_BYTES')
        files[name] = base64.b64encode(raw).decode()
    ssh = runpy.run_path(str(HELPER))['ssh']
    LOCAL.mkdir(parents=True,exist_ok=False)
    epoch = time.time()
    remote = REMOTE_PARENT+'/'+SOURCE.name+'-'+request
    payload = dict(epoch=epoch, request=request, remote=remote, files=files, publication=publication)
    bootstrap = '''import base64,json,os,pathlib,runpy,sys,time
p=PAYLOAD
host_epoch=time.time()
root=pathlib.Path(p['remote']);root.mkdir(mode=0o700)
for name,raw in p['files'].items():
 assert pathlib.Path(name).name==name
 with (root/name).open('xb') as f:f.write(base64.b64decode(raw,validate=True));f.flush();os.fsync(f.fileno())
sys.path.insert(0,str(root));sys.argv=[str(root/'dispatch.py'),str(host_epoch),p['request']]
runpy.run_path(sys.argv[0],run_name='__main__')
'''.replace('PAYLOAD',repr(payload))
    script = bootstrap.encode()
    if len(script) > 131072:
        raise RuntimeError('SOURCE_UPLOAD_CAP')
    (LOCAL/'intent.json').write_text(json.dumps(dict(request=request,client_epoch=epoch,client_timeout_seconds=245,
         remote=remote,publication=publication,body_SHA256=sha(script))),encoding='utf-8')
    result = dict(request=request,epoch=epoch,remote_outcome='UNKNOWN')
    try:
        with (LOCAL/'stdout.private').open('xb') as out,(LOCAL/'stderr.private').open('xb') as err:
            r = subprocess.run([ssh[0],'-o','StrictHostKeyChecking=yes',*ssh[1:],
                               '/usr/local/python3.12/bin/python3 -I -S -B -'],input=script,
                               stdout=out,stderr=err,cwd=REPO,timeout=245)
        result.update(remote_outcome='UNKNOWN' if r.returncode==255 else 'RETURNED',rc=r.returncode)
    except subprocess.TimeoutExpired:
        result.update(rc=None)
    (LOCAL/'transport.json').write_text(json.dumps(result),encoding='utf-8')
    print(json.dumps(result),flush=True)
    return int(result.get('rc')!=0)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1],sys.argv[2]))
