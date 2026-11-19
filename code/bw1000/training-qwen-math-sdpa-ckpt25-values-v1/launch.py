"""Launch the published in-place CPU checkpoint value comparison."""
import base64
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
GIT=ROOT.parents[2]
HELPER=Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(commit):
    manifest=(ROOT/'SOURCE_FILES.json').read_bytes()
    remote_blob=subprocess.check_output(['git','-c','safe.directory='+str(GIT),
            'cat-file','blob',commit+':'+ROOT.relative_to(GIT).as_posix()+'/SOURCE_FILES.json'],cwd=GIT,timeout=12)
    if manifest!=remote_blob or sha(HELPER.read_bytes())!='3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1':
        raise RuntimeError('Published CPU scope or connection helper differs')
    validate=json.loads(manifest)['files']
    artifacts={}
    for n in ('host.py','compute.py','CLAIM.json','launch.py'):
        raw=(ROOT/n).read_bytes()
        if (len(raw),sha(raw))!=(validate[n]['bytes'],validate[n]['SHA256']):
            raise RuntimeError('CPU source does not match the published member')
        artifacts[n]=raw
    claim=json.loads(artifacts['CLAIM.json'])
    intent=dict(request=claim['request_id'],claim=commit,
                package={'compute.py':base64.b64encode(artifacts['compute.py']).decode()})
    bootstrap=("import io,sys\nsys.stdin=io.StringIO("+repr(json.dumps(intent))+")\n").encode()+artifacts['host.py']
    local=GIT/'local-only'/ROOT.name
    local.mkdir(mode=0o700)
    (local/'intent.json').write_text(json.dumps({'request':intent['request'],'source_publication':commit,'payload_SHA256':sha(bootstrap)}),encoding='utf-8')
    ssh=runpy.run_path(str(HELPER))['ssh']
    result=dict(request=intent['request'], rc=None, result=None)
    try:
        run=subprocess.run([ssh[0],'-o','StrictHostKeyChecking=yes',*ssh[1:],'/usr/local/python3.12/bin/python3 -I -S -B -'],
                           input=bootstrap,capture_output=True,timeout=307)
        result.update(rc=run.returncode,stderr_SHA256=sha(run.stderr),stdout_SHA256=sha(run.stdout))
        lines=run.stdout.splitlines()
        if len(lines)==1 and len(lines[0])<=12000:
            result['result']=json.loads(lines[0])
    except subprocess.TimeoutExpired:
        result['transport_timed_out']=True
    (local/'outcome.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,allow_nan=False))
    return int(result['rc']!=0)


if __name__=='__main__':
    raise SystemExit(main(sys.argv[1]))
