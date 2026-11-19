"""Stage reviewed source only through the existing A100 SSH configuration."""
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
G = ROOT.parents[2]
CONFIG = Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(publication, request):
    g=['git','-c','safe.directory='+G.as_posix()]
    raw=(ROOT/'SOURCE_FILES.json').read_bytes()
    relative=ROOT.relative_to(G).as_posix()
    def git(*args):
        return subprocess.check_output([*g,*args],cwd=G,timeout=10)
    assert raw==git('cat-file','blob',publication+':'+relative+'/SOURCE_FILES.json')
    assert not git('status','--short','--',relative).strip() and CONFIG.is_file()
    claim=json.loads((ROOT/'CLAIM.json').read_bytes())
    assert bool(re.fullmatch('[a-f0-9]{32}',request)) and claim['request']==request and claim['GPU_GO']
    files={}
    for name,pin in json.loads(raw)['files'].items():
        body=(ROOT/name).read_bytes()
        assert len(body)==pin['bytes'] and sha(body)==pin['SHA256']
        files[name]=dict(body=base64.b64encode(body).decode(),pin=pin)
    directory=G/'local-only'/(ROOT.name+'-A100')
    directory.mkdir(parents=True,exist_ok=False)
    payload=dict(publication=publication,request=request,files=files,
                 output='/home/whj/nvidia-baseline/alignment-assets/'+ROOT.name+'-'+request)
    bootstrap="""import base64,hashlib,pathlib,runpy,sys,time
p=PAYLOAD
epoch=time.time()
d=pathlib.Path(p['output'])
d.mkdir(mode=0o700)
for name,item in p['files'].items():
    assert pathlib.Path(name).name==name
    data=base64.b64decode(item['body'],validate=True)
    assert len(data)==item['pin']['bytes'] and hashlib.sha256(data).hexdigest()==item['pin']['SHA256']
    with (d/name).open('xb') as f: f.write(data)
sys.path.insert(0,str(d))
sys.argv=[str(d/'dispatch_a100.py'),str(epoch),p['request']]
runpy.run_path(sys.argv[0],run_name='__main__')
""".replace('PAYLOAD',repr(payload),1).encode()
    assert len(bootstrap)<=131072
    (directory/'intent.json').write_text(json.dumps(dict(request=request,publication=publication,
        source_body_SHA256=sha(bootstrap),CPU_GPU_host='existing_a100_26')),encoding='utf-8')
    rc=None
    with (directory/'stdout.private').open('xb') as out,(directory/'stderr.private').open('xb') as err:
        try:
            rc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-o',
                               'StrictHostKeyChecking=yes','-F',str(CONFIG),'a100-26','python3 -B -u -'],
                              input=bootstrap,stdout=out,stderr=err,timeout=785).returncode
        except subprocess.TimeoutExpired:
            pass
    result=dict(request=request,rc=rc,remote_outcome='UNKNOWN' if rc in (None,255) else 'RETURNED')
    (directory/'transport.json').write_text(json.dumps(result),encoding='utf-8')
    print(json.dumps(result),flush=True)
    return int(rc!=0)


if __name__=='__main__':
    raise SystemExit(main(sys.argv[1],sys.argv[2]))
