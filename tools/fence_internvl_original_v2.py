"""Permanently fence only the original unresolved InternVL v2 request before side effects."""

import argparse
import ast
import json
from pathlib import Path
import subprocess
from types import ModuleType
from uuid import uuid4

from run_hygon_bound_payload import digest, document, send


G = Path(__file__).resolve().parent.parent
REQUEST = "1cf3cd6cc3df4e06bb8bcc963d57aef8"
ORIGINAL = "39e05f9f8e66485fbeaf08d5a61d6f7e"
ORIGINAL_PUBLICATION = "5444003c51dc386b4fd417bf138c6707b7dc86e1"
LAUNCH = "code/bw1000/training-internvl-usgs72-first2-current-hip-v2/launch.py"

HOST = r'''
import fcntl,json,os,re,stat,subprocess,sys,time
from hashlib import sha256
from pathlib import Path

packet=json.loads(sys.stdin.buffer.read(4096))
request,old=packet['request'],packet['original']
assert request=='1cf3cd6cc3df4e06bb8bcc963d57aef8' and old=='39e05f9f8e66485fbeaf08d5a61d6f7e'
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
parent,node,standby='1103606','f13r1n14','1103606.2520'
stage=root/('training-internvl-usgs72-v2-stage-'+old)
label='training-internvl-usgs72-native-first2-hip-v2'
target=root/label
controller=root/(label+'-controller')
names={label+'-'+role for role in ('pre','gpu','post')}|{'training-internvl-v2-port-'+old[:12]}

def query(*argv):
    return subprocess.check_output(argv,text=True,timeout=8).splitlines()

def fields(value):
    return dict(x.split('=',1) for x in value.split() if '=' in x)

def state():
    [job]=map(fields,query('scontrol','show','job','-o',parent))
    assert job['JobId']==parent and job['JobState']=='RUNNING' and job['UserId'].endswith(f'({os.getuid()})')
    assert 'gres/dcu=320' in job['AllocTRES'].split(',') and node in query('scontrol','show','hostnames',job['NodeList'])
    live=[r for l in query('scontrol','show','step',parent,'--oneliner')
          if (r:=fields(l)).get('StepId')==standby or r.get('Name') in names]
    assert len(live)==1 and [live[0][k] for k in ('StepId','Name','NodeList','State','TRES')]==[
        standby,'training80-dcu-standby-b71dc03b40ad-f13r1n14',node,'RUNNING','cpu=8,gres/dcu=8,mem=16G,node=1']
    columns='JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'
    all_rows=[l.split('|') for l in query('sacct','-nP','-j',parent,'-S','2026-11-24T13:00:00','-o',columns)]
    assert all(len(row)==9 for row in all_rows) and not any(row[1] in names for row in all_rows)
    [old_standby]=[row for row in all_rows if row[0]==standby]
    assert old_standby[2:4]==['RUNNING','0:0'] and old_standby[5:7]==[node,'cpu=8,gres/dcu=8,mem=16G,node=1']
    assert not os.path.lexists(target) and not os.path.lexists(controller)
    return old_standby

def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

os.umask(0o077)
with (root/'.training80-dcu-standby.lock').open('ab') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=state()
    # Exactly the original launcher's first effect and exclusive mkdir. If the
    # original gets here first this raises, and no existing object is altered.
    stage.mkdir(mode=0o700)
    sync_dir(root)
    note={'request':request,'fenced_request':old,'node':node,'parent':parent,'epoch':time.time(),
          'original_launch_publication':packet['original_publication'],'original_launch_sha256':packet['original_launch_sha256'],
          'record_type':'PERMANENT_PRE_EXECUTION_FENCE','not_a_launch':True,'original_standby':before}
    raw=(json.dumps(note,sort_keys=True)+'\n').encode()
    marker=stage/'ORIGINAL_REQUEST_FENCED.json'
    with marker.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    sync_dir(stage)
    after=state()
    assert stat.S_ISDIR(stage.lstat().st_mode) and list(stage.iterdir())==[marker] and marker.read_bytes()==raw
print(json.dumps({'request':request,'fenced_request':old,'state':'FENCED_AT_ORIGINAL_EXCLUSIVE_MKDIR',
    'observed_epoch':time.time(),'fence_file_sha256':sha256(raw).hexdigest(),'original_standby':after,
    'any_original_role_observed':False,'new_GPU_steps':0},sort_keys=True))
'''


def exact_source():
    raw = subprocess.check_output(["git", "-c", f"safe.directory={G.as_posix()}", "show", ORIGINAL_PUBLICATION + ":" + LAUNCH], cwd=G)
    tree = ast.parse(raw)
    raw_host = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "HOST" for t in n.targets))
    host_tree = ast.parse(raw_host)
    assert "stage=results/('training-internvl-usgs72-v2-stage-'+request)" in raw_host
    claim = raw_host.index("stage.mkdir(mode=0o700)")
    assert claim < raw_host.index("port=zero_card_port_probe()") < raw_host.index("signal=subprocess.run(['scancel'")
    head = next(node for node in host_tree.body if isinstance(node, ast.Expr) and node.lineno == raw_host[:claim].count("\n") + 1)
    assert isinstance(head.value, ast.Call) and head.value.func.attr == "mkdir"
    assert not any(term in raw_host for term in ("shutil.rmtree", "stage.unlink(", "stage.rmdir("))
    return digest(raw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--publication")
    args = p.parse_args()
    old_pin = exact_source()
    compile(HOST, "<original request fence>", "exec")
    if args.check:
        print(json.dumps({"request": REQUEST, "original_launch_sha256": old_pin, "remote_calls": 0}))
        return 0
    for f in (Path(__file__), G / "tools/run_hygon_bound_payload.py"):
        actual = subprocess.check_output(["git","-c",f"safe.directory={G.as_posix()}","show",
                     args.publication + ":" + f.resolve().relative_to(G).as_posix()],cwd=G)
        if actual != f.read_bytes().replace(b"\r\n",b"\n"):
            raise ValueError("publication and local source differ")
    local = G / "local-only"
    path = local / "run-minicpm-sdpa-target-asset-preflight-original.py"
    raw = path.read_bytes()
    if digest(raw) != "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd":
        raise ValueError("strict transport differs")
    transport = ModuleType("existing_ssh")
    exec(compile(raw,str(path),"exec"),vars(transport))
    result = send(transport.ssh(G.parent), request=REQUEST, program=HOST,
                  data=document({"request":REQUEST,"original":ORIGINAL,"original_publication":ORIGINAL_PUBLICATION,"original_launch_sha256":old_pin}),
                  send_token=local / f"internvl-original-fence-{REQUEST}-send.json", outcome_seconds=40)
    out = local / f"internvl-original-fence-{REQUEST}-{uuid4().hex}.json"
    data = document(result)
    with out.open("xb") as f:
        f.write(data)
    print(json.dumps({"file":out.name,"sha256": digest(data), **result},sort_keys=True))
    return int(result["remote_outcome"] != "RESPONSE_OBSERVED")


if __name__ == "__main__":
    raise SystemExit(main())
