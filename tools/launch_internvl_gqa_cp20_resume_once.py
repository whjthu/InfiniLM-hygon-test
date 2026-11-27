"""Start one published native CP20 continuation with its private deployment binding."""

import argparse
from base64 import b64encode
import gzip
import json
from pathlib import Path
import re
import subprocess
from types import ModuleType
from uuid import uuid4

import render_internvl_gqa_cp20_resume as render
from launch_internvl_gqa64_first2_once import check_local_holds
from run_hygon_bound_payload import digest, document, send


REPO = Path(__file__).resolve().parents[1]
CLAIM = REPO / "coordination/tasks/training-internvl-gqa64-cp20-resume-hip-v2.json"
REMOTE = r'''
from base64 import b64decode
import gzip,json,os,re,subprocess,sys,time
from hashlib import sha256
from io import BytesIO
from pathlib import Path

def parts(raw):
    return {"bytes":len(raw),"SHA256":sha256(raw).hexdigest()}

def put(path,raw):
    with path.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())

packet=b64decode(sys.stdin.buffer.read(300001),validate=True)
with gzip.GzipFile(fileobj=BytesIO(packet)) as f:body=f.read(900001)
assert 0<len(body)<=900000
item=json.loads(body)
assert re.fullmatch(r'[0-9a-f]{40}',item['publication'])
files={key:b64decode(raw,validate=True) for key,raw in item['files'].items()}
manifest=json.loads(files['SOURCE_FILES.json'])
assert manifest['request']==REQUEST and set(files)-{'SOURCE_FILES.json'}==set(manifest['files'])
assert all(Path(name).name==name and parts(files[name])==stamp for name,stamp in manifest['files'].items())
claim=json.loads(files['CLAIM.json'])
assert claim['request']==REQUEST and claim['label']==LABEL and claim['planned_updates']==5
assert (claim['resume_step'],claim['max_steps'],claim['world_size'])==(20,25,8)
handoff=b64decode(item['handoff'],validate=True)
assert 0<len(handoff)<=8192 and parts(handoff)['SHA256']==claim['private_deployment_SHA256']
bindings=json.loads(handoff)
assert re.fullmatch(r'\d+\.\d+',bindings['standby_step']) and re.fullmatch(r'[0-9a-f]{32}',bindings['standby_request'])
results=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
target=results/(LABEL+'-dispatch')
os.umask(0o077)
os.environ.update(TZ='UTC',LC_ALL='C',PYTHONDONTWRITEBYTECODE='1')
target.mkdir(mode=0o700)
source=target/'source';source.mkdir()
for name,raw in files.items():put(source/name,raw)
put(target/'handoff.json',handoff)
receipt={'request':REQUEST,'publication':item['publication'],'epoch':time.time(),
         'manifest_SHA256':parts(files['SOURCE_FILES.json'])['SHA256'],'standby_step':bindings['standby_step']}
put(target/'dispatch-intent.json',(json.dumps(receipt,sort_keys=True)+'\n').encode())
with (target/'handoff.log').open('xb') as log:
    child=subprocess.Popen([sys.executable,'-B',str(source/'internvl_gqa_resume_handoff.py'),'dispatch'],
                           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
process=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()
out={**receipt,'rc':0,'dispatch_PID':child.pid,'dispatch_startticks':int(process[19])}
put(target/'dispatch-submission.json',(json.dumps(out,sort_keys=True)+'\n').encode())
print(json.dumps(out,sort_keys=True),flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--publication")
    args = parser.parse_args()
    files = render.render(CLAIM.read_bytes())
    program = f"REQUEST={render.REQUEST!r}\nLABEL={render.LABEL!r}\n" + REMOTE
    compile(program, "resume login sender", "exec")
    if args.check:
        print(json.dumps({"manifest_SHA256": digest(files["SOURCE_FILES.json"]),
                          "launcher_SHA256": digest(program.encode()), "members": len(files)}))
        return 0
    if not re.fullmatch(r"[0-9a-f]{40}", args.publication):
        raise ValueError("Exact published commit required")
    check_local_holds()
    for path in (Path(__file__).resolve(), Path(render.__file__), CLAIM,
                 REPO / "tools/run_hygon_bound_payload.py",
                 render.FAMILY / "internvl_gqa_resume_callback.py",
                 *(render.FAMILY / name for name in render.EXTRA)):
        name = path.relative_to(REPO).as_posix()
        current = path.read_bytes().replace(b"\r\n", b"\n")
        expected = subprocess.check_output(
            ["git", "-c", "safe.directory=" + REPO.as_posix(), "show", args.publication + ":" + name], cwd=REPO)
        if current != expected:
            raise ValueError("Published recovery source differs")
    local = REPO / "local-only"
    deployment = (local / (render.LABEL + "-private-handoff.json")).read_bytes()
    if digest(deployment) != json.loads(CLAIM.read_bytes())["private_deployment_SHA256"]:
        raise ValueError("Private handoff digest differs")
    helper = local / "run-minicpm-sdpa-target-asset-preflight-original.py"
    raw = helper.read_bytes()
    if digest(raw) != "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd":
        raise ValueError("Existing transport differs")
    mod = ModuleType("existing_ssh")
    exec(compile(raw, str(helper), "exec"), vars(mod))
    packet = document({"files": {name: b64encode(raw).decode() for name, raw in files.items()},
                       "handoff": b64encode(deployment).decode(), "publication": args.publication})
    response = send(mod.ssh(REPO.parent), request=render.REQUEST, program=program,
                    data=b64encode(gzip.compress(packet, mtime=0)),
                    send_token=local / f"internvl-gqa-cp20-hip-{render.REQUEST}-send.json", outcome_seconds=45)
    receipt = local / f"internvl-gqa-cp20-hip-{render.REQUEST}-{uuid4().hex}.json"
    with receipt.open("xb") as f:
        f.write(document(response))
    actual = response.get("response") or {}
    print(json.dumps({"receipt": receipt.name, "outcome": response["remote_outcome"],
                      "response": actual}, sort_keys=True))
    return int(response["remote_outcome"] != "RESPONSE_OBSERVED" or actual.get("rc") != 0)


if __name__ == "__main__":
    raise SystemExit(main())
