"""Stage and start the published eight-DCU InternVL paired blind evaluation once."""

import argparse
from base64 import b64encode
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType
from uuid import uuid4

from launch_internvl_gqa64_first2_once import check_local_holds
from run_hygon_bound_payload import digest, document, send


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "code/bw1000/training-domain-pilot-v1"
sys.path.insert(0, str(SOURCE))
import internvl_gqa_val96_run as scope


MEMBERS = ["internvl_gqa_val96.py", "internvl_gqa_val96_native.py", "internvl_gqa_val96_run.py",
           "internvl_gqa_miopen_home.py"]
CLAIM = REPO / "coordination/tasks/training-internvl-gqa-val96-paired-hip-v3.json"
REMOTE = r'''
from base64 import b64decode
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

def digest(raw):
    return sha256(raw).hexdigest()

def saved(path, value):
    raw=(json.dumps(value,sort_keys=True)+'\n').encode()
    with path.open('xb') as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    return digest(raw)

def main():
    packet=json.load(sys.stdin)
    assert re.fullmatch('[0-9a-f]{40}',packet['publication'])
    claim=json.loads(b64decode(packet['claim'],validate=True))
    assert claim['request']==REQUEST and claim['experiment']==LABEL and claim['root_GPU_GO'] is True
    assert claim['formal_domain_delivery'] is False
    assert claim['resources']=={'CPU':48,'memory_MiB':224000,'DCUs':8}
    raw_handoff=b64decode(packet['handoff'],validate=True)
    assert 0<len(raw_handoff)<=8192 and digest(raw_handoff)==claim['private_deployment_SHA256']
    handoff=json.loads(raw_handoff)
    handles=('prior_gpu_step','prior_clean_step','standby_step')
    assert set(handoff)=={*handles,'standby_request','standby_expected_sha','standby_worker_sha','prior_outcome_sha','prior_post_sha'}
    assert all(re.fullmatch(r'\d+\.\d+',handoff[name]) for name in handles)
    assert re.fullmatch(r'[0-9a-f]{32}',handoff['standby_request'])
    assert all(re.fullmatch(r'[0-9a-f]{64}',value) for k,value in handoff.items() if k.endswith('_sha'))
    files={key:b64decode(value,validate=True) for key,value in packet['files'].items()}
    assert set(files)==set(MEMBERS) and all(digest(raw)==claim['source_SHA256'][name] for name,raw in files.items())
    work=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/LABEL
    assert not work.exists()
    os.environ.update(TZ='UTC',LC_ALL='C',PYTHONDONTWRITEBYTECODE='1')
    os.umask(0o077)
    work.mkdir(mode=0o700)
    source=work/'source'
    source.mkdir(mode=0o700)
    for name,raw in files.items():
        compile(raw,name,'exec')
        with (source/name).open('xb') as out:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
    epoch=time.time()
    manifest={'request':REQUEST,'files':{name:{'bytes':len(raw),'SHA256':digest(raw)} for name,raw in files.items()}}
    pin=saved(source/'SOURCE.json',manifest)
    saved(work/'original-claim.json',claim)
    with (work/'handoff.json').open('xb') as out:
        out.write(raw_handoff)
        out.flush()
        os.fsync(out.fileno())
    saved(work/'submission-intent.json',{'request':REQUEST,'publication':packet['publication'],
          'epoch':epoch,'source_SHA256':pin,'original_standby':handoff['standby_step']})
    argv=[sys.executable,'-B',str(source/'internvl_gqa_val96_run.py'),'controller',str(epoch)]
    with (work/'controller.log').open('xb') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    proc=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()
    value={'request':REQUEST,'rc':0,'controller_PID':child.pid,'controller_PGID':int(proc[2]),
           'controller_startticks':int(proc[19]),'epoch':epoch,'source_SHA256':pin,'gpu_running_observed':False}
    receipt=saved(work/'controller-submission.json',value)
    print(json.dumps({**value,'receipt_SHA256':receipt},sort_keys=True),flush=True)

try:
    main()
except (ValueError,AssertionError,KeyError,OSError,subprocess.SubprocessError) as error:
    print(json.dumps({'request':REQUEST,'rc':1,'class':type(error).__name__,
                      'error_SHA256':digest(str(error).encode())},sort_keys=True),flush=True)
    sys.exit(1)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    switch = parser.add_mutually_exclusive_group(required=True)
    switch.add_argument("--check", action="store_true")
    switch.add_argument("--publication")
    args = parser.parse_args()
    files = {name: (SOURCE / name).read_bytes().replace(b"\r\n", b"\n") for name in MEMBERS}
    for name, raw in files.items():
        compile(raw, name, "exec")
    source = f"REQUEST={scope.REQUEST!r}\nLABEL={scope.LABEL!r}\nMEMBERS={MEMBERS!r}\n" + REMOTE
    compile(source, "original host admission", "exec")
    if args.check:
        print(json.dumps({"request": scope.REQUEST, "host_SHA256": digest(source.encode()),
                          "members": {name: digest(raw) for name, raw in files.items()}}))
        return 0
    if not re.fullmatch("[0-9a-f]{40}", args.publication):
        raise ValueError("Exact published revision required")
    for path in (Path(__file__).resolve(), CLAIM, REPO / "tools/run_hygon_bound_payload.py",
                 *(SOURCE / name for name in MEMBERS)):
        name = path.relative_to(REPO).as_posix()
        found = subprocess.check_output(["git", "-c", f"safe.directory={REPO.as_posix()}",
                                         "show", args.publication + ":" + name], cwd=REPO)
        if found != path.read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError("Requested original publication differs from this checkout")
    claim = CLAIM.read_bytes()
    claim_data = json.loads(claim)
    if claim_data["host_admission_SHA256"] != digest(source.encode()):
        raise ValueError("Published host admission differs")
    check_local_holds()
    local = REPO / "local-only"
    handoff = (local / (scope.LABEL + "-private-handoff.json")).read_bytes()
    if digest(handoff) != claim_data["private_deployment_SHA256"]:
        raise ValueError("Private placement differs from its published digest")
    helper = local / "run-minicpm-sdpa-target-asset-preflight-original.py"
    raw = helper.read_bytes()
    if digest(raw) != "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd":
        raise ValueError("Existing transport helper differs")
    transport = ModuleType("existing_ssh")
    exec(compile(raw, str(helper), "exec"), vars(transport))
    payload = {"files": {key: b64encode(raw).decode() for key, raw in files.items()},
               "claim": b64encode(claim).decode(), "handoff": b64encode(handoff).decode(),
               "publication": args.publication}
    result = send(transport.ssh(REPO.parent), request=scope.REQUEST, program=source, data=document(payload),
                  send_token=local / f"internvl-gqa96-hip-{scope.REQUEST}-send.json", outcome_seconds=40)
    receipt = local / f"internvl-gqa96-hip-{scope.REQUEST}-{uuid4().hex}.json"
    with receipt.open("xb") as out:
        out.write(document(result))
    view = result.get("response") or {}
    print(json.dumps({"request": scope.REQUEST, "original_receipt": receipt.name,
                      "remote_outcome": result["remote_outcome"], "result": view}, sort_keys=True))
    return int(result["remote_outcome"] != "RESPONSE_OBSERVED" or view.get("rc") != 0)


if __name__ == "__main__":
    raise SystemExit(main())
