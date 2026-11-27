"""Run the adopted GQA first-two controller once on its clear, owned node."""
import argparse
from base64 import b64encode
import gzip
import json
from pathlib import Path
import re
import subprocess
from types import ModuleType

import render_internvl_gqa64_first2 as source
from run_hygon_bound_payload import digest, document, send


HOST = r'''
from base64 import b64decode
import fcntl, gzip, json, os, re, subprocess, sys, time
from hashlib import sha256
from pathlib import Path

R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
REQUEST='962ea01bdb954b5592d38df1f8e106d9'
LABEL='training-internvl-gqa64-first2-hip-v1'
old_request='39e05f9f8e66485fbeaf08d5a61d6f7e'
fence_request='6063ab4872014f7dbbee8db7bd97abaa'
os.environ.update(TZ='UTC',LC_ALL='C')
assert os.getuid()==154459
sha=lambda data:sha256(data).hexdigest()
raw=sys.stdin.buffer.read(270001)
assert 0<len(raw)<=270000
with gzip.GzipFile(fileobj=__import__('io').BytesIO(b64decode(raw,validate=True))) as compressed:
    data=compressed.read(700001)
assert 0<len(data)<=700000
packet=json.loads(data)
files={name:b64decode(value,validate=True) for name,value in packet['files'].items()}
claim=json.loads(files['CLAIM.json'])
manifest=json.loads(files['SOURCE_FILES.json'])
assert len(files)==16 and claim['request']==manifest['request']==REQUEST and claim['label']==LABEL
assert claim['planned_updates']==2 and claim['planned_nonzero_updates']==1 and claim['planned_microbatches']==64
assert claim['root_GO'] is claim['GPU_GO'] is True and claim['formal17_qualified'] is claim['replay_allowed'] is False
assert re.fullmatch('[a-f0-9]{40}',packet['publication']) and set(files)-{'SOURCE_FILES.json'}==set(manifest['files'])
assert all(Path(name).name==name and {'bytes':len(files[name]),'SHA256':sha(files[name])}==record
           for name,record in manifest['files'].items())
assert claim['placement']==dict(cpus=64,dcu=8,mem_mib=224000,node='f13r1n14',parent='1103606',port=29967)

stage=R/(LABEL+'-source-'+REQUEST)
os.umask(0o077)
with (R/'.training80-dcu-standby.lock').open('ab') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=R/('training-internvl-usgs72-v2-stage-'+old_request)
    path=old/'ORIGINAL_REQUEST_FENCED.json'
    raw=path.read_bytes()
    fence=json.loads(raw)
    assert len(raw)<8192 and sha(raw)==packet['fence_SHA256'] and set(old.iterdir())=={path}
    assert (fence['request'],fence['fenced_request'],fence['record_type'],fence['not_a_launch'])==(
        fence_request,old_request,'PERMANENT_PRE_EXECUTION_FENCE',True)
    assert fence['original_standby'][0]=='1103606.2520' and fence['original_standby'][2:4]==['COMPLETED','0:0']
    assert not (R/LABEL).exists() and not (R/(LABEL+'-controller')).exists()
    stage.mkdir(mode=0o700)
    for name,raw in files.items():
        with (stage/name).open('xb') as output:
            output.write(raw)
    sys.path.insert(0,str(stage))
    from control_once import accounting,placement
    from native import binding
    from node import actual_model_plan
    placement()
    assert all(accounting(role) is None for role in ('pre','gpu','post'))
    asset,plan,view=actual_model_plan(claim['exports_SHA256'])
    asset.verify_originals(plan)
    exact=binding()
    assert len(exact['domain_source_rows'])==64 and len(exact['trainable_manifest'])==504 and len(view)==20
    with (stage/'controller-start-intent.json').open('x') as out:
        json.dump({'request':REQUEST,'fence_SHA256':packet['fence_SHA256'],'epoch':time.time()},out,sort_keys=True)
        out.flush();os.fsync(out.fileno())
    command=[sys.executable,'-B',str(stage/'control_once.py'),'--publication',packet['publication'],
             '--claim-sha256',sha(files['CLAIM.json']),'--manifest-sha256',sha(files['SOURCE_FILES.json']),
             '--exports-sha256',claim['exports_SHA256']]
    child=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,timeout=45)
    for name,content in [('controller.stdout',child.stdout),('controller.stderr',child.stderr)]:
        with (stage/name).open('xb') as out:out.write(content)
    result={'request':REQUEST,'rc':child.returncode,'stderr_SHA256':sha(child.stderr),'stdout_SHA256':sha(child.stdout)}
    if child.returncode==0 and 0<len(child.stdout)<8192:
        result['controller']=json.loads(child.stdout)
        assert result['controller']['request']==REQUEST and result['controller']['state']=='CONTROLLER_STARTED'
    print(json.dumps(result,sort_keys=True),flush=True)
'''


def check_local_holds():
    import report_training80_backlog as backlog

    repo = source.REPO
    items = json.loads((repo / "coordination/training/training80-backlog-v1.json").read_bytes())["items"]
    holds = backlog.read_queue_holds(repo / "local-only/tuning/queue.sqlite3")
    known = set(backlog.allocation(backlog.TRAINING80_LEDGER.read_bytes())[2])
    known.update(node for item in items for node in item["placement"])
    active, _, _ = backlog.read_original_holds(repo, items, known, holds)
    if any("f13r1n14" in item["nodes"] for item in [*holds, *active.values()]):
        raise ValueError("Existing local request holds the GQA node")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--publication")
    parser.add_argument("--fence-sha256")
    args = parser.parse_args()
    files = source.render()
    compile(HOST, "<GQA training controller>", "exec")
    if args.check:
        print(json.dumps({"request": source.REQUEST, "manifest_sha256": digest(files["SOURCE_FILES.json"]),
                          "controller_sha256": digest(HOST.encode()), "remote_calls": 0}))
        return 0
    if not (re.fullmatch("[0-9a-f]{40}", args.publication)
            and re.fullmatch("[0-9a-f]{64}", args.fence_sha256 or "")):
        raise ValueError("Exact public source and actual terminal-fence digest required")
    repo = source.REPO
    registered = (Path(__file__), Path(source.__file__), source.ROSTER, repo / "tools/run_hygon_bound_payload.py")
    for path in registered:
        relative = path.resolve().relative_to(repo).as_posix()
        published = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "show", args.publication + ":" + relative], cwd=repo)
        if published != path.read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError("Published GQA source and checkout differ")
    check_local_holds()
    local = repo / "local-only"
    helper = local / "run-minicpm-sdpa-target-asset-preflight-original.py"
    raw = helper.read_bytes()
    if digest(raw) != "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd":
        raise ValueError("Existing authenticated helper differs")
    transport = ModuleType("existing_ssh")
    exec(compile(raw, str(helper), "exec"), vars(transport))
    data = {"publication": args.publication, "fence_SHA256": args.fence_sha256,
            "files": {name: b64encode(content).decode("ascii") for name, content in files.items()}}
    result = send(transport.ssh(repo.parent), request=source.REQUEST, program=HOST,
                  data=b64encode(gzip.compress(document(data), mtime=0)),
                  send_token=local / f"internvl-gqa64-first2-{source.REQUEST}-send.json", outcome_seconds=100)
    path = local / f"internvl-gqa64-first2-{source.REQUEST}-result.json"
    with path.open("xb") as stream:
        stream.write(document(result))
    print(json.dumps({"receipt": path.name, "receipt_sha256": digest(path.read_bytes()), **result}, sort_keys=True))
    return int(result["remote_outcome"]!="RESPONSE_OBSERVED" or result["response"].get("rc")!=0)


if __name__ == "__main__":
    raise SystemExit(main())
