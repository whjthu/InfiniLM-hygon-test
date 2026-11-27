"""Bind the four recovered original LLaVA shards into the qualified model layout."""

import argparse
from base64 import b64encode
import json
from pathlib import Path
import re
import subprocess
from types import ModuleType

from run_hygon_bound_payload import digest, document, send


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local-only"
ROOT = "/public/home/aclsylqidf/hygon-inference-refactor-20261014"
ASSETS = REPO / "code/bw1000/training-llava-original-model18-v1"
SOURCE_PIN = "917c1d1fa9723121d30d208019b2fa662b2df54723a013d0a46deede056e9b16"
MANIFEST_PIN = "afb4c144925f4ef7d4342bc468ea90660b8dc2819ca100b1f157f3a8d32440dd"
TRANSPORT_PIN = "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd"
REQUESTS = {
    "preflight": "1ab23c8715e4403898fe76c09e445aed",
    "assemble": "66ce58d0681b405f9a6c6d1c6993c180",
}
SHARD_ROOTS = (
    "incoming/llava-t04-shard01-split-suffix-v1-1a22d5d7c5294a9faa2e0d54c6ff8669",
    "results/training-T04-public-official-shard02-a100-relay-v1",
    "incoming/llava-t04-shard03-split-suffix-v1-a4b1fda86f284fb89705d9b57d8c3377",
    "results/training-T04-public-official-shard04-verified-suffix-v1",
)
REMOTE = r"""
from base64 import b64decode
from hashlib import sha256
from pathlib import Path
import json,os,stat,subprocess,sys

root=Path(ROOT)
def pin(raw):return sha256(raw).hexdigest()
def doc(value):return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
def put(path,raw):
 with path.open('xb') as f:
  f.write(raw);f.flush();os.fsync(f.fileno())
def out(data):print(json.dumps(dict(request=REQUEST,**data),sort_keys=True),flush=True)
def file_stat(p):
 assert p==p.resolve(strict=True) and stat.S_ISREG(p.lstat().st_mode)
 return p.stat()
def main():
 packet=json.load(sys.stdin)
 assert set(packet)=={'assemble.py','MODEL_FILES.json','PLAN.json'}
 data={name:b64decode(raw,validate=True) for name,raw in packet.items()}
 assert all(pin(data[name])==PINS[name] for name in data)
 plan=json.loads(data['PLAN.json'])
 metadata=root/'results/training-vlm-T03-T04-metadata-v2/llava/model'
 output=root/'results/training-llava-original-model18-v1'
 manifest=json.loads(data['MODEL_FILES.json'])['files']
 st=(root/'results').stat()
 assert not os.path.lexists(output) and len(manifest)==18
 studies={}
 for name,record in sorted(manifest.items()):
  path=Path(plan['weights'][name]) if name in plan['weights'] else metadata/name
  if not path.is_file():
   studies[name]='MISSING'
   continue
  s=file_stat(path)
  if s.st_size!=record['bytes']:
   studies[name]='SIZE_MISMATCH'
  elif s.st_dev!=st.st_dev:
   studies[name]='HARDLINK_CROSS_DEVICE'
  elif name not in plan['weights'] and pin(path.read_bytes())!=record['SHA256']:
   studies[name]='METADATA_HASH_MISMATCH'
  else:
   studies[name]='EXTENT_AND_DEVICE_PASS' if name in plan['weights'] else 'FULL_HASH_PASS'
 failed=[name for name,state in studies.items() if state not in {'EXTENT_AND_DEVICE_PASS','FULL_HASH_PASS'}]
 if MODE=='preflight' or failed:
  out(dict(rc=0 if not failed else 1,state='DEPLOYED_LAYOUT_PREFLIGHT',files=studies,
           failed=failed,model_loaded=False,GPU_steps=0))
  return
 for index in (1,3):
  shard=f'model-{index:05d}-of-00004.safetensors'
  receipt=Path(plan['weights'][shard]).parent/'assembly-result.json'
  claimed=json.loads(receipt.read_bytes())
  assert claimed['state']=='PUBLISHED' and (claimed['bytes'],claimed['sha256'])==(
     manifest[shard]['bytes'],manifest[shard]['SHA256'])
 work=root/'incoming'/('llava-original-model18-source-'+REQUEST)
 os.umask(0o077)
 work.mkdir(mode=0o700)
 for name,raw in data.items():put(work/name,raw)
 argv=[sys.executable,'-I','-S','-B',str(work/'assemble.py'),'--plan',str(work/'PLAN.json'),
       '--plan-sha256',PINS['PLAN.json'],'--manifest-sha256',PINS['MODEL_FILES.json'],'--seconds','370']
 put(work/'intent.json',doc(dict(request=REQUEST,pins=PINS,publication=PUBLICATION,argv=argv,actual_GPU_steps=0)))
 p=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,timeout=385)
 put(work/'stdout.private',p.stdout);put(work/'stderr.private',p.stderr)
 if p.returncode:
  tail=p.stderr.decode('utf-8','replace').splitlines()[-1:]
  out(dict(rc=1,state='ORIGINAL_ASSEMBLER_TERMINAL',assembler_rc=p.returncode,
           error_class=tail[0].split(':',1)[0] if tail else None,stderr_SHA256=pin(p.stderr)))
  return
 result=json.loads((output/'receipt.json').read_bytes())
 assert result['verified_complete_shards']==4 and result['verified_original_files']==18
 assert result['copied_weight_bytes']==0 and result['hardlinked_model_directory'] is True
 assert result['plan_SHA256']==PINS['PLAN.json'] and result['manifest_SHA256']==PINS['MODEL_FILES.json']
 selected={k:result[k] for k in ('verified_original_files','verified_complete_shards','hash_bytes',
       'copied_weight_bytes','hardlinked_model_directory','actual_GPU_steps','model_loaded',
       'initial_trainables_qualified','runtime_loader_qualified','wall_seconds','plan_SHA256','manifest_SHA256')}
 out(dict(rc=0,state='18_FILE_MODEL_DIRECTORY_VERIFIED',original_receipt_SHA256=pin((output/'receipt.json').read_bytes()),
          output_directory=str(output/'model'),publication=PUBLICATION,**selected))
try:
 main()
except (ValueError,OSError,KeyError,AssertionError,subprocess.SubprocessError) as exc:
 out(dict(rc=1,error_type=type(exc).__name__,error_SHA256=pin(str(exc).encode())))
"""


def bindings():
    original = (ASSETS / "assemble.py").read_bytes()
    manifest = (ASSETS / "MODEL_FILES.json").read_bytes()
    if (digest(original), digest(manifest)) != (SOURCE_PIN, MANIFEST_PIN):
        raise ValueError("Qualified model assembly source differs")
    old = "sources = input_mapping(plan, files, PROJECT, METADATA)"
    if original.count(old.encode()) != 1:
        raise ValueError("Qualified path contract differs")
    # Keep the existing verifier and hardlink logic. The two recovered shards live under incoming.
    source = original.replace(old.encode(), b"sources = input_mapping(plan, files, PROJECT.parent, METADATA)", 1)
    compile(source, "original assembly with the recovered source root", "exec")
    shards = {}
    for i, path in enumerate(SHARD_ROOTS, 1):
        name = f"model-{i:05d}-of-00004.safetensors"
        shards[name] = f"{ROOT}/{path}/{name}"
    return {"assemble.py": source, "MODEL_FILES.json": manifest,
            "PLAN.json": document(dict(case="T04.sft", weights=shards))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", *REQUESTS))
    parser.add_argument("--publication")
    args = parser.parse_args()
    files = bindings()
    pins = {key: digest(raw) for key, raw in files.items()}
    if args.mode == "check":
        print(json.dumps(dict(pins=pins, original_assembler_SHA256=SOURCE_PIN), sort_keys=True))
        return 0
    if args.mode == "assemble":
        if not re.fullmatch("[0-9a-f]{40}", args.publication or ""):
            raise ValueError("Exact published assembly source required")
        path = Path(__file__).resolve()
        member = path.relative_to(REPO).as_posix()
        published = subprocess.check_output(
            ["git", "-c", f"safe.directory={REPO.as_posix()}", "show", f"{args.publication}:{member}"], cwd=REPO
        )
        if published != path.read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError("Published assembly wrapper differs")
    request = REQUESTS[args.mode]
    helper = LOCAL / "run-minicpm-sdpa-target-asset-preflight-original.py"
    raw = helper.read_bytes()
    if digest(raw) != TRANSPORT_PIN:
        raise ValueError("Existing authenticated transport differs")
    client = ModuleType("existing_transport")
    client.__file__ = str(helper)
    exec(compile(raw, str(helper), "exec"), vars(client))
    settings = dict(ROOT=ROOT, REQUEST=request, PINS=pins, MODE=args.mode, PUBLICATION=args.publication)
    program = "\n".join(f"{k}={v!r}" for k, v in settings.items()) + "\n" + REMOTE
    result = send(client.ssh(REPO.parent), request=request, program=program,
                  data=document({k: b64encode(raw).decode() for k, raw in files.items()}),
                  send_token=LOCAL / f"llava-model18-{request}-send.json", outcome_seconds=420)
    path = LOCAL / f"llava-model18-{request}.json"
    with path.open("xb") as f:
        f.write(document(result))
    print(json.dumps(dict(receipt=path.name, outcome=result["remote_outcome"],
                          response=result["response"]), sort_keys=True))
    return int(result["remote_outcome"] != "RESPONSE_OBSERVED" or result["response"]["rc"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
