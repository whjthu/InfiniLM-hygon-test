"""Prepare the exact dormant successor or a root-supplied one-shot runtime capsule."""

import argparse
import ast
from base64 import b64decode, b64encode
import json
from pathlib import Path, PurePosixPath
import re
import runpy
import sys
import time
import zlib


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import admission as a


OLD_LABEL = "training-gemma-usgs-r120-two-updates-native-v1"
OLD_DISPATCH = "923cbc0697ed4534b990999fa34732d2"
BASE = HERE.parent / OLD_LABEL
ORIGINAL_OWNER = REPO / "local-only/gemma_r120_launch_after_release.py"
OWNER_PIN = "a29dd9ee9888b48b0fb99d3da4b6cc6b398bac4cebe65f2732584073518db2b4"
HISTORY = REPO / "results/bw1000/2026-11-27/training-gemma-r120-original4081-natural-terminal-independent-4167-post.json"
CONTAINER_ROOT = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014")


def read(path, digest, limit):
    a.require(path.is_file() and not path.is_symlink(), "ORIGINAL_REGULAR_FILE_REQUIRED")
    with path.open("rb") as stream:
        raw = stream.read(limit+1)
    a.require(0<len(raw)<=limit and a.digest(raw)==digest, "EXACT_SOURCE_OR_METADATA_SHA256")
    return raw


def place(folder, members):
    folder.mkdir(mode=0o700)
    for key, value in members.items():
        with (folder / key).open("xb") as stream:
            stream.write(value)


def stable_members():
    manifest = json.loads(read(BASE/"SOURCE_FILES.json",a.BASE_MANIFEST,16384))
    blobs = {}
    replacements = ((OLD_LABEL,a.LABEL),(a.OLD_SCOPE,a.SCOPE),(OLD_DISPATCH,a.DISPATCH),
                    ("32120",str(a.PORT)),("/tmp/g120","/tmp/gr120a"))
    changed = []
    for name, pin in manifest["files"].items():
        a.require(re.fullmatch(r"[A-Za-z0-9_.-]+", name) and not name.startswith("."), "SOURCE_NAME")
        raw = read(BASE/name,pin["SHA256"],65536)
        a.require(len(raw)==pin["bytes"],"SOURCE_EXTENT")
        result = raw
        for before, after in replacements:
            result = result.replace(before.encode(),after.encode())
        if result!=raw:
            changed.append(name)
        if name.endswith(".py"):
            compile(result,name,"exec")
        blobs[name]=result
    a.require(len(blobs)==32 and set(changed)=={"CLAIM.json","contract.py","controller.py","short_mount.py","verify_original.py"},
              "ONLY_EXACT_SUCCESSOR_IDENTITY_CONSTANTS_CHANGED")
    claim=json.loads(blobs["CLAIM.json"])
    a.require(claim["root_GO"] is claim["GPU_GO"] is claim["training_GO"] is False,
              "STABLE_CLAIM_MUST_BE_DISABLED")
    return blobs,manifest["schema"]


def source_manifest(blobs,schema):
    return a.encode(dict(schema=schema, files={name:a.pin(raw) for name,raw in blobs.items()}))


def verified_stable():
    blobs, schema = stable_members()
    manifest = source_manifest(blobs, schema)
    a.require((HERE / "SUCCESSOR_SOURCE_FILES.json").read_bytes() == manifest, "EXACT_FROZEN_SUCCESSOR_MANIFEST")
    return blobs, schema, manifest


def operational_blobs(stable,schema,approval,release):
    check=json.loads(approval)
    claim=json.loads(stable["CLAIM.json"])
    for cap, required in (("cpu_3481_independent_SHA256","exact_cpu_3481_receipt_SHA256"),
        ("sampler_source_manifest_SHA256","exact_swift_source_manifest_SHA256"),
        ("prior_diagnostic_terminal_SHA256","prior_diagnostic_terminal_SHA256"),
        ("zero_gres_source_SHA256","cleanup_CPU_source_SHA256")):
        a.require(check[cap]==claim[required],"FROZEN_R120_AUTHORITY_COHORT")
    a.require(check["cpu_original_independent_terminal_SHA256"]=="74bac702f6e94de63d3db93ad8744fffdb45b797881510792587ef76e415de02"
              and re.fullmatch("[0-9a-f]{32}",check["zero_gres_request"])
              and check["zero_gres_request"]!=a.SCOPE, "FROZEN_R120_CPU_PREREQUISITES")
    claim.update(root_GO=True,GPU_GO=True,training_GO=True,placement_authority_SHA256=a.digest(approval),
                 latest_holder_observation_SHA256=a.digest(release),state="ROOT_PINNED_REQUIRES_LIVE_ADMISSION")
    effective=stable | {"CLAIM.json":a.encode(claim)}
    return effective,source_manifest(effective,schema)


def check_receiver(body, effective, manifest, context):
    assignments={n.targets[0].id:n.value for n in ast.parse(body).body if isinstance(n,ast.Assign)
                 and isinstance(n.targets[0],ast.Name)}
    payload=ast.literal_eval(assignments["payload"])
    decoded={n:b64decode(raw,validate=True) for n,raw in payload.items()}
    a.require(decoded==effective | {"SOURCE_FILES.json":manifest}
              and ast.literal_eval(assignments["remote_root"])==str(CONTAINER_ROOT/"results"/a.LABEL)
              and context["source_manifest"]==a.pin(manifest),"EXACT_REAL_R120_RECEIVER_CLOSURE")


def host_program(auth, release, history, original_release, natural, stable_pin, context, body_sha):
    read(ORIGINAL_OWNER,OWNER_PIN,32768)
    legacy=runpy.run_path(str(ORIGINAL_OWNER))["program"](a.POST,a.digest(release))
    for before,after in ((OLD_LABEL,a.LABEL),(a.OLD_SCOPE,a.SCOPE),(OLD_DISPATCH,a.DISPATCH)):
        a.require(legacy.count(before)==1,"EXACT_ORIGINAL_OWNER_SCOPE_SPLICE")
        legacy=legacy.replace(before,after)
    source=(HERE/"admission.py").read_bytes()
    anchor="assert not os.path.lexists(root/'results'/label)"
    a.require(legacy.count(anchor)==1,"EXACT_PRE_MUTATION_OWNER_ANCHOR")
    insert=(
        "gate_namespace = {'__name__':'runtime_admission'}\n"
        f"exec(compile({source!r}, '<pinned runtime admission>', 'exec'), gate_namespace)\n"
        f"assert context == {context!r} and a['body_SHA256'] == {body_sha!r}\n"
        "def evidence(key, sha, limit):\n"
        " value=b64decode(a[key],validate=True)\n"
        " assert 0<len(value)<=limit and sha256(value).hexdigest()==sha\n"
        " return value\n"
        f"history=evidence('history',{a.digest(history)!r},8192)\n"
        f"original=evidence('original_release',{a.digest(original_release)!r},16384)\n"
        f"natural=evidence('natural',{a.digest(natural)!r},16384)\n"
        f"approval=evidence('raw_approval',{a.digest(auth)!r},16384)\n"
        f"rebuilt=gate_namespace['authorize'](history,original,natural,approval,context['publication'],{stable_pin!r},time.time())\n"
        "assert rebuilt == (auth,proof)\n")
    out=legacy.replace(anchor,insert+anchor,1)
    compile(out,"<bound Gemma R120 successor>", "exec")
    return out


def render(output,stable,schema):
    mf=source_manifest(stable,schema)
    place(output,stable | {"SOURCE_FILES.json":mf})
    return dict(state="DISABLED_SOURCE_ONLY",label=a.LABEL,scope=a.SCOPE,request=a.DISPATCH,
                source_members=len(stable),manifest=a.pin(mf))


def prepare(output,publication,published_sha,approval,old_release,natural,now):
    stable,schema,stable_mf=verified_stable()
    stable_pin=a.pin(stable_mf)
    a.require(published_sha==stable_pin["SHA256"],"EXACT_PUBLISHED_STABLE_SOURCE_MANIFEST")
    history=read(HISTORY,a.HISTORY,8192)
    auth,release=a.authorize(history,old_release,natural,approval,publication,stable_pin,now)
    result,effective_mf=operational_blobs(stable,schema,auth,release)
    output.mkdir(mode=0o700)
    package=output/"effective-source"
    place(package,result | {"SOURCE_FILES.json":effective_mf})
    sys.path.insert(0,str(package))
    try:
        launch=runpy.run_path(str(package/"launch.py"))
    finally:
        sys.path.pop(0)
    make=launch["make_body"]
    make.__globals__["ROOT"]=CONTAINER_ROOT/"results"/a.LABEL
    mf=a.pin(effective_mf)
    body=make(publication,now,mf)
    context=dict(request_id=a.DISPATCH,training_request_id=a.SCOPE,publication=publication,epoch=now,
                 deadline=now+a.WHOLE,source_manifest=mf,remote_command=launch["HOST_COMMAND"])
    check_receiver(body,result,effective_mf,context)
    program=host_program(approval,release,history,old_release,natural,stable_pin,context,a.digest(body))
    wire=a.encode(dict(request=a.DISPATCH,context=context,body=b64encode(zlib.compress(body,6)).decode(),
                       body_SHA256=a.digest(body), authority=b64encode(auth).decode(),authority_SHA256=a.digest(auth),
                       release=b64encode(release).decode(),history=b64encode(history).decode(),
                       original_release=b64encode(old_release).decode(),raw_approval=b64encode(approval).decode(),
                       natural=b64encode(natural).decode()))
    count=len(b64encode(a.encode(dict(program=program,data=b64encode(wire).decode()))))
    a.require(len(body)<=280000 and count<=560000,"EXISTING_CARRIER_CAP")
    artifacts={"HOST_PROGRAM.py":program.encode(),"WIRE_DATA.json":wire}
    for name,raw in artifacts.items():
        with (output/name).open("xb") as stream:
            stream.write(raw)
    return dict(state="LOCAL_CAPSULE_NO_SEND",scope=a.SCOPE,request=a.DISPATCH,publication=publication,
                stable_source_manifest=stable_pin,effective_source_manifest=mf,wire_base64_bytes=count,
                files={n:a.pin(raw) for n,raw in artifacts.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    tools=p.add_subparsers(dest="mode",required=True)
    tools.add_parser("freeze").add_argument("--output",type=Path,required=True)
    live=tools.add_parser("prepare")
    live.add_argument("--output",type=Path,required=True)
    live.add_argument("--publication",required=True)
    live.add_argument("--stable-manifest-sha256",required=True)
    for kind in ("approval","release","natural"):
        live.add_argument("--"+kind,type=Path,required=True)
        live.add_argument("--"+kind+"-sha256",required=True)
    arg=p.parse_args()
    if arg.mode=="freeze":
        blobs, schema, _=verified_stable()
        value=render(arg.output,blobs,schema)
    else:
        value=prepare(arg.output,arg.publication,arg.stable_manifest_sha256,
                      *(read(getattr(arg,k),getattr(arg,k+"_sha256"),16384) for k in ("approval","release","natural")),time.time())
    print(json.dumps(value,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
