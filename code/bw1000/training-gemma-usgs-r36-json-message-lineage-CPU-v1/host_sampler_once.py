"""Use the qualified CPU transport with the published R31 HOST body."""
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys
from types import FunctionType

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
LABEL=HERE.name
CARRIER=REPO/"code/bw1000/training-gemma-usgs-r28-sampler-native-CPU-v1/host_sampler_once.py"
PIN="74774933dd02537b26ff146ca3cae9e60d9b60d4e7a23e70628a8eb62ed225b1"
CLAIM=REPO/"coordination/tasks"/f"{LABEL}.json"
FILES=("probe.py","probe_r36.py","arrow_lineage.py","cleanup.py","r30_cache_gate.py","r31_modelscope_cache.py","ipc_witness.py","MODEL_PROCESSOR_FILES.json","SOURCE_FILES.json","prepare.py","HOST_BODY.py")


def digest(blob):
    return sha256(blob).hexdigest()


def main():
    if digest(CARRIER.read_bytes())!=PIN:
        raise ValueError("Immutable qualified carrier source differs")
    claim=json.loads(CLAIM.read_bytes())
    original=runpy.run_path(str(CARRIER))
    if set(claim["sources"])!=set(FILES) or claim["host_sampler_SHA256"]!=digest(Path(__file__).read_bytes()):
        raise ValueError("R31 source closure differs")
    pinned=original["pinned"]
    parts={name:pinned(HERE/name,claim["sources"][name]) for name in FILES}
    # HOST_BODY is bound by the published claim and exact invocation hash; only compute members enter the target src mount.
    body=parts.pop("HOST_BODY.py").decode()
    compile(body,"<bound original R31 HOST body>","exec")
    if len(sys.argv)==2 and sys.argv[1]=="check":
        scope=claim["request"]
        namespace={"__file__":f"/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/{LABEL}/{scope}/prepare.py",
                   "__name__":"same_request_command"}
        exec(compile(parts["prepare.py"],"<exact published R31 entry>","exec"),namespace)
        argv=namespace["argv"](claim["parent"],claim["node"],scope,1795591200,claim["sources"]["SOURCE_FILES.json"],
                               claim["allow_component_fallback"])
        manifest=json.loads(parts["SOURCE_FILES.json"])
        assert set(manifest["files"])==set(parts)-{"SOURCE_FILES.json","prepare.py"}
        assert all(digest(parts[name])==pin["SHA256"] and len(parts[name])==pin["bytes"] for name,pin in manifest["files"].items())
        assert argv.count("--bind")==13 and "--gres=none" in argv and "--overlap" not in argv
        assert "1103606.3355" in body and "training80-dcu-standby-4d63273bbce5-f13r1n12" in body
        print(json.dumps(dict(compiled=True,native_argv=True,CPU=4,GPU=0,exact_manifest_members=len(manifest["files"]))))
        return
    # Keep the published full claim in the transport. Project out the HOST-only source at the remote data boundary.
    anchor="and {key:digest(raw) for key,raw in sources.items()}==CLAIM['sources']"
    if body.count(anchor)!=1:
        raise ValueError("Exact remote source manifest anchor changed")
    body=body.replace(anchor,"and {key:digest(raw) for key,raw in sources.items()}=={k:v for k,v in CLAIM['sources'].items() if k!='HOST_BODY.py'}",1)
    env=dict(original,__file__=str(Path(__file__).resolve()),ROOT=REPO,PACKAGE=HERE,LABEL=LABEL,CLAIM=CLAIM,
             BUNDLE=tuple(parts),HOST=body)
    env["render"]=FunctionType(original["render"].__code__,env)
    import ast
    import inspect
    check=inspect.getsource(original["main"])
    anchor='outcome_seconds=154 if args.mode=="launch" else 42'
    if check.count(anchor)!=1:
        raise ValueError("Exact original owner wait anchor changed")
    check=check.replace(anchor,'outcome_seconds=278 if args.mode=="launch" else 42',1)
    exec(compile(ast.parse(check),"<pinned original carrier with full 240s wait>","exec"),env)
    env["main"]()


if __name__=="__main__":
    main()
