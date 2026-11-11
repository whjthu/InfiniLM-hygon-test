"""Read exact VLM launch/input identities and bounded asset metadata, with no model execution."""
import ast,datetime,hashlib,json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
A=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline")
B=BASE.parent
recipes=[
("T01.sft","node26","qwen-cuda8-gc-on-v1-sdpa-r1"),
("T02.sft","node27","gemma-cuda8-gc-off-v7-sdpa-r1"),
("T03.sft","node27","internvl-cuda8-gc-on-v1-sdpa-r1"),
("T04.sft","node27","llava-cuda8-gc-off-v2-eager-r1"),
("T05.sft","node27","minicpm-cuda8-gc-off-v2-native-fast-sdpa-r1")]
module=ast.parse((B/"hygon-training-integration/tools/hygon_vlm_recipe.py").read_bytes())
pins=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="A100_LAUNCH_SHA256" for t in n.targets))
binding=[]
for case,node,label in recipes:
    path=A/"cross_platform_reference_r1/raw"/node/"node/results"/label/"launch.json"
    raw=path.read_bytes(); pin=hashlib.sha256(raw).hexdigest()
    assert pin==pins[case]
    value=json.loads(raw); argv=value["command"]
    def opt(name):return argv[argv.index(name)+1] if name in argv else None
    i=argv.index("--external_plugins")+1
    plugins=[]
    while i<len(argv) and not argv[i].startswith("--"):
        plugins.append(argv[i]);i+=1
    binding.append(dict(case=case,launch_path=path.relative_to(A).as_posix(),launch_SHA256=pin,
        model=opt("--model"),dataset=opt("--dataset"),plugins=plugins,
        recipe={name:opt("--"+name) for name in ("model_type","template","max_length","attn_impl","gradient_checkpointing","freeze_vit","freeze_aligner","lora_rank","lora_alpha","lora_dropout","per_device_train_batch_size","gradient_accumulation_steps","max_steps","warmup_steps","warmup_ratio")},
        world_size=int(value["environment"]["NPROC_PER_NODE"])))
meta_path=A/"coordination/a100-alignment-v1/vlm-coco64-and-callback-original-metadata-r2.json"
raw=meta_path.read_bytes(); meta=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()=="e2c86d2575311d80897ec2406ac7a175f04e80bea168ca714a3188288101a83f"
assert meta["row_count"]==64 and meta["ordered_unique_images"]==64
common=dict(metadata_SHA256=hashlib.sha256(raw).hexdigest(),
    JSONL_files=meta["JSONL_files"],canonical_rows_SHA256=meta["canonical_rows_sha256"],
    callback={k:v for k,v in meta["callback"].items() if k!="source_base64"},
    image_count=64,image_bytes=meta["images_bytes_freshly_read_and_SHA_matched"],
    image_identity=[{k:v for k,v in row.items() if k in ("bytes","relative_role","sha256","ordered_index")} for row in meta["image_identity"]])
with (BASE/"original-bindings.json").open("x",encoding="utf8") as f:json.dump(dict(cases=binding,common=common),f,indent=2)
REMOTE=r"""
import datetime,hashlib,json,subprocess
from pathlib import Path
def small(p):
    if not p.is_file():return dict(exists=False)
    size=p.stat().st_size
    if size>8*1024*1024:return dict(exists=True,bytes=size,hash_not_read=True)
    raw=p.read_bytes()
    return dict(exists=True,bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
def model(p):
    row=dict(path=str(p),exists=p.is_dir())
    if not row["exists"]:return row
    names=sorted(p.iterdir(),key=lambda x:x.name)
    assert len(names)<=256
    row["immediate_files"]=[dict(name=x.name,bytes=x.stat().st_size,is_file=x.is_file()) for x in names]
    row["small_identities"]={x.name:small(x) for x in names if x.is_file() and (x.name in ("config.json","tokenizer_config.json","preprocessor_config.json","processor_config.json") or x.name.endswith(".index.json"))}
    return row
out=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
if PLATFORM=="a100":
    out["models"]=[model(Path(p)) for p in MODELS]
    out["dataset"]=small(Path(DATASET))
    out["plugins"]={p:small(Path(p)) for p in PLUGINS}
else:
    root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
    out["parent"]=subprocess.check_output(["squeue","-h","-j","1087656","-o","%i|%T|%D|%L|%N"],text=True,timeout=12)
    out["steps"]=subprocess.check_output(["squeue","--steps","-h","-j","1087656","-o","%i|%j|%N"],text=True,timeout=12)
    shared=Path("/public/home/aclsylqidf/shared")
    out["shared_root_exists"]=shared.is_dir()
    if shared.is_dir():
        entries=sorted(shared.iterdir(),key=lambda x:x.name)
        assert len(entries)<=256
        out["shared_immediate_names"]=[x.name for x in entries]
        wanted={Path(p).name.lower() for p in MODELS}
        out["matching_models"]=[model(p) for p in entries if p.name.lower() in wanted]
    out["exact_destination_candidates"]={str(root/"results/training-vlm-original-assets-v1"):dict(exists=(root/"results/training-vlm-original-assets-v1").exists())}
print(json.dumps(out))
"""
models=[x["model"] for x in binding]
dataset=binding[0]["dataset"]
plugins=sorted(set(p for x in binding for p in x["plugins"]))
results={}
for platform in ("a100","hygon"):
    if platform=="a100":
        ssh=["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(A/"ssh_config"),"a100-26","python3"]
    else:
        prelude=(B/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf8")
        namespace={"__file__":str(Path(__file__).resolve())}
        exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),namespace)
        ssh=namespace["ssh"]
    code=f"PLATFORM={platform!r}\nMODELS={models!r}\nDATASET={dataset!r}\nPLUGINS={plugins!r}\n"+REMOTE
    reply=subprocess.run(ssh,input=code.encode(),capture_output=True,timeout=65)
    (BASE/(platform+"-stdout.json")).write_bytes(reply.stdout)
    (BASE/(platform+"-stderr.bin")).write_bytes(reply.stderr)
    row=dict(rc=reply.returncode,stdout_SHA256=hashlib.sha256(reply.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(reply.stderr).hexdigest())
    if reply.returncode==0:row["value"]=json.loads(reply.stdout)
    results[platform]=row
with (BASE/"original-observation.json").open("x",encoding="utf8") as f:json.dump(results,f,indent=2)
print(json.dumps({k:dict(rc=v["rc"],models_present=sum(m["exists"] for m in v.get("value",{}).get("models",[])),parent=v.get("value",{}).get("parent"),steps=v.get("value",{}).get("steps"),shared_names=v.get("value",{}).get("shared_immediate_names")) for k,v in results.items()}))
