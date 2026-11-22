"""Verify and exercise the sealed nine-pair VGGT input on the original common runtime."""

import argparse
import ast
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import sys


PARENT, NODE = "1103606", "f13r4n09"
TORCH = "2.10.0+das.opt1.dtk2604.2606161421.g91a580"
HIP = "6.3.26113"
FRAMEWORK = "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"
UPSTREAM = "64380b341ce377fb7e4b24c3f4f5986b6557b5339f4f7db57950dfa7e66b04c0"
ASSETS_MANIFEST = "647b78e286687cc1d55670f64472fbe0f395c1b81cbeb928999fcf415f6a59f7"
ORIGINAL = "fdc6b68196db7403a00dada66bda274d08d737e83336cb4c3758362f7438b14c"
ADDITION = "9a39e38c2acf5da7aacfc08729c60307cc048ea2293be81c83b40a69f6695cfa"
ORDER = "432012b6fce5fe874dd5883df184cd50f4f6d7dfb45f4b2699536839100c44d0"
WEIGHT_STAT = (5026874952, 20701942394, 1794489337000000000)
WEIGHT_SHA = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
OLD_BACKEND = "49bfe39b7aa6a14df9fdf1dda73e6db00836a81d2806012b312c16b2d3fbed6c"
BASE = PurePosixPath("generation/src/gencuda")
BACKEND = "backends/vggt_training.py"
SOURCES = {
    BACKEND: "762fc0d74a6ebb3ebbd052a8bdfbe302477a518a6868ec322bd89ef07e16c998",
    "vggt_eight_pair_settings.py": "6aef14e2326162edeca285a691e5629e74ca3800cb134348c668e2ff35f88816",
    "tum_vggt_pair.py": "c453a9c561c6411b8c2f25652f5c004c928db423c03ca0a45eba7d47ed9e3962",
    "data_contract.py": "aef0765500140c237fb8c57adfa53d0ba024064c67a1716e9610ca3448f4d898",
    "vggt_three_recording_inputs.py": "8eb59c13178aaa0236e1a7a00d6ebdbc2d864b64231eebfa4fd5259d5282cef2",
    "domain_overlay.py": "86823dac0aaf9e2b2436dbd1b3f2911f5ca16a948ded3fb708a3f13c23671d9b",
}
RECORDINGS = {"freiburg3_structure_texture_near": 7, "freiburg3_nostructure_texture_far": 1,
              "freiburg3_sitting_xyz": 1}
TENSOR_LAYOUTS = {
    "images": ((1,2,3,224,224), "torch.float32"),
    "depths": ((1,2,224,224), "torch.float32"),
    "extrinsics": ((1,2,3,4), "torch.float32"),
    "intrinsics": ((1,2,3,3), "torch.float32"),
    "point_masks": ((1,2,224,224), "torch.bool"),
}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(raw):
    return sha256(raw).hexdigest()


def read(path, expected, cap, count=None):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "expected an unaliased original file")
    with path.open("rb") as stream:
        raw = stream.read(cap+1)
    require(0 < len(raw) <= cap and (count is None or len(raw)==count) and digest(raw)==expected,
            f"frozen file differs: {path.name}")
    return raw


def relative(name):
    path=PurePosixPath(name)
    require(path.parts and not path.is_absolute() and ".." not in path.parts and "\\" not in name
            and path.as_posix()==name,"invalid frozen member name")
    return path


def save(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as out:
        out.write(raw)


def document(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


def dataset(stage_data):
    """Read only the 74 exact addressed files already staged from public assets."""
    merged=read(stage_data/"bundle.json",ASSETS_MANIFEST,2048)
    require(json.loads(merged)==dict(schema="vggt-tum-three-recording-native-inputs-v1",original_root="original",
                                     addition_root="sitting_xyz",input_order_SHA256=ORDER),"merged bundle differs")
    original=read(stage_data/"original/bundle.json",ORIGINAL,5000)
    files={"bundle.json":merged,"original/bundle.json":original}
    descriptors=[("original/"+relative(item["manifest"]).as_posix(),item["manifest_SHA256"],item["sequence"])
                 for item in json.loads(original)["pairs"]]
    descriptors.append(("sitting_xyz/pair-manifest.json",ADDITION,"freiburg3_sitting_xyz"))
    for name,pin,sequence in descriptors:
        raw=read(stage_data/name,pin,4096)
        pair=json.loads(raw)
        require(pair["sequence"]==sequence and len(pair["files_SHA256"])==7 and name not in files,
                "original RGB-D pair closure differs")
        files[name]=raw
        for member, expected in pair["files_SHA256"].items():
            entry=(PurePosixPath(name).parent/relative(member)).as_posix()
            require(entry not in files,"overlapping original RGB-D pair member")
            files[entry]=read(stage_data/entry,expected,2<<20)
    require(len(files)==74 and sum(map(len,files.values()))==12766333,"original 74-file asset closure differs")
    return files


def overlay(stage_sources):
    sources={key:read(stage_sources/key,pin,70000) for key,pin in SOURCES.items()}
    for raw in sources.values():
        ast.parse(raw,feature_version=(3,10))
    return sources


def install(stage, framework, framework_manifest, upstream, upstream_manifest, weights, candidate):
    """Check full original inputs before creating a separate immutable-use candidate."""
    assets=dataset(stage/"data")
    sources=overlay(stage/"gencuda")
    manifest_raw=read(framework_manifest,FRAMEWORK,1<<18)
    members=json.loads(manifest_raw)["files"]
    require(len(members)==393 and members[(BASE/BACKEND).as_posix()]["SHA256"]==OLD_BACKEND,
            "original common393 framework or VGGT base changed")
    frozen={}
    for name,item in members.items():
        canonical=relative(name).as_posix()
        frozen[canonical]=read(framework/canonical,item["SHA256"],2<<20,item["bytes"])
    upstream_members=json.loads(read(upstream_manifest,UPSTREAM,1<<17))["files"]
    require(len(upstream_members)==69,"original upstream source count differs")
    for item in upstream_members:
        read(upstream/relative(item["name"]).as_posix(),item["SHA256"],2<<20,item.get("bytes"))
    model=weights/"model.pt"
    info=model.lstat()
    require(model.resolve(strict=True)==model and stat.S_ISREG(info.st_mode)
            and (info.st_size,info.st_ino,info.st_mtime_ns)==WEIGHT_STAT,
            "qualified target weight stat changed")
    require(candidate.is_absolute() and candidate.parent.resolve(strict=True)==candidate.parent,
            "candidate requires a new canonical private namespace")
    candidate.mkdir(mode=0o700)
    frozen.update({(BASE/name).as_posix():raw for name,raw in sources.items()})
    for collection, subroot in ((frozen,candidate/"framework"),(assets,candidate/"data")):
        for name,raw in collection.items():
            save(subroot/name,raw)
    receipt=dict(state="EXACT_COMMON393_NINE_PAIR_TARGET_SOURCES_INSTALLED",framework_manifest_SHA256=FRAMEWORK,
                 upstream_manifest_SHA256=UPSTREAM,common_framework_original_members=393,
                 final_framework_members=len(frozen),overlay_SHA256=SOURCES,asset_bundle_SHA256=ASSETS_MANIFEST,
                 asset_files=len(assets),asset_bytes=sum(map(len,assets.values())),input_order_SHA256=ORDER,
                 planned_GPU_parent=PARENT,planned_GPU_node=NODE,qualified_model_stat=list(WEIGHT_STAT),
                 original_model_SHA256_from_successful_training=WEIGHT_SHA,model_payload_read=False,
                 native_callback_exercised=False,optimizer_updates=0,training_READY=False)
    save(candidate/"FRAMEWORK_FILES.json",manifest_raw)
    save(candidate/"install.json",document(receipt))
    return receipt


def native(candidate, install_sha):
    """Inside the original container/runtime, use Torch to execute the native data callback only."""
    receipt=json.loads(read(candidate/"install.json",install_sha,8192))
    require(receipt["framework_manifest_SHA256"]==FRAMEWORK and receipt["asset_bundle_SHA256"]==ASSETS_MANIFEST
            and receipt["overlay_SHA256"]==SOURCES and receipt["training_READY"] is False,
            "original candidate install receipt differs")
    require(os.environ.get("SLURM_JOB_ID")==PARENT and re.fullmatch(r"[0-9]+",os.environ.get("SLURM_STEP_ID",""))
            and socket.gethostname().split(".")[0]==NODE
            and all(os.environ.get(name)=="" for name in ("ROCR_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES")),
            "preflight requires the fixed CPU-only step and hidden accelerator visibility")
    assets=dataset(candidate/"data")
    framework=candidate/"framework"
    common=json.loads(read(candidate/"FRAMEWORK_FILES.json",FRAMEWORK,1<<18))["files"]
    changed={(BASE/name).as_posix():pin for name,pin in SOURCES.items()}
    for name,item in common.items():
        if name not in changed:
            read(framework/relative(name).as_posix(),item["SHA256"],2<<20,item["bytes"])
    for name,pin in changed.items():
        read(framework/name,pin,70000)
    require(len(common)==393 and receipt["final_framework_members"]==len(common.keys()|changed.keys()),
            "candidate common and overlay member closure differs")
    src=framework/"generation/src"
    sys.path.insert(0,str(src))
    import torch
    from gencuda import domain_overlay
    from gencuda.backends import vggt_training as backend

    require(importlib.metadata.version("torch")==TORCH and torch.version.hip==HIP
            and Path(backend.__file__)==src/"gencuda"/BACKEND
            and Path(domain_overlay.__file__)==src/"gencuda/domain_overlay.py",
            "preflight must use the installed common393 provider and qualified Torch runtime")
    root=candidate/"data"
    config=dict(domain_overlay.SETTINGS,dataset_root=str(root),annotation_root=str(root),
                dataset_manifest=str(root/"bundle.json"),dataset_sha=ASSETS_MANIFEST)
    require(backend._vggt_training_preflight(config)==("camera_head",False),"native training boundary differs")
    batches, evidence=domain_overlay.batches(torch,config,backend._tensor_batch,backend.valid_depth_batch)
    require(len(batches)==evidence["verified_pairs"]==9 and evidence["physical_recordings"]==RECORDINGS
            and evidence["input_order_SHA256"]==ORDER,"native input order or recordings differ")
    signatures=[]
    for batch in batches:
        signatures.append({})
        for name,(shape,dtype) in TENSOR_LAYOUTS.items():
            tensor=batch[name]
            require(tensor.device.type=="cpu" and tuple(tensor.shape)==shape and str(tensor.dtype)==dtype
                    and (name=="point_masks" or torch.isfinite(tensor).all().item()),
                    "native CPU tensor shape, dtype or values differ")
            signatures[-1][name]=digest(tensor.contiguous().numpy().tobytes())
        require(torch.equal(batch["point_masks"],batch["depths"]>0) and backend.valid_depth_batch(batch["point_masks"]),
                "native depth mask or valid pixels differ")
    return dict(state="NINE_REAL_NATIVE_TORCH_CPU_INPUTS_VERIFIED",install_SHA256=install_sha,
                source_framework_manifest_SHA256=FRAMEWORK,overlay_SHA256=SOURCES,asset_bundle_SHA256=ASSETS_MANIFEST,
                input_order_SHA256=ORDER,view_order_SHA256=digest(document(backend._vggt_view_order(batches))),
                batch_tensor_signatures_SHA256=digest(document(signatures)),asset_files=len(assets),
                native_real_batches=9,recordings=RECORDINGS,minimum_valid_depth_pixels=100,
                cpu_step=PARENT+"."+os.environ["SLURM_STEP_ID"],node=NODE,torch=TORCH,hip_metadata=HIP,
                model_payload_read=False,explicit_gpu_calls=0,optimizer_updates=0,training_READY=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    phases=parser.add_subparsers(dest="mode",required=True)
    assets=phases.add_parser("assets")
    assets.add_argument("--stage",type=Path,required=True)
    staged=phases.add_parser("install")
    for name in ("stage","framework","framework_manifest","upstream","upstream_manifest","weights","candidate"):
        staged.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    run=phases.add_parser("native")
    run.add_argument("--candidate",type=Path,required=True)
    run.add_argument("--install-sha256",required=True)
    args=parser.parse_args()
    fields=vars(args).copy()
    fields.pop("mode")
    if args.mode=="assets":
        stage=args.stage.absolute()
        source=overlay(stage/"gencuda")
        data=dataset(stage/"data")
        result=dict(asset_files=len(data),asset_bytes=sum(map(len,data.values())),source_files=len(source),
                    asset_bundle_SHA256=ASSETS_MANIFEST,input_order_SHA256=ORDER,training_READY=False)
    elif args.mode=="install":
        result=install(**fields)
        result["install_SHA256"]=digest((args.candidate/"install.json").read_bytes())
    else:
        result=native(args.candidate,args.install_sha256)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
