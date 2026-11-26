"""Inspect or encode the exact rendered ABO clip through the deployed native VAE."""

import argparse
from hashlib import sha256
import inspect
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import struct
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import contract as c


def placement(a):
    env=os.environ
    numeral=env.get("SLURM_STEP_ID","")
    c.require(re.fullmatch(r"[a-z][a-z0-9]{2,20}",a["node"]) is not None
              and re.fullmatch(r"[0-9]+",a["parent"]) is not None and re.fullmatch("[0-9]+",numeral)
              and env.get("SLURM_JOB_ID")==a["parent"] and socket.gethostname().split(".")[0]==a["node"]
              and all(env.get(k)=="1" for k in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE")),
              "EXACT_NUMERIC_OFFLINE_ALLOCATION")
    if a["mode"]=="inspect":
        c.require(env.get("SLURM_STEP_GPUS","") in ("","none","N/A","(null)")
            and all(env.get(k,"0") in ("","0","(null)") for k in ("SLURM_GPUS_PER_TASK","SLURM_GPUS_ON_NODE"))
            and not re.search("gpu|dcu",env.get("SLURM_TRES_PER_TASK",""),re.I)
            and all(env.get(k)=="-1" for k in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES")),
            "EXACT_ZERO_GRES_VISIBILITY")
    return a["parent"]+"."+numeral


def asset_metadata(authorization,outside,full):
    proof_raw=c.read(outside,8192)
    c.require(c.pin(proof_raw)["SHA256"]==c.VAE_READBACK_SHA,"ORIGINAL_VAE_METADATA_WITNESS")
    original=json.loads(proof_raw)
    c.require(original["original_verified_payload_SHA256"]==c.WEIGHTS_PIN["SHA256"]
              and original["original_terminal_SHA256"]==c.VAE_ORIGINAL_SHA
              and original["header_SHA256"]==c.HEADER_PIN["SHA256"],"ORIGINAL_VAE_CHAIN")
    config=json.loads(c.read(c.VAE/"config.json",c.CONFIG_PIN["bytes"],c.CONFIG_PIN))
    c.require(config["_class_name"]=="AutoencoderKLConv3D" and config["latent_channels"]==32
              and config["scaling_factor"]==1.03682 and config.get("shift_factor") is None,"ORIGINAL_VAE_CONFIG")
    path=c.VAE/c.MODEL
    stream,before=c.open_regular(path)
    with stream:
        prefix=stream.read(8)
        c.require(len(prefix)==8 and struct.unpack("<Q",prefix)[0]==c.HEADER_PIN["bytes"]
                  and before.st_size==c.WEIGHTS_PIN["bytes"],"ORIGINAL_VAE_HEADER_AND_EXTENT")
        header=stream.read(c.HEADER_PIN["bytes"])
        c.require(c.pin(header)==c.HEADER_PIN,"ORIGINAL_VAE_HEADER_SHA")
        parsed=json.loads(header)
        entries=[value for name,value in parsed.items() if name!="__metadata__"]
        c.require(len(entries)==218 and max(v["data_offsets"][1] for v in entries)==before.st_size-8-len(header),
                  "ORIGINAL_VAE_SAFETENSORS_TENSOR_EXTENT")
        actual=None
        if full:
            c.require(authorization["weight_payload_GO"] is True,"FULL_NATIVE_WEIGHT_ACCESS_UNAUTHORIZED")
            h=sha256(prefix+header)
            while chunk:=stream.read(8<<20):
                h.update(chunk)
            actual=h.hexdigest()
            c.require(actual==c.WEIGHTS_PIN["SHA256"],"ACTUAL_WHOLE_VAE_SHA")
        c.stable(path,before,stream.fileno())
    return dict(config=c.CONFIG_PIN,header=c.HEADER_PIN,tensors=len(entries),weight=c.WEIGHTS_PIN,
                actual_whole_weight_SHA256=actual,historical_whole_weight_SHA256=c.WEIGHTS_PIN["SHA256"],
                current_weight_stamp=list(c.stamp(before)),current_weight_bytes=before.st_size,
                weight_payload_bytes_read=before.st_size-8-len(header) if full else 0)


def verify_source(path):
    raw=c.read(path,32768)
    c.require(c.pin(raw)["SHA256"]==c.SOURCE_BINDING_SHA,"ORIGINAL_UPSTREAM_BINDING")
    binding=json.loads(raw)
    files=binding["files"]
    c.require(len(files)==38 and files[c.SOURCE]["SHA256"]==c.SOURCE_PIN["SHA256"]
              and files[c.SOURCE]["bytes"]==c.SOURCE_PIN["bytes"],
              "ORIGINAL_DEPLOYED_VAE_SOURCE_IDENTITY")
    for name,item in files.items():
        rel=Path(name)
        c.require(not rel.is_absolute() and ".." not in rel.parts and type(item["bytes"]) is int,"SAFE_SOURCE_MEMBER")
        c.read(c.NATIVE/rel,item["bytes"],{k:item[k] for k in ("bytes","SHA256")})
    return dict(upstream_commit=binding["upstream_commit"],member_count=len(files),active_class_source=c.SOURCE_PIN)


def decode_frames(media):
    from PIL import Image
    import numpy as np

    raw=c.read(media/"render.json",8192)
    receipt=json.loads(raw)
    c.require(c.pin(raw)["SHA256"]==c.RENDER_RECEIPT_SHA and receipt["device"]=="CPU"
              and receipt["frame_count"]==9,"EXACT_RENDER_RECEIPT")
    ledger=c.read(media/"frames.jsonl",16384)
    c.require(c.pin(ledger)["SHA256"]==receipt["ledger_SHA256"],"EXACT_RENDER_LEDGER")
    decoded=[]
    rgb=[]
    png=[]
    rows=ledger.splitlines()
    c.require(len(rows)==9,"EXACT_NINE_FRAME_ROWS")
    for ordinal,data in enumerate(rows):
        row=json.loads(data)
        name=f"frame-{ordinal:02d}.private.png"
        c.require(row["name"]==name and row["index"]==ordinal and row["azimuth_degrees"]==ordinal*15,
                  "FROZEN_FRAME_SEQUENCE")
        raw=c.read(media/name,2<<20,row["pin"])
        with Image.open(io.BytesIO(raw),formats=("PNG",)) as image:
            c.require(image.mode=="RGB" and image.size==(848,480),"ORIGINAL_FRAME_RGB_EXTENT")
            value=image.tobytes()
        actual=c.pin(value)["SHA256"]
        c.require(actual==c.RGB[ordinal],"FROZEN_DECODED_RGB_SHA")
        rgb.append(actual)
        png.append(row["pin"])
        decoded.append(np.frombuffer(value,dtype=np.uint8).reshape(480,848,3))
    return np.stack(decoded),dict(decoded_RGB_SHA256=rgb,png=png,shape_THWC=[9,480,848,3],
                                  source_render_SHA256=c.RENDER_RECEIPT_SHA)


def device_context(out):
    c.require(os.environ.get("PYTHONDONTWRITEBYTECODE")=="1","NO_SHARED_BYTECODE")
    for name in ("HOME","XDG_CONFIG_HOME","XDG_CACHE_HOME"):
        loc=Path(os.environ[name]).resolve(strict=True)
        c.require(loc.is_dir() and loc.is_relative_to(out.parent) and loc.stat().st_uid==os.getuid()
                  and loc.stat().st_mode&0o077==0 and not os.statvfs(loc).f_flag&os.ST_RDONLY,
                  "REQUEST_PRIVATE_WRITABLE_HOME_CACHE")


def native_encode(frames,asset,source,report):
    import torch
    c.require(torch.version.hip=="6.3.26113" and torch.cuda.device_count()==1,"EXACT_ONE_DCU_HIP")
    arch=str(torch.cuda.get_device_properties(0).gcnArchName).split(":",1)[0]
    c.require(arch=="gfx936","TARGET_GFX936_REQUIRED")
    report["actual_GPU"]=1
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    sys.path.insert(0,str(c.NATIVE))
    from hyvideo.models.autoencoders.hunyuanvideo_15_vae import AutoencoderKLConv3D

    module=Path(inspect.getfile(AutoencoderKLConv3D)).resolve(strict=True)
    c.require(module==c.NATIVE/c.SOURCE and c.pin(c.read(module,c.SOURCE_PIN["bytes"]))==c.SOURCE_PIN,
              "ACTUAL_IMPORTED_NATIVE_VAE_CLASS")
    c.require(set(x.name for x in c.VAE.iterdir())=={"config.json",c.MODEL},"EXACT_TWO_VAE_MODEL_MEMBERS")
    path=c.VAE/c.MODEL
    c.require(list(c.stamp(path.lstat()))==asset["current_weight_stamp"],"VAE_IDENTITY_BEFORE_NATIVE_LOAD")
    vae,info=AutoencoderKLConv3D.from_pretrained(str(c.VAE),torch_dtype=torch.bfloat16,
                        local_files_only=True,use_safetensors=True,output_loading_info=True)
    c.require(isinstance(info,dict) and all(not info.get(key) for key in
        ("missing_keys","unexpected_keys","mismatched_keys","error_msgs"))
        and list(c.stamp(path.lstat()))==asset["current_weight_stamp"],"STRICT_NATIVE_VAE_LOAD")
    vae=vae.requires_grad_(False).to("cuda:0").eval()
    c.require(float(vae.config.scaling_factor)==1.03682 and getattr(vae.config,"shift_factor",None) is None,
              "LOADED_VAE_SCALE_SHIFT")
    with torch.inference_mode():
        x=(torch.from_numpy(frames).permute(3,0,1,2).unsqueeze(0).float()*(2.0/255.0)-1.0).contiguous()
        c.require(list(x.shape)==c.INPUT_SHAPE and bool(torch.isfinite(x).all()) and float(x.min())>=-1.0
                  and float(x.max())<=1.0,"RGB_NEGATIVE_ONE_TO_ONE_NCTHW")
        x=x.to(device="cuda:0",dtype=torch.bfloat16)
        report["native_encode_attempts"]=1
        encoded=vae.encode(x,return_dict=True)
        report["actual_native_encode_calls"]=1
        distribution=encoded.latent_dist
        mode=distribution.mode()
        c.require(list(mode.shape)==c.SHAPE and mode.device.type=="cuda" and mode.dtype in (torch.bfloat16,torch.float32),
                  "NATIVE_POSTERIOR_MODE_SHAPE_AND_DTYPE")
        scale=float(vae.config.scaling_factor)
        shift=float(getattr(vae.config,"shift_factor",None) or 0.0)
        final=(mode.float()-shift)*scale
        torch.cuda.synchronize()
        payload=final.detach().cpu().contiguous().numpy().astype("<f4",copy=False).tobytes()
    return payload,dict(actual_native_encode_calls=1,device="cuda:0",architecture=arch,hip=torch.version.hip,
                        torch=str(torch.__version__),config_scale=scale,config_shift=shift,posterior="mode",
                        native_input_shape=c.INPUT_SHAPE,native_input_dtype="bfloat16",
                        posterior_mode_dtype=str(mode.dtype),
                        compute_tiling_requested=False,active_class_source=source["active_class_source"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("inspect","encode"))
    for field in ("authority","vae-metadata","media","upstream-binding","out"):
        p.add_argument("--"+field,type=Path,required=True)
    p.add_argument("--authority-sha256",required=True)
    p.add_argument("--request",required=True)
    args=p.parse_args()
    raw=c.read(args.authority,16384)
    c.require(c.pin(raw)["SHA256"]==args.authority_sha256,"EXACT_ROOT_AUTHORITY_SHA256")
    auth=c.permit(raw,args.request,args.mode,time.time())
    step=placement(auth)
    c.require(args.out.is_absolute() and args.out.parent.name==args.request,"NEW_PRIVATE_OPERATION_DIRECTORY")
    out=args.out
    out.mkdir(mode=0o700)
    def stop(*_):
        raise TimeoutError("EXACT_WORKER_EPOCH")
    signal.signal(signal.SIGALRM,stop)
    signal.setitimer(signal.ITIMER_REAL,max(.001,auth["deadline"]-time.time()))
    def no_network(event,_):
        if event in {"socket.connect","socket.getaddrinfo","subprocess.Popen","os.system"}:
            raise RuntimeError("OFFLINE_ONLY")
    sys.addaudithook(no_network)
    report=dict(schema="HUNYUAN_ABO_VAE_CANDIDATE_WORKER_V1",request=args.request,mode=args.mode,actual_step=step,
        recipe=c.RECIPE,authority_SHA256=args.authority_sha256,passed=False,pid=os.getpid(),planned_frames=9,
        node=auth["node"],epoch=auth["epoch"],deadline=auth["deadline"],
        validated_frames=0,planned_candidate_latents=1 if args.mode=="encode" else 0,
        actual_candidate_latents=0,training_rows=0,actual_GPU=0,native_encode_attempts=0,actual_native_encode_calls=0,
        optimizer_updates=0,text_embeddings=0,old_cache_equivalence_verified=False)
    try:
        report["phase"]="deployed_source_and_media"
        c.require(all(os.statvfs(path).f_flag&os.ST_RDONLY for path in
                      (c.NATIVE,c.VAE,args.media,args.vae_metadata,args.upstream_binding)),
                  "EXACT_INPUTS_MOUNTED_READONLY")
        report["source"]=verify_source(args.upstream_binding)
        frames,report["media"]=decode_frames(args.media)
        report["validated_frames"]=len(frames)
        report["phase"]="vae_asset_identity"
        report["asset"]=asset_metadata(auth,args.vae_metadata,args.mode=="encode")
        if args.mode=="encode":
            report["phase"]="native_encode"
            device_context(out)
            payload,report["runtime"]=native_encode(frames,report["asset"],report["source"],report)
            report["latent"]=c.payload_summary(payload,c.SHAPE)
            c.require(c.write(out/c.LATENT,payload)==report["latent"]["payload"],"LATENT_WRITTEN_BYTES")
            report["actual_candidate_latents"]=1
        report.update(passed=True,phase="sealed")
    except Exception as e:
        tag=str(e) if type(e) is ValueError else ""
        report["failure"]=dict(type=type(e).__name__,code=tag if re.fullmatch("[A-Z0-9_]{1,100}",tag) else None)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
    result=c.write(out/"worker.json",c.encoded(report))
    print(json.dumps(dict(request=args.request,worker=result,actual_candidate_latents=report["actual_candidate_latents"],
                          validated_frames=report["validated_frames"],passed=report["passed"]),sort_keys=True))
    return int(not report["passed"])


if __name__=="__main__":
    raise SystemExit(main())
