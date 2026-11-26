"""Frozen source identity and independent ABO VAE latent payload verification."""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import struct


LABEL="training-hunyuanvideo-abo-vae-encode-v1"
PROJECT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
VAE=PROJECT/"results/training-hunyuanvideo-public-vae-v1/vae"
NATIVE=PROJECT/"results/training-hunyuanvideo-original-source-v1/source"
SOURCE="hyvideo/models/autoencoders/hunyuanvideo_15_vae.py"
SOURCE_PIN=dict(bytes=38331,SHA256="4078879df38616dd95b44f0f4660ffcfdf60c348fa0da33574a21d8eedad8a5b")
SOURCE_BINDING_SHA="26f1c4e5ef857bb04f5287b4386d4aa9a17ff8cd0bf65d0cece23a5a402633a7"
MODEL="diffusion_pytorch_model.safetensors"
CONFIG_PIN=dict(bytes=462,SHA256="899e63dae033b2fa14291cefbe6e1736741d816991456e96d4d497918dacf8af")
HEADER_PIN=dict(bytes=24512,SHA256="c95c98a4a65009e8e411132b303acc95d6ae553aef00a223d7bd31e2d03b2dee")
WEIGHTS_PIN=dict(bytes=5042560980,SHA256="2609ed7033c052fdf164afcac71e8a7c82afccdbb001d673444a72f194fbb918")
VAE_READBACK_SHA="e0e33d5f24a6a9fa95baa3817d8d8e8f79b896db0f7586c83353d278f4c18443"
VAE_ORIGINAL_SHA="915eb64e703db7657b4729b46d938f08c5069ac444655e505ae10c74aa1676bc"
RENDER_RECEIPT_SHA="0c0749760fcabab3f01be289c1103e1fa47fc167dd736d295cd51f24e7a44b5f"
MEDIA_PUBLIC_SHA="18561826f244f1de300d593a5854a5551d2b518a0fdc37905f87d4b78012be58"
RGB=(
    "3cbafdba5a00f28414a75ecc92584181e7fd969ef8f6ac81eee1245a9bb573a4",
    "08c33287ce29f3de8fc2e909fe40726f5beb340fbe47c004592577614d2fcf09",
    "c4bcff8a75eb40ad5e2b7d2bc633e6f8651c31fde58062cf08a0dcb1e05e7655",
    "61a76ece75705528383701663cf6b2cc0faa3d4bc378f8fd1555dd82e779df8d",
    "83496066277f57040d93fa61d71848fef0e2f1ebda51c1f123b03809ab8aa02d",
    "7e792c93bf9833372fde48dc16bf1de72a0f3c4743ea1db8cfbd947a8f1c7d5e",
    "fa2d56f0ea46061757a231c71d403fa8776a060b8857f6041886a42ae0c9da55",
    "653c006f64569c0fc6e06151bda7d22031e0634287165d10ee5b8aa85d882749",
    "5a301603feddc8a34350197069f255a12bec904d31a6333d7e294efcd079897b",
)
INPUT_SHAPE=[1,3,9,480,848]
SHAPE=[1,32,3,30,53]
BYTES=610560
LATENT="latent.private.f32le"
RECIPE="ABO_RGB9_NCTHW_BF16_NATIVE_POSTERIOR_MODE_FP32_SCALED_V1"


def require(test, code):
    if not test:
        raise ValueError(code)


def pin(raw):
    return dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())


def encoded(value):
    return (json.dumps(value,sort_keys=True,allow_nan=False)+"\n").encode()


def stamp(st):
    return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns


def open_regular(path):
    before=path.lstat()
    require(stat.S_ISREG(before.st_mode),"REGULAR_UNALIASED_FILE")
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_BINARY",0))
    return os.fdopen(fd,"rb"),before


def stable(path,before,fd):
    now,after=stamp(os.fstat(fd)),stamp(path.lstat())
    original=stamp(before)
    # Windows descriptor and path ctime can follow different semantics.
    require(original==after and (now[:4]==original[:4] if os.name=="nt" else now==original),"FILE_MUTATED")


def read(path,cap,expected=None):
    stream,before=open_regular(path)
    with stream:
        raw=stream.read(cap+1)
        stable(path,before,stream.fileno())
    require(0<len(raw)<=cap and (expected is None or pin(raw)==expected),"EXACT_FILE_EXTENT_OR_SHA")
    return raw


def write(path,raw):
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return pin(raw)


def payload_summary(raw,shape):
    require(shape==SHAPE and len(raw)==BYTES,"EXACT_LATENT_SHAPE_AND_SIZE")
    lo=math.inf
    hi=-math.inf
    count=0
    for (value,) in struct.iter_unpack("<f",raw):
        require(math.isfinite(value),"LATENT_NONFINITE")
        lo,hi=min(lo,value),max(hi,value)
        count+=1
    require(lo<hi and count==math.prod(SHAPE),"LATENT_CONSTANT_OR_EMPTY")
    return dict(payload=pin(raw),dtype="float32",byte_order="little",layout="C/NCTHW",
                shape=SHAPE,elements=count,minimum=lo,maximum=hi,finite=True,nonconstant=True)


def permit(raw,request,mode,now):
    a=json.loads(raw)
    require(re.fullmatch("[0-9a-f]{32}",request) is not None and a["request"]==request
            and a["label"]==LABEL and a["mode"]==mode and a["recipe"]==RECIPE
            and a["root_authorized"] is True and a["network_GO"] is a["training_GO"] is a["text_GO"] is False
            and a["zero_GRES_GO"] is (mode=="inspect") and a["GPU_GO"] is (mode=="encode")
            and a["weight_payload_GO"] is (mode=="encode") and type(a["allocatable_DCU"]) is int
            and a["allocatable_DCU"]==int(mode=="encode")
            and all(type(a[k]) in (float,int) and math.isfinite(a[k]) for k in ("epoch","deadline"))
            and a["epoch"]<=now<a["deadline"]<=a["epoch"]+(120 if mode=="inspect" else 600)
            and a["media_receipt_SHA256"]==RENDER_RECEIPT_SHA and a["upstream_binding_SHA256"]==SOURCE_BINDING_SHA
            and a["vae_metadata_SHA256"]==VAE_READBACK_SHA and a["original_full_weight_SHA256"]==WEIGHTS_PIN["SHA256"],
            "EXACT_MODE_SPECIFIC_ROOT_PERMISSION")
    return a
