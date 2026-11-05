"""Typed component-only API. Import does not load a library or import Torch."""
import ctypes as C
import json
from hashlib import sha256
from pathlib import Path

def declare(lib):
    P=C.c_void_p;Z=C.c_size_t;I=C.c_int;S=C.c_char_p
    signatures={"error":([],S),"id_size":([],Z),"id":([P],I),
        "create":([I,I,S,P,I],P),"keys":([P],S),"load":([P,S],I),
        "finish":([P],I),"step":([P,Z,Z,P,I],I),"snapshot":([P],S),
        "read":([P,Z,P,Z],I),"reset":([P],I),"destroy":([P,P],I)}
    for name,(args,ret) in signatures.items():
        f=getattr(lib,"glm_l0_"+name);f.argtypes=args;f.restype=ret
    return lib

def checked(lib,rc):
    if rc:raise RuntimeError(lib.glm_l0_error().decode())

def open_library(path,expected):
    if sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError("component DSO differs")
    return declare(C.CDLL(str(path)))

def load_checkpoint(lib,ctx,*,model,torch,safe_open,expected_metadata,deadline_check):
    """Same raw FP8/U8, BF16/F32 and Native Weight sharding as actual loader."""
    model=Path(model);raw=(model/"model.safetensors.index.json").read_bytes()
    if sha256(raw).hexdigest()!=expected_metadata["index_SHA256"]:raise ValueError("checkpoint index differs")
    mapping=json.loads(raw)["weight_map"];keys=json.loads(lib.glm_l0_keys(ctx))
    if set(keys)!=set(expected_metadata["selected_keys"])or len(keys)!=21:raise ValueError("actual 21 Layer0 checkpoint keys differ")
    loaded=[];source_bytes=0
    # One true checkpoint tensor lives through synchronous Native load return.
    for key in sorted(keys):
        deadline_check("checkpoint read "+key)
        with safe_open(model/mapping[key],framework="pt",device="cpu")as f:
            w=f.get_tensor(key).contiguous();meta=expected_metadata["selected_keys"][key]
            if list(w.shape)!=meta["shape"]or w.numel()*w.element_size()!=meta["tensor_bytes"]:raise ValueError("checkpoint tensor shape/bytes differ")
            if w.dtype==torch.float8_e4m3fn:w=w.view(torch.uint8)
            types={torch.float32:0,torch.bfloat16:1,torch.uint8:2}
            if w.dtype not in types:raise ValueError("checkpoint dtype differs")
            desc=[dict(name=key,pointer=w.data_ptr(),shape=list(w.shape),dtype=types[w.dtype])]
            checked(lib,lib.glm_l0_load(ctx,json.dumps(desc).encode()));source_bytes+=w.numel()*w.element_size();loaded.append(key)
    checked(lib,lib.glm_l0_finish(ctx))
    return dict(checkpoint_keys=loaded,source_tensor_bytes=source_bytes,Model=False)

def read_snapshot(lib,ctx):
    raw=lib.glm_l0_snapshot(ctx)
    if raw is None:raise RuntimeError(lib.glm_l0_error().decode())
    meta=json.loads(raw);fields={}
    for i,row in enumerate(meta["fields"]):
        b=C.create_string_buffer(row["bytes"]);checked(lib,lib.glm_l0_read(ctx,i,b,row["bytes"]))
        fields[row["name"]]=dict(row,raw=b.raw,SHA256=sha256(b.raw).hexdigest())
    return dict(metadata=meta,fields=fields)

def run_first_query(lib,ctx,above,*,deadline_check):
    """Two original resets; identical exact original prefix chunk sequence.

    above must come from the pinned original freezer, not a synthetic hidden fixture.
    Caller retains actual rank/owner, plan/source/library identities and normal destroy.
    No continuation, publish, logits/head, generation or Model quality assertion.
    """
    if (above["geometry"]!="above"or len(above["prompt"])!=2949 or len(above["proposal"])!=8
        or above["accepted"]!=8 or sum(above["chunk_widths"])!=2949
        or any(type(w)is not int or not 1<=w<=512 for w in above["chunk_widths"])):raise ValueError("original above geometry differs")
    result={}
    for width in (1,8):
        deadline_check("reset");checked(lib,lib.glm_l0_reset(ctx));past=0
        for count in above["chunk_widths"]:
            values=above["prompt"][past:past+count];ids=(C.c_int64*count)(*values)
            deadline_check("prefix step");checked(lib,lib.glm_l0_step(ctx,past,count,ids,0));past+=count
        values=above["proposal"][:width];ids=(C.c_int64*width)(*values)
        deadline_check("first query");checked(lib,lib.glm_l0_step(ctx,past,width,ids,1));result[width]=read_snapshot(lib,ctx)
    checked(lib,lib.glm_l0_reset(ctx))
    return result


def read_original_above(path):
    import runpy
    module=runpy.run_path(str(Path(__file__).resolve().parent/"freeze_continuation_inputs.py"))
    raw=module["freeze"](Path(path).read_bytes())
    if sha256(raw).hexdigest()!="f71463189b3da010aabac90cdb28a14a0a232e58e22cad05dfa60b4c321ecac4":raise ValueError("frozen original continuation plan differs")
    rows=[c for c in json.loads(raw)["cases"]if c["geometry"]=="above"]
    if len(rows)!=1:raise ValueError("single original above case required")
    return rows[0]
