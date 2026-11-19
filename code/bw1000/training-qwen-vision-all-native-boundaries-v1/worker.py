"""Replay the frozen original Qwen vision encoder on one physical accelerator."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import time

import torch
from safetensors import safe_open
from transformers.models.qwen3_vl import modeling_qwen3_vl as vision
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig


ROOT = Path("/operator")
MODEL = Path("/model")
CAPTURE = Path("/capture")
RANKS = (1,)
REPEATS = 8


def check(condition, code):
    if not condition:
        raise ValueError(code)


def sha(path, pin, deadline):
    start = path.stat()
    check(start.st_size == pin["bytes"], "FROZEN_INPUT_BYTES")
    out = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 << 20), b""):
            check(time.time() < deadline, "ORIGINAL_EPOCH")
            out.update(part)
    end = path.stat()
    identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
    check(identity(start) == identity(end) and out.hexdigest() == pin["SHA256"], "FROZEN_INPUT_SHA")
    return out.hexdigest()


def device_identity():
    lease = json.loads((ROOT / "lease-before.json").read_bytes())
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    check(step == lease["step"] and lease["kfd_holder_count"] == 0, "EXACT_DEVICE_LEASE")
    check(torch.version.hip == "6.3.26113" and torch.cuda.device_count() == 1, "ONE_TARGET_HIP_DEVICE")
    lib = ctypes.CDLL("libamdhip64.so")
    lib.hipDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    buf = ctypes.create_string_buffer(32)
    check(lib.hipDeviceGetPCIBusId(buf, len(buf), 0) == 0, "HIP_PCI")
    pci = buf.value.decode().lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci):
        pci = "0000:" + pci
    check(pci in lease["pci"], "HIP_PCI_NOT_IN_HOST_LEASE")
    return dict(step=step, node=lease["node"], PCI=pci, runtime=str(torch.__version__),
                hip=torch.version.hip, name=torch.cuda.get_device_name(0))


def tensor_note(value):
    check(isinstance(value, torch.Tensor), "NATIVE_BOUNDARY_TENSOR")
    copied = value.detach().cpu().contiguous()
    check(not copied.is_floating_point() or bool(torch.isfinite(copied).all().item()),
          "NONFINITE_NATIVE_BOUNDARY")
    raw = copied.reshape(-1).view(torch.uint8).numpy()
    return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                SHA256=hashlib.sha256(memoryview(raw)).hexdigest(), requires_grad=value.requires_grad)


class Observations:
    def __init__(self, model):
        self.rows = {}
        self.retained_bytes = 0
        self.handles = []
        self.baseline = {}
        self.notes = {}
        for name, module in (("patch_conv3d", model.patch_embed.proj), ("patch_output", model.patch_embed),
                             ("position_embedding", model.pos_embed), ("rotary_output", model.rotary_pos_emb)):
            self.output(name,module)
        self.handles.append(model.blocks[0].register_forward_pre_hook(self.first_input,with_kwargs=True))
        for i, block in enumerate(model.blocks):
            name=f"block_{i:02d}_"
            for edge,module in (("entry",block),("attention_proj_entry",block.attn.proj),("after_attention_add",block.norm2)):
                self.entry(name+edge,module)
            for edge,module in (("norm1",block.norm1),("attention_qkv",block.attn.qkv),
                                ("attention_proj",block.attn.proj),("norm2",block.norm2),
                                ("mlp_fc1",block.mlp.linear_fc1),("mlp_activation",block.mlp.act_fn),
                                ("mlp_fc2",block.mlp.linear_fc2),("output",block)):
                self.output(name+edge,module)
        for i, merger in enumerate(model.deepstack_merger_list):
            self.output(f"deepstack_merger_{i}",merger)
        self.output("vision_merger",model.merger)

    def first_input(self,_block,_args,kwargs):
        self.mark("cu_seqlens",kwargs["cu_seqlens"])
        cosine,sine=kwargs["position_embeddings"]
        self.mark("position_cosine",cosine)
        self.mark("position_sine",sine)

    def mark(self,name,value):
        check(name not in self.rows and isinstance(value,torch.Tensor),"NATIVE_BOUNDARY_SCHEMA")
        self.retained_bytes+=value.numel()*value.element_size()
        check(self.retained_bytes<=1280<<20,"BOUNDARY_MEMORY_BUDGET")
        self.rows[name]=(value,value._version)

    def entry(self,name,module):
        def hook(_module,args):
            check(bool(args),"NATIVE_OPERAND_PRESENT")
            self.mark(name,args[0])
        self.handles.append(module.register_forward_pre_hook(hook))

    def output(self,name,module):
        def hook(_module,_args,result):
            self.mark(name,result)
        self.handles.append(module.register_forward_hook(hook))

    def finish(self):
        check(len(self.rows)==308,"ALL27_NATIVE_BOUNDARY_COVERAGE")
        result={}
        first=None
        anchor=not self.baseline
        for name,(t,version) in self.rows.items():
            check(t._version==version,"OBSERVED_BOUNDARY_MUTATED")
            value=t.detach().cpu().contiguous()
            note=tensor_note(value)
            note.update(requires_grad=t.requires_grad,stride=list(t.stride()))
            result[name]=note
            if anchor:
                self.baseline[name]=value
                self.notes[name]=note
            elif first is None and not same_bytes(note,self.notes[name]):
                original=self.baseline[name].double()
                diff=value.double()-original
                norm=float(original.square().sum().item())
                distance=float(diff.square().sum().item())
                first=dict(boundary=name,shape=note["shape"],dtype=note["dtype"],
                    changed_elements=int((diff!=0).sum().item()),
                    max_abs=float(diff.abs().max().item()),relative_L2=(distance/norm)**.5 if norm else None)
        self.rows.clear()
        self.retained_bytes=0
        return result,first

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.rows.clear()
        self.baseline.clear()
        self.notes.clear()


def load_model(pin, deadline):
    config_path=MODEL/"config.json"
    index_path=MODEL/"model.safetensors.index.json"
    shard_path=MODEL/pin["shard_name"]
    for path, p in [(config_path,pin["config"]), (index_path,pin["index"]), (shard_path,pin["shard"])]:
        sha(path,p,deadline)
    cfg=Qwen3VLVisionConfig(**json.loads(config_path.read_bytes())["vision_config"])
    cfg._attn_implementation="sdpa"
    check(cfg.depth==27 and cfg.hidden_size==1152 and cfg.deepstack_visual_indexes==[8,16,24],
          "ORIGINAL_VISION_GEOMETRY")
    camera=vision.Qwen3VLVisionModel(cfg)
    weights=json.loads(index_path.read_bytes())["weight_map"]
    names={key[len("model.visual."):]:key for key,shard in weights.items()
           if key.startswith("model.visual.") and shard==shard_path.name}
    check(len(names)==351 and set(names)==set(camera.state_dict()) and
          sum(k.startswith("model.visual.") for k in weights)==351, "EXACT_ORIGINAL_VISION_WEIGHTS")
    with safe_open(str(shard_path),framework="pt",device="cpu") as data:
        state={name:data.get_tensor(key) for name,key in names.items()}
        check(all(t.dtype==torch.bfloat16 for t in state.values()) and
              sum(t.numel()*t.element_size() for t in state.values())==1152776672,"VISION_WEIGHT_VALUES_LAYOUT")
        camera.load_state_dict(state,strict=True,assign=True)
    camera.requires_grad_(False)
    camera.to(device="cuda:0")
    camera.train()
    check(camera.config._attn_implementation=="sdpa" and len(camera.blocks)==27 and len(camera.deepstack_merger_list)==3,
          "ORIGINAL_PROVIDER_AND_LAYERS")
    check(camera.rotary_pos_emb.inv_freq.dtype==torch.float32,"NATIVE_ROTARY_BUFFER_DTYPE")
    return camera


def same_bytes(first,second):
    return (first["SHA256"],first["shape"],first["dtype"]) == (second["SHA256"],second["shape"],second["dtype"])


def evaluate(model, observer, rank, pin, deadline):
    folder=CAPTURE/f"rank-{rank:02d}"
    path=folder/"forward-01-input.pt"
    sha(folder/"receipt.json",pin["receipt"],deadline)
    sha(path,pin["files"]["forward-01-input.pt"],deadline)
    archive=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    check(archive["global_step"]==0 and archive["micro_ordinal"]==1,"ORIGINAL_FIRST_TRAINING_MICROBATCH")
    call=archive["kwargs"]
    check(isinstance(call,dict) and not archive["args"] and call.get("pixel_values_videos") is None,
          "ORIGINAL_IMAGE_ONLY_ARGUMENT_BOUNDARY")
    pixels=call["pixel_values"]
    grid=call["image_grid_thw"]
    check(pixels.ndim==2 and pixels.shape[1]==1536 and pixels.is_floating_point() and
          grid.ndim==2 and grid.shape[1]==3 and grid.dtype==torch.long and
          int(grid.prod(-1).sum().item())==pixels.shape[0],"ORIGINAL_PATCH_GRID")
    check(torch.isfinite(pixels).all().item() and bool((grid>0).all().item()),"FINITE_ORIGINAL_IMAGES")
    known=("bilinear_indices","bilinear_weights","cu_seqlens")
    extra={k:call[k].to("cuda:0") for k in known if isinstance(call.get(k),torch.Tensor)}
    check(all(call.get(k) is None or k in extra for k in known),"REAL_VISION_METADATA_TYPES")
    img=pixels.to(device="cuda:0",dtype=model.dtype)
    device_grid=grid.to("cuda:0")
    rng=archive["RNG_before_forward"]
    inputs=dict(pixel=tensor_note(pixels),grid=tensor_note(grid),
                effective_pixel=tensor_note(img),vision_extra_names=sorted(extra))
    runs=[]
    for _ in range(REPEATS):
        connected=True
        check(time.time()<deadline,"ORIGINAL_EPOCH")
        torch.set_rng_state(rng["torch_CPU"])
        torch.cuda.set_rng_state(rng["torch_device"],0)
        call_image=img.detach().requires_grad_(connected)
        with torch.enable_grad(), torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            out=model(call_image, grid_thw=device_grid,return_dict=True,**extra)
        check(len(out.deepstack_features)==3 and out.pooler_output.requires_grad==connected and
              all(t.requires_grad==connected for t in out.deepstack_features),"BOTH_CONSUMED_VISION_FIELD_GRAD_FLAGS")
        trace,first=observer.finish()
        trace["final_pooler"]=tensor_note(out.pooler_output)
        for i,t in enumerate(out.deepstack_features):
            trace[f"final_deepstack_{i}"]=tensor_note(t)
        check(same_bytes(tensor_note(img),inputs["effective_pixel"]),"ORIGINAL_INPUT_MUTATION")
        runs.append(dict(connected=connected,trace=trace,first_difference=first))
        del out, call_image
    return dict(rank=rank,inputs=inputs,runs=runs)


def main():
    intent=json.loads((ROOT/"intent.json").read_bytes())
    deadline=intent["epoch"]+600
    pin=json.loads((ROOT/"INPUTS.json").read_bytes())
    check(hashlib.sha256(Path(vision.__file__).read_bytes()).hexdigest()==pin["provider_SHA256"],
          "FROZEN_VISION_PROVIDER")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    actual=device_identity()
    t0=time.monotonic()
    model=load_model(pin,deadline)
    load_seconds=time.monotonic()-t0
    observer=Observations(model)
    initial_versions={name:t._version for name,t in model.named_parameters()}
    try:
        ranks=[evaluate(model,observer,r,pin["ranks"][str(r)],deadline) for r in RANKS]
        check({name:t._version for name,t in model.named_parameters()}==initial_versions,
              "FROZEN_WEIGHT_MUTATION")
        torch.cuda.synchronize()
        result=dict(passed=True,identity=actual,ranks=ranks,vision_load_seconds=load_seconds,
            actual_forwards=len(ranks)*REPEATS,vision_provider=pin["provider_SHA256"],
            frozen_visual_weights=351,vision_weight_bytes=1152776672,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),actual_backward=0,actual_updates=0,
            full_model=False,formal_training=False)
    finally:
        observer.close()
    with (ROOT/"result.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,allow_nan=False,separators=(",",":"))


if __name__=="__main__":
    main()
