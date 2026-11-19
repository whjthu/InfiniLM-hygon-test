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
ORDER = (False, True, True, False, True, False, False, True) * 3


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
        for name, module in (
            ("patch_conv3d",model.patch_embed.proj), ("patch_output",model.patch_embed),
            ("position_embedding",model.pos_embed), ("rotary_output",model.rotary_pos_emb),
            ("block_00_norm1",model.blocks[0].norm1), ("block_00_qkv",model.blocks[0].attn.qkv),
            ("block_00_attention",model.blocks[0].attn), ("block_00_norm2",model.blocks[0].norm2),
            ("block_00_mlp",model.blocks[0].mlp), ("vision_merger",model.merger)
        ):
            self.output(name,module)
        for i, block in enumerate(model.blocks):
            self.output(f"block_{i:02d}_output",block)
        for i in (0,20):
            block=model.blocks[i]
            pre=f"block_{i:02d}"
            self.handles.append(block.mlp.register_forward_pre_hook(self.mlp_operand(pre)))
            for suffix,part in (("mlp_fc1",block.mlp.linear_fc1),("mlp_activation",block.mlp.act_fn),
                                ("mlp_fc2",block.mlp.linear_fc2)):
                self.output(pre+"_"+suffix,part)
        for suffix,part in (("norm1",model.blocks[20].norm1),("attention",model.blocks[20].attn),
                            ("norm2",model.blocks[20].norm2),("mlp",model.blocks[20].mlp)):
            self.output("block_20_"+suffix,part)
        for i, merger in enumerate(model.deepstack_merger_list):
            self.output(f"deepstack_merger_{i}",merger)
        self.handles.append(model.blocks[0].register_forward_pre_hook(self.first_input,with_kwargs=True))

    def mlp_operand(self,prefix):
        def observe(_m,args):
            check(len(args)==1,"ONE_NATIVE_MLP_OPERAND")
            self.mark(prefix+"_mlp_entry",args[0])
        return observe

    def first_input(self, _block, args, kwargs):
        self.mark("post_position_add",args[0])
        self.mark("first_block_cu_seqlens",kwargs["cu_seqlens"])
        cosine,sine=kwargs["position_embeddings"]
        self.mark("first_block_cosine",cosine)
        self.mark("first_block_sine",sine)

    def mark(self,name,value):
        check(name not in self.rows and isinstance(value,torch.Tensor), "NATIVE_BOUNDARY_SCHEMA")
        self.retained_bytes += value.numel()*value.element_size()
        check(self.retained_bytes <= 256<<20, "BOUNDARY_MEMORY_BUDGET")
        self.rows[name]=(value,value._version)

    def output(self,name,module):
        def hook(_module,_args,result):
            self.mark(name,result)
        self.handles.append(module.register_forward_hook(hook))

    def finish(self):
        check(len(self.rows)==56, "EXACT_VISION_PATH_COVERAGE")
        result={}
        for name,(t,version) in self.rows.items():
            check(t._version==version, "OBSERVED_BOUNDARY_MUTATED")
            result[name]=tensor_note(t)
        self.rows.clear()
        self.retained_bytes=0
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.rows.clear()


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


class BlockBarriers:
    def __init__(self, model):
        self.active=False
        self.count=0
        self.hooks=[block.register_forward_hook(self.after) for block in model.blocks]

    def after(self,_module,_args,_result):
        if self.active:
            torch.cuda.current_stream().synchronize()
            self.count+=1

    def close(self):
        for h in self.hooks:
            h.remove()


def delta(current, baseline):
    check(current.shape==baseline.shape and current.dtype==baseline.dtype,"SAME_OUTPUT_LAYOUT")
    a=current.double()
    b=baseline.double()
    diff=a-b
    norm=float(b.square().sum().item())
    l2=float(diff.square().sum().item())
    return dict(different=int((diff!=0).sum().item()),max_abs=float(diff.abs().max().item()),
                relative_L2=(l2/norm)**.5 if norm else None)


def evaluate(model, observer, barriers, rank, pin, deadline):
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
    endpoints={}
    for sync_blocks in ORDER:
        check(time.time()<deadline,"ORIGINAL_EPOCH")
        torch.set_rng_state(rng["torch_CPU"])
        torch.cuda.set_rng_state(rng["torch_device"],0)
        check(torch.equal(torch.get_rng_state(),rng["torch_CPU"]) and
              torch.equal(torch.cuda.get_rng_state(0),rng["torch_device"]),"FROZEN_RNG_ENTRY")
        call_image=img.detach().requires_grad_(True)
        stream=torch.cuda.current_stream()
        stream.synchronize()
        barriers.active=sync_blocks
        barriers.count=0
        start=time.monotonic()
        with torch.enable_grad(), torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            out=model(call_image, grid_thw=device_grid,return_dict=True,**extra)
        stream.synchronize()
        forward_seconds=time.monotonic()-start
        expected=27 if sync_blocks else 0
        check(barriers.count==expected,"EXACT_BLOCK_BARRIER_COUNT")
        check(len(out.deepstack_features)==3 and out.pooler_output.requires_grad and
              all(t.requires_grad for t in out.deepstack_features),"PRESERVE_GRAPH_CONNECTED_OUTPUTS")
        trace=observer.finish()
        final={"final_pooler":out.pooler_output}
        final.update({f"final_deepstack_{i}":t for i,t in enumerate(out.deepstack_features)})
        host={key:t.detach().cpu().contiguous() for key,t in final.items()}
        for key,t in final.items():
            trace[key]=tensor_note(t)
        same_arm=endpoints.setdefault(sync_blocks,host)
        default=endpoints[False]
        check(same_bytes(tensor_note(img),inputs["effective_pixel"]),"ORIGINAL_INPUT_MUTATION")
        runs.append(dict(sync_blocks=sync_blocks,block_barrier_count=barriers.count,
            forward_seconds=forward_seconds,trace=trace,
            delta_vs_first_same_arm={key:delta(value,same_arm[key]) for key,value in host.items()},
            delta_vs_first_default={key:delta(value,default[key]) for key,value in host.items()}))
        del out, call_image,final
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
    barriers=BlockBarriers(model)
    initial_versions={name:t._version for name,t in model.named_parameters()}
    try:
        ranks=[evaluate(model,observer,barriers,r,pin["ranks"][str(r)],deadline) for r in RANKS]
        check({name:t._version for name,t in model.named_parameters()}==initial_versions,
              "FROZEN_WEIGHT_MUTATION")
        torch.cuda.synchronize()
        result=dict(passed=True,identity=actual,ranks=ranks,vision_load_seconds=load_seconds,
            actual_forwards=len(ranks)*len(ORDER),vision_provider=pin["provider_SHA256"],
            frozen_visual_weights=351,vision_weight_bytes=1152776672,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),actual_backward=0,actual_updates=0,
            full_model=False,formal_training=False)
    finally:
        barriers.close()
        observer.close()
    with (ROOT/"result.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,allow_nan=False,separators=(",",":"))


if __name__=="__main__":
    main()
