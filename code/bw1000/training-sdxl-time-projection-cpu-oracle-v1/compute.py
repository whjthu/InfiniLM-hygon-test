"""Compare SDXL projection rounding only after reproducing the two original operand hashes."""
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import signal
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open


BINDING = Path("/fix/INPUT_BINDING.json")
STYLES = ("single_round", "rounded_dot_then_bias")
PROJECTION = "down_blocks.0.resnets.0.time_emb_proj"
PREFIXES = ("time_embedding", "add_embedding", PROJECTION)


def require(test, label):
    if not test:
        raise ValueError(label)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    with path.open("rb") as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(value):
    require(value.device.type == "cpu", "CPU tensor only")
    return sha(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())


def rounded_affine(x, layer, style):
    require(style in STYLES and x.dtype == layer.weight.dtype == layer.bias.dtype == torch.bfloat16,
            "BF16 affine operands")
    product = x.double() @ layer.weight.double().T
    if style == STYLES[1]:
        product = product.to(torch.bfloat16).double()
    return (product + layer.bias.double()).to(torch.bfloat16)


def activate(x):
    return F.silu(x.double()).to(torch.bfloat16)


def options(module, x):
    require(module.cond_proj is None and module.post_act is None
            and type(module.act) is torch.nn.SiLU, "original two-affine SiLU topology")
    result = {"cpu_torch_native_bf16": module(x)}
    for a, b in itertools.product(STYLES, repeat=2):
        result[a + "/" + b] = rounded_affine(activate(rounded_affine(x, module.linear_1, a)), module.linear_2, b)
    return result


def load_modules(model, archive, binding):
    observed = {}
    elements_loaded = 0
    with safe_open(str(archive), framework="pt", device="cpu") as fp:
        for prefix in PREFIXES:
            module = model.get_submodule(prefix)
            state = {}
            for name, meta in module.state_dict().items():
                key = prefix + "." + name
                value = fp.get_tensor(key)
                require(value.shape == meta.shape and value.dtype == torch.float32, "original checkpoint range shape")
                actual_hash = tensor_sha(value)
                stored = binding["verified_projection_ranges"].get(key)
                if stored is not None:
                    require(stored["sha256"] == actual_hash and stored["shape"] == list(value.shape)
                            and stored["bytes"] == value.numel()*4, "original time projection range hash")
                observed[key] = actual_hash
                state[name] = value.to(torch.bfloat16)
                elements_loaded += value.numel()
            moved = module.load_state_dict(state, strict=True, assign=True)
            require(not moved.missing_keys and not moved.unexpected_keys, "original model module exact membership")
            module.requires_grad_(False)
    require(elements_loaded < 20_000_000 and set(binding["verified_projection_ranges"]) <= observed.keys(),
            "read only bounded original checkpoint ranges")
    return observed, elements_loaded


def check_env():
    epoch = float(os.environ["REFERENCE_READER_EPOCH"])
    deadline = float(os.environ["REFERENCE_READER_DEADLINE"])
    remain = deadline-time.time()
    require(deadline == epoch+870 and 0 < remain <= 870, "single original epoch")
    signal.setitimer(signal.ITIMER_REAL,remain)
    require(os.environ["CUDA_VISIBLE_DEVICES"] == "-1" and not torch.cuda.is_initialized(), "CPU only")
    def no_device(*args, **kwargs):
        raise RuntimeError("No device in CPU oracle")
    torch.cuda._lazy_init = no_device
    torch.set_num_threads(2)


def run():
    check_env()
    from diffusers import UNet2DConditionModel
    binding = json.loads(BINDING.read_bytes())
    config_path = Path("/model/unet/config.json")
    config = config_path.read_bytes()
    require(sha(config) == binding["model_config"], "original model config hash")
    archive = config_path.with_name("diffusion_pytorch_model.safetensors")
    with archive.open("rb") as f:
        header_length = f.read(8)
        header_size = int.from_bytes(header_length,"little")
        require(0 < header_size <= 2*1024**2,"original weight header budget")
        header = f.read(header_size)
    header_forms = {"body":sha(header), "with_length_prefix":sha(header_length+header)}
    matched_forms = [k for k,v in header_forms.items() if v==binding["archive_header_sha256"]]
    require(archive.stat().st_size == binding["archive_bytes"] and len(header)==header_size
            and len(matched_forms)==1, "original checkpoint header")
    ref = Path("/reference/reference.pt")
    require(ref.stat().st_size == binding["reference_bytes"] and sha_file(ref) == binding["reference_sha256"],
            "original realized input archive")
    with torch.device("meta"):
        model = UNet2DConditionModel.from_config(json.loads(config))
    require(all(p.is_meta for p in model.parameters()) and model.time_embed_act is None, "original model topology")
    manifest, elements = load_modules(model, archive, binding)
    realized = torch.load(ref, map_location="cpu", weights_only=True)["inputs"]
    expected = binding["expected_operand_sha256"]
    with torch.inference_mode():
        time_input = model.get_time_embed(sample=realized["noisy_model_input"],timestep=realized["timesteps"]).to(torch.bfloat16)
        conditions = realized["unet_added_conditions"]
        time_ids = conditions["time_ids"]
        text = conditions["text_embeds"]
        time_values = model.add_time_proj(time_ids.flatten()).reshape((text.shape[0],-1))
        add_input = torch.cat((text,time_values),dim=-1).to(torch.bfloat16)
        in_pins = {"main":tensor_sha(time_input),"addition":tensor_sha(add_input)}
        observed = dict(prefix_input_sha256=in_pins,
                        prefix_inputs_match_original={k:in_pins[k]==expected[k+"_in"] for k in in_pins})
        qualified = None
        branches = {}
        for label,module,x in (("main",model.time_embedding,time_input),("addition",model.add_embedding,add_input)):
            branch = options(module,x)
            fingerprints = {k:tensor_sha(v) for k,v in branch.items()}
            matches = [k for k,h in fingerprints.items() if h==expected[label+"_out"]]
            observed[label] = dict(method_sha256=fingerprints,original_matches=matches)
            if matches and observed["prefix_inputs_match_original"][label]:
                branches[label] = branch[matches[0]]
        if set(branches) == {"main","addition"}:
            fused = branches["main"]+branches["addition"]
            need_operand = expected["projection_in"]
            if tensor_sha(fused) == expected["fused"]:
                act_candidates = {"cpu_torch_silu":F.silu(fused), "cpu_f64_silu":activate(fused)}
                observed["silu_sha256"] = {k:tensor_sha(v) for k,v in act_candidates.items()}
                observed["fused_sha256"] = tensor_sha(fused)
                qualified = next((v for k,v in act_candidates.items() if tensor_sha(v) == need_operand),None)
        result = dict(observations=observed,matched_original_two_platform_operand=qualified is not None,
                      archive_header_hash_scope=matched_forms[0],
                      materialized_weight_elements=elements,checkpoint_range_sha256=manifest,
                      versions={k:importlib.metadata.version(k) for k in ("torch","diffusers","safetensors")},
                      actual_gpu=0,full_unet_forwards=0,optimizer_updates=0,raw_operand_transfer_bytes=0)
        if qualified is not None:
            p = model.get_submodule(PROJECTION)
            values = {style:rounded_affine(qualified,p,style) for style in STYLES}
            values["cpu_torch_native_bf16"] = p(qualified)
            pins = {k:tensor_sha(v) for k,v in values.items()}
            truth = binding["original_projection_output_sha256"]
            result["projection"] = dict(method_sha256=pins,
                exact_original_outputs={k:{side:h==right for side,right in truth.items()} for k,h in pins.items()},
                source_prefix_operand_shape=list(qualified.shape),target_bfloat16_shape=list(p.bias.shape),
                different_elements_between_bias_orders=int(torch.count_nonzero(values[STYLES[0]]!=values[STYLES[1]]).item()))
    require(not torch.cuda.is_initialized(),"no GPU in CPU oracle")
    with Path("/evidence/compute.json").open("x",encoding="utf8") as f:
        json.dump(result,f,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    run()
