"""Observe the original A100 TRELLIS first backward at four LoRA projections."""

import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import torch

from lora_projection_reference import capture_lora, measure, qualify_cpu_vjp, SITES, weight_vjp_cpu

BASE = Path("/home/whj/nvidia-baseline")
ROOT = Path("/mnt/data/nvidia-baseline-20261010")
BACKEND_SHA = "bef3cb4fa61e8f5631da7a0134ddd27c18ca5d5f5f945a8c17a201ed3dd19973"
REFERENCE_SHA = "c164b02f2bd4edea0d266daebf7d93adef17e753589d95fb530f81b1fba9e612"
LAUNCH_SHA = "d5b23154d64af63332e54e079d6b0fda0a2904f53bdd0e6a0541fa265fad29fa"


class CaptureComplete(Exception):
    pass


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def tensor_identity(value):
    value = value.detach().cpu().contiguous()
    return {"shape":list(value.shape),"dtype":str(value.dtype),
            "SHA256":hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
            "l2":torch.linalg.vector_norm(value.double()).item()}


def exact(actual, expected):
    if isinstance(expected,torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual.detach().cpu(),expected)
    elif isinstance(expected,dict):
        assert list(actual)==list(expected)
        for key,value in expected.items():
            exact(actual[key],value)
    elif isinstance(expected,(list,tuple)):
        assert type(actual) is type(expected) and len(actual)==len(expected)
        for left,right in zip(actual,expected,strict=True):
            exact(left,right)
    else:
        assert actual==expected


def main():
    output = Path(sys.argv[1])
    assert not output.exists()
    assert torch.__version__ == "2.8.0a0+5228986c39.nv25.06" and torch.version.cuda == "12.9"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0,str(ROOT/"cuda_source_r3/generation/src"))
    from gencuda.backends import trellis2_training as backend
    source = Path(inspect.getsourcefile(backend))
    launch_path = BASE/"results/trellis-cuda-shape-baseline-r1-repeat1/launch.json"
    reference_path = BASE/"alignment-assets/a100-alignment-v1/trellis-shape-first-step-r1/reference.pt"
    assert file_sha(source)==BACKEND_SHA and file_sha(launch_path)==LAUNCH_SHA
    assert file_sha(reference_path)==REFERENCE_SHA
    reference = torch.load(reference_path,map_location="cpu",weights_only=True)
    launch = json.loads(launch_path.read_bytes())
    lines = source.read_text().splitlines()
    context = None
    captured = None
    qualified = False
    observed = None
    initial = None

    def trace(frame,event,arg):
        nonlocal context,captured,qualified,observed,initial
        if event!="line" or frame.f_code.co_filename!=str(source) or frame.f_code.co_name!="run_trellis2_training":
            return trace
        values = frame.f_locals
        if values.get("step")!=1:
            return trace
        line = lines[frame.f_lineno-1].strip()
        if line=='with torch.autocast(device_type="cuda", dtype=torch.bfloat16):' and context is None:
            parameters = values["parameters"]
            initial = {name:p.detach().cpu().clone() for name,p in parameters.items()}
            exact(initial,reference["initial_adapter"])
            exact(values["optimizer"].state_dict(),reference["initial_optimizer"])
            exact({key:values[key] for key in reference["inputs"]},reference["inputs"])
            context = capture_lora(values["model"])
            captured = context.__enter__()
        elif line=="loss.backward()":
            exact(values["prediction"].feats,reference["prediction"])
            exact(values["prediction"].coords,reference["prediction_coords"])
            exact(values["loss"],reference["loss"])
        elif line=="grads = preclip_gradients(parameters)" and captured is not None:
            context.__exit__(None,None,None)
            context = None
            parameters=values["parameters"]
            observed={name:p.grad.detach().cpu().clone() for name,p in parameters.items()}
            exact(observed,reference["gradients_before_clip"])
            exact({name:p.detach().cpu() for name,p in parameters.items()},initial)
            exact(values["optimizer"].state_dict(),reference["initial_optimizer"])
            qualified = True
            raise CaptureComplete()
        return trace

    qualify_cpu_vjp()
    output.mkdir()
    sys.settrace(trace)
    try:
        backend.run_trellis2_training(Path(launch["weights"]),launch["config"],output/"native-intermediate",Path(launch["source"]))
    except CaptureComplete:
        pass
    finally:
        sys.settrace(None)
        if context is not None:
            context.__exit__(None,None,None)
    assert qualified and set(captured)==set(SITES) and observed is not None
    rows,payload = {},{}
    for site in SITES:
        row=captured[site]
        value=row["input"].cpu()
        upstream=row["upstream"].cpu()
        grad=observed[site+".lora_B"]
        oracle=weight_vjp_cpu(value.reshape(-1,2),upstream.reshape(-1,upstream.shape[-1]))
        rows[site]={"input":tensor_identity(value),"upstream":tensor_identity(upstream),
                    "native_gradient":tensor_identity(grad),
                    "native_vs_cpu_f64_rounded_bf16":measure(grad,oracle.to(torch.bfloat16).float()),
                    "input_visible_shape":row["input_shape"]}
        payload[site]={"input":value,"upstream":upstream,"native_gradient":grad}
    path=output/"original-a100-lora-operands.pt"
    torch.save(payload,path)
    assert file_sha(source)==BACKEND_SHA and file_sha(reference_path)==REFERENCE_SHA
    result={"state":"CAPTURED_ORIGINAL_A100_LORA_OPERANDS","GPU_optimizer_updates":0,
            "physical_card":os.environ["CUDA_VISIBLE_DEVICES"],"torch":str(torch.__version__),"cuda":torch.version.cuda,
            "backend_SHA256":BACKEND_SHA,"original_reference_SHA256":REFERENCE_SHA,"launch_SHA256":LAUNCH_SHA,
            "original_inputs_initial_state_prediction_loss_gradients_exact":True,
            "private_operands_SHA256":file_sha(path),"private_operands_bytes":path.stat().st_size,
            "sites":rows,"peak_reserved_bytes":torch.cuda.max_memory_reserved()}
    with (output/"receipt.json").open("x") as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=="__main__":
    main()
