"""Qualify Qwen tokenizer input and the original HYTextModel crop calculation."""

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
from typing import Union


def source_methods(text_encoder, model_constants):
    constants = ast.parse(model_constants)
    prompt, = [ast.literal_eval(node.value) for node in constants.body if isinstance(node, ast.Assign)
               and [getattr(t, "id", None) for t in node.targets] == ["PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION"]]
    source = ast.parse(text_encoder)
    owner, = [node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "HYTextModel"]
    selected = []
    for name in ("apply_text_to_template", "_compute_crop_start"):
        method, = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name]
        method.decorator_list = []
        selected.append(method)
    template = [{"role": "system", "content": prompt}, {"role": "user", "content": "{}"}]
    namespace = {"Union": Union, "LLM_ENCODER_LAYOUT": {"qwen3": {"template": template}}}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<pinned-hymotion-methods>", "exec"), namespace)
    return namespace["apply_text_to_template"], namespace["_compute_crop_start"], template


def hash_ids(values):
    return sha256(b"".join(int(x).to_bytes(8, "little", signed=True) for x in values)).hexdigest()


def verify(path, pin):
    with path.open("rb") as stream:
        raw = stream.read(pin["bytes"] + 1)
    if len(raw) != pin["bytes"] or sha256(raw).hexdigest() != pin["SHA256"]:
        raise ValueError(f"target input differs: {path.name}")
    return raw


def main():
    stage, model = Path("/input-stage"), Path("/input-qwen")
    claim = json.loads((stage / "CLAIM.json").read_bytes())
    node = socket.gethostname().split(".")[0]
    job, number = os.environ["SLURM_JOB_ID"], os.environ["SLURM_STEP_ID"]
    if (job != claim["parent"] or node != claim["node"] or not number.isdigit()
            or os.environ.get("SLURM_STEP_GPUS") or os.environ["SLURM_CPUS_PER_TASK"] != "4"
            or any(os.environ.get(k) != "-1" for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))):
        raise ValueError("CPU owner or visibility differs")
    for name, pin in claim["qwen_files"].items():
        verify(model / name, pin)
    if (model / "chat_template.jinja").exists():
        raise ValueError("unbound standalone chat template")
    upstream = [verify(stage / name, claim["payload_files"][name]) for name in ("text_encoder.py", "model_constants.py")]
    apply_template, compute_crop, template = source_methods(*upstream)
    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model), padding_side="right", local_files_only=True, trust_remote_code=False
    )
    if torch.cuda.is_initialized():
        raise ValueError("GPU runtime was initialized")
    # The upstream crop helper checks the model field but only calls the tokenizer.
    context = SimpleNamespace(llm_type="qwen3", llm_text_encoder=object(), llm_tokenizer=tokenizer,
                              apply_text_to_template=apply_template)
    crop = compute_crop(context)
    width = 128 + crop
    if not 0 < crop < 512:
        raise ValueError("unexpected native crop length")
    rendered = [tokenizer.apply_chat_template(apply_template(row["prompt"], template), tokenize=False,
                                               add_generation_prompt=False, enable_thinking=False)
                for row in claim["inputs"]]
    enc = tokenizer(rendered, return_length=False, return_overflowing_tokens=False, truncation=True,
                    return_attention_mask=True, max_length=width, padding="max_length", return_tensors="pt")
    rows = []
    for record, text, ids_tensor, mask_tensor in zip(claim["inputs"], rendered, enc["input_ids"], enc["attention_mask"]):
        ids, mask = ids_tensor.tolist(), mask_tensor.tolist()
        n = sum(mask)
        if not crop < n < width or len(ids) != width or mask != [1] * n + [0] * (width - n):
            raise ValueError("unexpected untruncated right-padding or mask boundary")
        rows.append(dict(case=record["case"], prompt_SHA256=sha256(record["prompt"].encode()).hexdigest(),
                         rendered_SHA256=sha256(text.encode()).hexdigest(), ids_SHA256=hash_ids(ids), mask_SHA256=hash_ids(mask),
                         rendered_nonpadding_tokens=n, context_length=min(128, n - crop), padded_width=width))
    if len(rows) != len(claim["inputs"]) or len({row["ids_SHA256"] for row in rows}) != len(rows):
        raise ValueError("tokenizer evidence lacks independent motion text variants")
    result = dict(passed=True, actual_step=job + "." + number, node=node, crop_start=crop,
                  output_context_width=128, rows=rows, tokenizer=type(tokenizer).__name__, transformers=transformers.__version__,
                  torch=torch.__version__, encoder_calls=0, model_weight_bytes_read=0, accepted_training_rows=0,
                  GPU_cards=0, CUDA_initialized=torch.cuda.is_initialized(), training_READY=False)
    with Path("/evidence/CPU-result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
