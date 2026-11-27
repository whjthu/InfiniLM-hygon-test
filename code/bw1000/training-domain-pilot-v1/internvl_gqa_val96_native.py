"""Run an assigned share of the paired GQA work in one fresh Swift process per DCU."""

import argparse
from collections import Counter
import ctypes
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import random
import re
import time

import internvl_gqa_val96 as experiment
from internvl_gqa_val96 import need
from internvl_gqa_miopen_home import verify as verify_miopen_home


INPUT_SHA = "98f729fce9e6af91edb10bbf4dcc9b0d06f1322a38247c7001d05b02a21fc46b"
PROCESSED_INPUT_SHA = "0b4514748751d2544dfc30993e0f37747b42f1a59b0aab0d56a11e78b617187d"
MODEL = Path("/InternVL3_5-8B-HF")
CHECKPOINT = Path("/gqa-domain25")
ADAPTER_NAME = "internvl_gqa_domain25"
ADAPTER_PINS = {
    "adapter_config.json": (1080, "35e6ca230c487e51b45f16b96fc8f4b76393d7130f981b67f0f30529a1c6374f"),
    "adapter_model.safetensors": (174663096, "fbd97669eb0040e580e05b3d4286b57988f88effa0f196f96b10bb27a0a89d73"),
}
VERSIONS = {"torch": "2.10.0+das.opt1.dtk2604.2606161421.g91a580", "transformers": "5.5.0",
            "ms-swift": "4.4.2", "peft": "0.19.1", "accelerate": "1.14.0", "tokenizers": "0.22.2"}
ENGINE_PINS = {
    "infer_engine/transformers_engine.py": "b1c229267347569f63bc2afd0e263c2d0d919535e596db454e05b8f825493dbb",
    "infer_engine/infer_engine.py": "857f18c2b25a1e6787679a3f96b8fa96f5de7efd040e6933bb62e0a798634969",
    "infer_engine/utils.py": "5aa9dc2376402d4d6a0167c5f1253a6cd085e10b157a0f3f5a7b72c0f8607644",
    "infer_engine/protocol.py": "f7abe5f7fb9ed9cf390d54a3303d28d6c68c28679ab304dc461f645bbd5aac6e",
}


def stats(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def pinned_adapter(deadline):
    before = {}
    for name, (extent, pin) in ADAPTER_PINS.items():
        path = CHECKPOINT / name
        need(path.is_file() and not path.is_symlink(), "Canonical adapter member required")
        before[name] = stats(path)
        h = sha256()
        total = 0
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(1 << 20), b""):
                deadline()
                total += len(data)
                need(total <= extent, "Adapter member exceeded qualified size")
                h.update(data)
        need((total, h.hexdigest()) == (extent, pin) and stats(path) == before[name], "Adapter bytes differ")
    return before


def record(slot, status, pin=None, text=None, tokens=None):
    return {"slot": slot, "status": status, "text": text, "processed_input_SHA256": pin, "generated_tokens": tokens}


def user_input(prompt, root):
    return {"messages": [{"role": "user", "content": "<image>\n" + prompt["question"]}],
            "images": [str(root / prompt["image"])]}


def native_input_pin(value, fingerprint):
    need(value.get("labels") is None and 0 < value["input_ids"].shape[-1] <= 4096
         and value.get("pixel_values") is not None, "Native user-only image input required")
    traces = {k: {name: pin for name, pin in fingerprint(v).items() if name != "floating_l2"}
              for k, v in value.items()}
    return experiment.digest(experiment.encode(traces))


def prior_input_proof(path, prompts):
    proof = json.loads(experiment.read(path, 1 << 17, PROCESSED_INPUT_SHA))
    need(proof["input_SHA256"] == INPUT_SHA and proof["step"] == "1103606.4392"
         and proof["phase"] == "CPU_NATIVE_USER_ONLY_INPUTS"
         and proof["input_fields"] == ["attention_mask", "input_ids", "pixel_values"]
         and all(proof[key] is False for key in ("model_weights_loaded", "GPU_initialized", "gold_read"))
         and len(proof["rows"]) == len(prompts) == experiment.COUNT, "Exact native CPU pilot required")
    expected = {}
    for row, prompt in zip(proof["rows"], prompts):
        need((row["slot"], row["kind"]) == (prompt["slot"], prompt["type"])
             and experiment.HEX.fullmatch(row["processed_SHA256"])
             and type(row["input_tokens"]) is int and 0 < row["input_tokens"] <= 4096,
             "Native CPU input records differ from frozen prompt identities")
        expected[row["slot"]] = (row["processed_SHA256"], row["input_tokens"])
    return expected


def select_generation(raw, eos, pad):
    need(len(raw) <= experiment.CAP and all(type(v) is int and v >= 0 for v in raw),
         "Invalid native generated IDs")
    effective = tuple(raw)
    if not effective:
        return None, 0
    for index, token in enumerate(effective):
        if token in eos:
            need(all(t == pad for t in effective[index + 1:]), "Non-padding token followed generation EOS")
            return "eos", index + 1
    return ("cap", len(effective)) if len(effective) == experiment.CAP else (None, len(effective))


def rank_slots(rank, world):
    need(world == 8 and 0 <= rank < world and experiment.COUNT % world == 0, "Original eight-card pairing differs")
    return range(rank + 1, experiment.COUNT + 1, world)


def physical_pci(device):
    library = ctypes.CDLL("libamdhip64.so")
    address = ctypes.create_string_buffer(32)
    need(library.hipDeviceGetPCIBusId(address, len(address), device) == 0, "HIP device PCI query failed")
    value = address.value.decode("ascii").lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value):
        value = "0000:" + value
    need(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value), "Invalid HIP PCI")
    return value


def run(args):
    def deadline():
        need(args.epoch <= time.time() < args.deadline, "Original evaluation deadline exhausted")

    deadline()
    need(re.fullmatch(r"[0-9a-f]{32}", args.request) and re.fullmatch(r"[0-9]+\.[0-9]+", args.step)
         and args.step == os.environ.get("SLURM_JOB_ID", "") + "." + os.environ.get("SLURM_STEP_ID", "")
         and 0 < args.deadline - args.epoch <= 3300 and args.input.is_absolute() and args.output.is_absolute(),
         "Original admitted step identity required")
    slots = rank_slots(args.rank, args.world)
    all_prompts = experiment.inputs(args.input, INPUT_SHA)
    frozen_native = prior_input_proof(args.input_proof, all_prompts)
    prompts = [all_prompts[slot - 1] for slot in slots]
    adapter_before = pinned_adapter(deadline)
    need({name: metadata.version(name) for name in VERSIONS} == VERSIONS, "Qualified runtime differs")
    for name, pin in ENGINE_PINS.items():
        experiment.read(Path("/swift-runtime/site/swift") / name, 2 << 20, pin)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    miopen = verify_miopen_home(Path("/tmp/gqa-rank-homes") / str(args.rank))
    args.output.mkdir(mode=0o700)
    experiment.write(args.output / "miopen-home.json", experiment.encode(miopen))
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from swift.infer_engine import AdapterRequest, InferRequest, RequestConfig, TransformersEngine
    from swift.model import get_model_processor
    from swift.template import get_template
    from vlm.runtime.first_update_audit import fingerprint

    need(torch.version.hip == "6.3.26113" and not torch.version.cuda and torch.cuda.device_count() == args.world,
         "Exactly eight allocated visible HIP devices required")
    torch.cuda.set_device(args.rank)
    pci = physical_pci(args.rank)
    experiment.write(args.output / "device.json", experiment.encode(
        {"step": args.step, "request": args.request, "rank": args.rank, "PCI": pci, "PID": os.getpid()}))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    started = time.monotonic()
    model, processor = get_model_processor(str(MODEL), model_type="internvl", load_model=True,
                                           download_model=False, device_map={"": args.rank},
                                           torch_dtype=torch.bfloat16, attn_impl="sdpa")
    load_seconds = time.monotonic() - started
    deadline()
    need(processor.model_meta.template == "internvl_hf" and not hasattr(model, "peft_config")
         and {str(p.device) for p in model.parameters()} == {f"cuda:{args.rank}"},
         "The unadapted original model must live on the allocated card")
    model.eval()
    model.requires_grad_(False)
    template = get_template(processor, template_type="internvl_hf", max_length=4096, max_pixels=262144,
                            padding_side="right", padding_free=False, truncation_strategy="raise")
    need(template.template_meta.template_type == "internvl_hf", "Qualified InternVL template differs")
    engine = TransformersEngine(model, template=template, max_batch_size=1)
    config = RequestConfig(max_tokens=experiment.CAP, temperature=0, num_beams=1, n=1,
                           stream=False, return_details=True, stop=[])
    effective = engine._prepare_generation_config(config)
    need(not effective.do_sample and effective.num_beams == 1 and effective.max_new_tokens == experiment.CAP,
         "Fixed decoding policy differs")
    state = load_file(str(CHECKPOINT / "adapter_model.safetensors"), device="cpu")
    need(len(state) == 504 and sum(p.numel() for p in state.values()) == 43_646_976
         and all(re.search(r"\.lora_[AB]\.weight$", k) and bool(torch.isfinite(p).all()) for k, p in state.items())
         and all(stats(CHECKPOINT / name) == witness for name, witness in adapter_before.items()),
         "Pinned CP25 state or file lifetime differs")
    arm_facts = {}
    paired_pins = {}
    current = {}
    adapter_checked = False
    collate, generate, get_ids = template.data_collator, template.generate, template.get_generate_ids

    def observe_collate(*params, **options):
        value = collate(*params, **options)
        need(current["input_sha"] is None, "Expected one native collation per request")
        current["input_sha"] = native_input_pin(value, fingerprint)
        need((current["input_sha"], value["input_ids"].shape[-1]) == frozen_native[current["slot"]],
             "Native GPU driver input differs from the frozen CPU preflight")
        other = paired_pins.get(current["slot"])
        need(other is None or current["input_sha"] == other, "Native processed pair differs")
        paired_pins[current["slot"]] = current["input_sha"]
        return value

    def observe_generate(*params, **options):
        nonlocal adapter_checked
        dc = options["generation_config"]
        name = current["arm"]
        need(current["eos"] is None and current["input_sha"] is not None and not dc.do_sample
             and dc.num_beams == 1 and dc.max_new_tokens == experiment.CAP
             and not engine.model.training and options.get("past_key_values") is None,
             "Unexpected InternVL native generation path")
        if name == "base":
            need(engine.model is model and not options.get("adapter_names"), "Base arm activated an adapter")
        else:
            need(options.get("adapter_names") == [ADAPTER_NAME] and engine.model.get_base_model() is model,
                 "Domain arm must use the original base with the registered CP25")
            if not adapter_checked:
                live = dict(engine.model.named_parameters())
                for k, tensor in sorted(state.items()):
                    key = k.replace(".lora_A.", f".lora_A.{ADAPTER_NAME}.").replace(".lora_B.", f".lora_B.{ADAPTER_NAME}.")
                    need(key in live and torch.equal(live[key].detach().cpu(), tensor),
                         "Actual live CP25 value differs from original safetensors")
                need(all(stats(CHECKPOINT / key) == v for key, v in adapter_before.items()), "Adapter changed while loading")
                adapter_checked = True
        raw_eos = dc.eos_token_id
        eos = raw_eos if isinstance(raw_eos, list) else [raw_eos]
        need(eos and all(type(token) is int and token >= 0 for token in eos), "Actual EOS identity required")
        current["eos"] = eos
        return generate(*params, **options)

    def observe_ids(*params, **options):
        value = get_ids(*params, **options)
        need(current["raw"] is None and value.ndim == 2 and value.shape[0] == 1, "Expected a single decoded sequence")
        current["raw"] = value[0].detach().cpu().tolist()
        return value

    template.data_collator, template.generate, template.get_generate_ids = observe_collate, observe_generate, observe_ids
    original_generation_failure = None
    raw_pins = {}
    for arm in experiment.ARMS:
        rows = []
        adapter = AdapterRequest(name=ADAPTER_NAME, path=str(CHECKPOINT)) if arm == "domain25" else None
        for prompt in prompts:
            slot = prompt["slot"]
            current.clear()
            current.update(arm=arm, slot=slot, input_sha=None, eos=None, raw=None)
            if original_generation_failure is not None:
                value = record(slot, "missing")
            else:
                try:
                    deadline()
                    response, = engine.infer([InferRequest(**user_input(prompt, args.input))],
                                             config, use_tqdm=False, adapter_request=adapter)
                    if isinstance(response, BaseException):
                        raise response
                    choice, = response.choices
                    raw = current["raw"]
                    pad = engine.tokenizer.pad_token_id
                    need(choice.token_ids == [token for token in raw if token != pad], "Decoded token IDs differ")
                    status, tokens = select_generation(raw, current["eos"], pad)
                    need(status is not None and isinstance(choice.message.content, str), "Incomplete native return")
                    value = record(slot, status, current["input_sha"], choice.message.content, tokens)
                except Exception as error:
                    value = record(slot, "error", current["input_sha"])
                    original_generation_failure = {
                        "arm": arm, "slot": slot, "class": type(error).__name__,
                        "message_SHA256": experiment.digest(str(error).encode()),
                        "generation_entered": current["eos"] is not None, "native_ids_returned": current["raw"] is not None,
                    }
            experiment.write(args.output / f"{arm}-{slot:03d}.json", experiment.encode(value))
            rows.append(value)
        arm_facts[arm] = dict(Counter(row["status"] for row in rows))
        raw = experiment.encode({"schema": "INTERNVL_GQA_VAL96_PREDICTION_V1", "arm": arm,
                                 "input_SHA256": INPUT_SHA, "rows": rows})
        experiment.prediction_rows(raw, arm, INPUT_SHA, slots)
        experiment.write(args.output / f"{arm}.json", raw)
        raw_pins[arm] = experiment.fields(raw)
    complete = original_generation_failure is None and adapter_checked
    seal = experiment.encode({"schema": "INTERNVL_GQA_VAL96_SEALED_V1", "input_SHA256": INPUT_SHA, "arms": raw_pins})
    if complete:
        experiment.write(args.output / "SEALED.json", seal)
    outcome = {"step": args.step, "request": args.request, "fresh_process": True, "rank": args.rank, "PCI": pci,
               "planned": 2 * len(slots), "counts": arm_facts, "load_seconds": load_seconds, "model_loads": 1,
               "live_checkpoint_tensors_verified": len(state) if adapter_checked else 0,
               "seal_SHA256": experiment.digest(seal) if complete else None,
               "input_SHA256": INPUT_SHA, "CPU_processed_input_proof_SHA256": PROCESSED_INPUT_SHA,
               "gold_read": False, "failure": original_generation_failure}
    experiment.write(args.output / "outcome.json", experiment.encode(outcome))
    return int(not complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--step", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-proof", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epoch", required=True, type=float)
    parser.add_argument("--deadline", required=True, type=float)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world", type=int, default=8)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
