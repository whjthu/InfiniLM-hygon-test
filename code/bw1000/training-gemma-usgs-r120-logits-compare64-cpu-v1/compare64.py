"""Certify selector positions and synthetic-logit loss gradients over the 64 original Gemma training slots."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

import probe
from alignment import IGNORE, positions, require


BUDGET = 240
PRIOR = "6f6af8f89c6a885b09fcc1cbb1ee4b717f7474eb248184217ed4ad402b5451cc"
LOSS_SHA = "83db16b24ce5c3a0642097624aa9a0ae7eb72445c41d8b4d5059a129862fdbbe"
ATOL = 2e-6


def blobs(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def write(stream, value):
    raw = blobs(value)
    stream.write(raw)
    stream.flush()
    os.fsync(stream.fileno())
    return raw


def compare_loss(torch, loss, full, compact, indices, supervised):
    """Use the installed Transformers criterion with deterministic synthetic FP32 logits."""
    values = sorted(set(full + compact) - {IGNORE})
    mapping = {value: i for i, value in enumerate(values)}
    width = len(values) + 29
    def targets(labels):
        return torch.tensor([[mapping.get(x, IGNORE) for x in labels]], dtype=torch.int64)
    left_targets, right_targets = targets(full), targets(compact)
    generator = torch.Generator(device="cpu").manual_seed(58372)
    shape = (1, len(full), width)
    base = torch.randn(shape, dtype=torch.float32, generator=generator)
    index = torch.tensor(indices, dtype=torch.int64)
    loss_error = gradient_error = 0.0
    for denominator in (None, supervised + 3):
        logits = base.detach().clone().requires_grad_(True)
        packed_logits = base.index_select(1, index).detach().requires_grad_(True)
        a = loss(logits, left_targets, vocab_size=width, num_items_in_batch=denominator, ignore_index=IGNORE)
        b = loss(packed_logits, right_targets, vocab_size=width, num_items_in_batch=denominator, ignore_index=IGNORE)
        full_grad, = torch.autograd.grad(a, logits)
        reduced_grad, = torch.autograd.grad(b, packed_logits)
        dense_grad = torch.zeros_like(full_grad).index_copy_(1, index, reduced_grad)
        delta = float((a-b).abs().detach())
        derivative = float((full_grad-dense_grad).abs().max())
        require(delta <= ATOL and derivative <= ATOL, "LOSS_OR_DLOGIT_GRADIENT_MISMATCH")
        loss_error = max(loss_error, delta)
        gradient_error = max(gradient_error, derivative)
    return dict(loss_max_absolute_error=loss_error, dlogit_max_absolute_error=gradient_error)


def application(args, summary, ledger):
    probe.source_bind(probe.NATIVE_MANIFEST)
    import torch
    require(not torch.cuda.is_initialized(), "UNEXPECTED_CUDA_INITIALIZATION")
    def deny(*_, **__):
        raise RuntimeError("GPU_CONTEXT_FORBIDDEN")
    torch.cuda._lazy_init = deny
    torch.set_num_threads(2)
    from transformers.loss.loss_utils import ForCausalLMLoss
    from cpu_index import PROCESSOR, verify_inventory
    from native_worker import NEW, trust_cpu_stage, read
    from gemma_usgs_r39_inputs import input_signature, batch_gate_metadata
    from swift.model import get_model_processor
    from swift.template import get_template

    path = Path(inspect.getfile(ForCausalLMLoss)).resolve()
    require(path == Path("/minicpm-transformers/transformers/loss/loss_utils.py"), "ORIGINAL_LOSS_PROVIDER_PATH")
    probe.checked(path, 8540, LOSS_SHA)
    require(inspect.getsourcelines(ForCausalLMLoss)[1] == 49, "ORIGINAL_LOSS_ENTRY")
    summary["phase"] = "original_cpu_view_and_template"
    view, index = trust_cpu_stage()
    verify_inventory(lambda:None)
    order = json.loads(read(NEW/"lineage-private.json",view["lineage"]["SHA256"],size=view["lineage"]["bytes"]))
    slots = json.loads(read(NEW/"native64-index-private.json",index["native_index"]["SHA256"],size=index["native_index"]["bytes"]))
    data = read(NEW/"train800-swift.jsonl",view["view"]["SHA256"],size=view["view"]["bytes"]).splitlines(keepends=True)
    require(len(data)==800 and set(slots["rank"])=={str(rank) for rank in range(8)}, "EXACT_ORIGINAL_SLOT_GROUPING")
    model, processor = get_model_processor(str(PROCESSOR), model_type="gemma3_vision", load_model=False,
        download_model=False,device_map="cpu",attn_impl="sdpa",torch_dtype=torch.bfloat16)
    require(model is None, "MODEL_WEIGHTS_MUST_NOT_LOAD")
    template = get_template(processor,template_type="gemma3_vision",max_length=512,max_pixels=262144,
        padding_side="right",padding_free=False,truncation_strategy="raise")
    template.set_mode("train")
    trainer, methods, capability, pins = probe.provider(torch,template,processor)
    summary["effective_callable_pins"] = pins
    summary["phase"] = "all_slots_native_preparation_and_loss"
    records = sha256()
    for rank in range(8):
        rank_rows=slots["rank"][str(rank)]
        require(len(rank_rows)==8,"EXACT_EIGHT_SLOT_RANK")
        for index_in_rank, row in enumerate(rank_rows):
            ordinal = row["ordinal"]
            require(1<=ordinal<=800 and row["forward"]==index_in_rank+1
                and order["rank_first64"][str(rank)][index_in_rank]==ordinal
                and sha256(data[ordinal-1]).hexdigest()==row["line_SHA256"],"EXACT_FROZEN_SLOT_AND_INPUT_IDS")
            record=json.loads(data[ordinal-1])
            batch=template.data_collator([template.encode({k:record[k] for k in ("messages","images")})])
            require(input_signature(batch)==row["input_SHA256"],"EXACT_FULL_INPUT_SIGNATURE_REQUIRED")
            summary["full_collator_slots"] += 1
            prepared, enabled, called=probe.native_prepare(torch,trainer,methods,capability,template,batch,None)
            comparison=probe.analyze(torch,batch,prepared,enabled,called,input_signature,batch_gate_metadata)
            require(enabled and comparison["selector_kind"]=="native_mask" and not comparison["loss_scale_present"]
                    and comparison["off_by_one_rejected"],"FROZEN_AUTO_BS1_MASK_AND_OFF_BY_ONE_CONTROL")
            full=batch["labels"][0].tolist()
            compact=prepared["labels"][0].tolist()
            selector=positions(prepared["logits_to_keep"].tolist(),len(full))
            compared=compare_loss(torch,ForCausalLMLoss,full,compact,selector,comparison["reference"]["supervised"])
            record=dict(rank=rank,slot=index_in_rank+1,full_label_length=len(full),candidate_label_length=len(compact),
                        full_signature=row["input_SHA256"],reference=comparison["reference"],**compared)
            records.update(write(ledger,record))
            summary["verified_slots"]+=1
            summary["supervised_total"]+=comparison["reference"]["supervised"]
            summary["off_by_one_rejected"]+=1
            for key, value in compared.items():
                summary["max_error"][key]=max(summary["max_error"].get(key,0),value)
    require(summary["verified_slots"]==64 and not torch.cuda.is_initialized(),"EXACT_64_SLOT_COMPLETE")
    summary["rows_SHA256"]=records.hexdigest()
    summary["passed"]=True


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--epoch",required=True,type=float)
    cli.add_argument("--request",required=True)
    cli.add_argument("--out",required=True,type=Path)
    args=cli.parse_args()
    require(re.fullmatch("[0-9a-f]{32}",args.request) and args.request in args.out.parent.name
        and args.out.is_absolute() and not args.out.exists(),"NEW_ABSOLUTE_PRIVATE_RESULT")
    probe.STEP_BUDGET=BUDGET
    step=probe.gate(args.epoch)
    summary=dict(request=args.request,actual_step=step,planned_slots=64,full_collator_slots=0,verified_slots=0,
        off_by_one_rejected=0,supervised_total=0,actual_GPU=0,actual_optimizer_updates=0,actual_model_forwards=0,
        prior_one_slot_report_SHA256=PRIOR,loss_provider_SHA256=LOSS_SHA,passed=False,max_error={},
        synthetic_FP32_logits=True,full_vocabulary_dimension_qualified=False,actual_training_loss_qualified=False,
        cpu_forward_capability_proxy=True,training_ready=False)
    def expired(*_):
        raise TimeoutError("CPU_COMPARE64_DEADLINE")
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,args.epoch+BUDGET-time.time())
    sys.addaudithook(probe.bounded_audit)
    try:
        with open(os.devnull,"w") as silent, redirect_stdout(silent),redirect_stderr(silent), \
                args.out.with_name("slots-private.jsonl").open("xb") as ledger:
            application(args,summary,ledger)
    except Exception as e:
        summary["failure"]=probe.failure(e)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
    with args.out.open("xb") as out:
        raw=write(out,summary)
    print(json.dumps(dict(request=args.request,passed=summary["passed"],verified_slots=summary["verified_slots"],
                          result_SHA256=sha256(raw).hexdigest())))
    return int(not summary["passed"])


if __name__=="__main__":
    raise SystemExit(main())
