"""Localize the first repeated step-21 difference after a byte-bound native restore."""

from hashlib import sha256
import inspect
import json
from math import isfinite
import os
from pathlib import Path
import time

from coco20_inputs import load_dataset, sample
from coco20_step import exact, pinned_framework
from observe_update import Witness


PACKAGE = Path("/fusion-run")
ORIGINAL = Path("/original-coco20")
FRAMEWORK = Path("/common-framework/generation/src/gencuda")


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def read_pinned(claim, name):
    path = ORIGINAL / name
    raw = path.read_bytes()
    pin = claim["original_files"][name]
    need(path.resolve() == path and len(raw) == pin["bytes"] and sha256(raw).hexdigest() == pin["SHA256"],
         "Original " + name + " differs")
    return raw


def append(stream, record):
    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def write(path, record):
    with path.open("x", encoding="utf-8") as stream:
        append(stream, record)


class BoundaryWitness(Witness):
    def __init__(self, torch, named, deadline, journal, hash_tree, forward):
        super().__init__(torch, tuple(named.values()), deadline, journal)
        self.named_by_name = named
        self.hash_tree = hash_tree
        self.original = forward
        self.gradient_sha256 = {}

    def forward_value(self, prediction, loss):
        super().forward_value(prediction, loss)
        need(self.operands == self.original["operands"] and self.loss == self.original["loss"],
             "Step 21 native operands, prediction or loss differ from the original")

    def gradients(self, stage, norm=None):
        self.deadline()
        # Hash after the real backward or clipping return, before the next native operation.
        # Copy only tensor identities into the receipt.
        if stage == "postclip":
            original_norm = float(norm)
            need(isfinite(original_norm) and original_norm > 0, "Invalid native clipping norm")
        self.gradient_sha256[stage] = {
            name: self.hash_tree(parameter.grad)
            for name, parameter in self.named_by_name.items()
        }
        super().gradients(stage, norm)


def run(model, native, torch, output, mode, claim, deadline):
    import flash_attn
    from diffsynth.diffusion import loss
    from gencuda import hy_resume_checkpoint as common
    from gencuda import zimage_training_resume_checkpoint as resume

    for module in (common, resume):
        path = Path(module.__file__).resolve()
        need(path == FRAMEWORK / path.name, "Checkpoint provider escaped the deployed framework")
        exact(path, claim["framework_checkpoint_modules"][path.name])
    captures, backend = pinned_framework(claim)
    update = captures.instrument_update(backend._train_step)
    exact(Path(loss.__file__).resolve(), claim["loss_provider"])
    need(native.FlowMatchSFTLoss is loss.FlowMatchSFTLoss, "Original native loss identity differs")

    recipe = json.loads(read_pinned(claim, "continuous/contract.json"))
    originals = {}
    for original_mode in ("continuous", "resume20"):
        rows = [json.loads(line) for line in read_pinned(claim, original_mode + "/steps.jsonl").splitlines()]
        [originals[original_mode]] = [row for row in rows if row["global_step"] == 21]
    forward = originals["continuous"]
    need(all(forward[field] == originals["resume20"][field] for field in ("input_identity", "operands", "loss")),
         "Original continuous and resumed forward records do not match")
    need(recipe["seed"] == claim["base_seed"], "Original training seed differs")
    # The optimizer may attach the supplied state tensors by reference. Every arm
    # decodes its own checkpoint payload before restoring and updating.
    checkpoint_name = "continuous/checkpoint-20.private.pt"
    read_pinned(claim, checkpoint_name)
    checkpoint = ORIGINAL / checkpoint_name
    dataset, rows, original_manifest_present = load_dataset(PACKAGE, claim, deadline)
    model.train()
    scheduler = model.pipe.scheduler
    scheduler.set_timesteps(1000, training=True)
    scheduler_contract = resume.scheduler_spec(scheduler)
    named = {name: p for name, p in model.named_parameters() if p.requires_grad}
    frozen = {name: p for name, p in model.named_parameters() if not p.requires_grad}
    versions = {name: p._version for name, p in frozen.items()}
    need(len(named) == 476 and sum(p.numel() for p in named.values()) == 9922560, "Native LoRA boundary differs")
    optimizer = torch.optim.AdamW(list(model.trainable_modules()), lr=1e-4, weight_decay=.01)
    need({id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in named.values()},
         "Native optimizer parameter set differs")
    original_worker = json.loads(read_pinned(claim, "continuous/worker-result.json"))
    before = json.loads((PACKAGE / "pre.json").read_bytes())
    need(original_worker["physical_PCI"] in before["PCIs"], "Original device left the current node")
    try:
        signature = inspect.signature(flash_attn.flash_attn_func)
        deterministic_argument = "deterministic" in signature.parameters
    except (TypeError, ValueError):
        deterministic_argument = None

    arms = claim["arms"][mode]
    finished = []
    with (output / "optimizer-journal.jsonl").open("x", encoding="utf-8") as journal, \
            (output / "steps.jsonl").open("x", encoding="utf-8") as metrics:
        for arm in arms:
            deadline()
            started = time.monotonic()
            payload = resume.load_checkpoint(checkpoint)
            progress, restored = resume.restore_checkpoint(payload, recipe, model, optimizer, scheduler, scheduler_contract)
            resume.assert_restored_boundary(payload, model, optimizer, progress, scheduler, [0])
            need(restored["metadata"]["global_step"] == 20 and restored["next"]["sample_cursor"] == 0
                 and restored["next"]["global_step"] == 21, "Restored first use differs")
            restored_sha = payload["integrity_sha256"]
            del payload
            data, input_identity = sample(dataset, rows, 0)
            need(input_identity == forward["input_identity"], "Restored step 21 input differs")
            witness = BoundaryWitness(torch, named, deadline,
                                      lambda event: append(journal, dict(global_step=21, arm=arm, event=event)),
                                      common.tree_sha256, forward)
            with captures.observe_native_loss(native, witness):
                value = update(model, optimizer, data, recipe["seed"] + 21, 1., observer=witness)
            append(journal, dict(global_step=21, arm=arm, event="optimizer_returned"))
            witness.bind_return(value)
            progress = common.advance_optimizer_progress(progress, model, optimizer)
            need(all(p.grad is None and p._version == versions[name] for name, p in frozen.items()),
                 "A frozen parameter received a gradient or changed version")
            need(all(bool(torch.isfinite(p).all()) for p in named.values()), "Nonfinite updated LoRA parameter")
            after = resume.make_payload(contract=recipe, global_step=21, model=model, optimizer=optimizer,
                                        optimizer_progress=progress, scheduler=scheduler,
                                        expected_scheduler=scheduler_contract)
            append(metrics, dict(
                arm=arm, global_step=21, restored_integrity_SHA256=restored_sha,
                original_checkpoint_SHA256=claim["original_files"][checkpoint_name]["SHA256"],
                input_identity=input_identity, operands=witness.operands, loss=value["loss"],
                preclip_norm=value["grad_norm"], gradients=witness.gradient_sha256,
                adapter_SHA256=after["adapter_sha256"],
                adapter_tensor_SHA256={name: common.tree_sha256(p) for name, p in named.items()},
                optimizer_SHA256=common.tree_sha256(after["state"]["optimizer"]),
                rng_SHA256=common.tree_sha256(after["rng"]),
                diffusion_scheduler_SHA256=common.tree_sha256(after["scheduler"]),
                diagnostic_seconds=time.monotonic() - started,
            ))
            del after
            finished.append(arm)
    write(output / "repeat-receipt.json",
          dict(arms=finished, original_manifest_present=original_manifest_present,
               flash2_has_deterministic_argument=deterministic_argument,
               original_reference_pci=original_worker["physical_PCI"], formal_A100=False))
    return dict(arms=finished, optimizer_updates=len(finished), restored_exact=True,
                formal_A100=False, formal_performance=False)
