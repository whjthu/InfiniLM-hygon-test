"""Compare native step 21 with and without ten prior updates in the process."""
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path

from coco20_inputs import load_dataset, sample
from coco20_step import exact, pinned_framework
from domain_step import data_hash
from layer_trace import observe, validate
from observe_update import Witness


PACKAGE = Path("/fusion-run")
ORIGINAL = Path("/original-coco20")
FRAMEWORK = Path("/common-framework/generation/src/gencuda")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(name, claim):
    path = ORIGINAL / name
    pin = claim["original_files"][name]
    need(path.resolve(strict=True) == path and path.stat().st_size == pin["bytes"],
         f"Original file size or identity differs: {name}")
    with path.open("rb") as source:
        digest = sha256()
        for data in iter(lambda: source.read(4 << 20), b""):
            digest.update(data)
    need(digest.hexdigest() == pin["SHA256"], f"Original file SHA differs: {name}")
    return path


class ForwardCaptured(Exception):
    """Unwind the native train step before its backward and optimizer calls."""


class ForwardWitness(Witness):
    def forward_value(self, prediction, loss):
        super().forward_value(prediction, loss)
        raise ForwardCaptured()

    def gradients(self, stage, norm=None):
        raise RuntimeError("A measured forward crossed into backward")


def run(model, native, torch, output, mode, claim, deadline, attention):
    from diffsynth.diffusion import loss
    from gencuda import hy_resume_checkpoint as common
    from gencuda import zimage_training_resume_checkpoint as resume

    exact(Path(loss.__file__).resolve(), claim["loss_provider"])
    need(native.FlowMatchSFTLoss is loss.FlowMatchSFTLoss, "Native loss entry differs")
    dit_source = Path(inspect.getfile(type(model.pipe.dit))).resolve()
    need(dit_source == Path("/native-source/diffsynth/models/z_image_dit.py"), "Native DiT route differs")
    exact(dit_source, claim["native_dit"])
    for module in (common, resume):
        path = Path(module.__file__).resolve()
        need(path == FRAMEWORK / path.name, "Checkpoint provider escaped the frozen framework")
        exact(path, claim["framework_checkpoint_modules"][path.name])
    captures, backend = pinned_framework(claim)
    update = captures.instrument_update(backend._train_step)
    recipe = json.loads(pinned("continuous/contract.json", claim).read_bytes())
    old = [json.loads(line) for line in pinned("continuous/steps.jsonl", claim).read_bytes().splitlines()]
    need([row["global_step"] for row in old] == list(range(1, 26)), "Original denominator differs")

    dataset, rows, _ = load_dataset(PACKAGE, claim, deadline)
    model.train()
    scheduler = model.pipe.scheduler
    scheduler.set_timesteps(1000, training=True)
    scheduler_contract = resume.scheduler_spec(scheduler)
    named = {name: p for name, p in model.named_parameters() if p.requires_grad}
    frozen = {name: (p, p._version) for name, p in model.named_parameters() if not p.requires_grad}
    need(len(named) == 476 and sum(p.numel() for p in named.values()) == 9922560,
         "Native rank-4 trainable boundary differs")
    optimizer = torch.optim.AdamW(list(model.trainable_modules()), lr=1e-4, weight_decay=.01)
    need({id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in named.values()},
         "Native AdamW ownership differs")

    def restore(step):
        deadline()
        checkpoint = pinned(f"continuous/checkpoint-{step}.private.pt", claim)
        payload = resume.load_checkpoint(checkpoint)
        progress, restored = resume.restore_checkpoint(payload, recipe, model, optimizer,
                                                        scheduler, scheduler_contract)
        resume.assert_restored_boundary(payload, model, optimizer, progress, scheduler, [0])
        need(restored["metadata"]["global_step"] == step and
             restored["next"]["sample_cursor"] == step % 20 and
             restored["next"]["global_step"] == step + 1, "Original checkpoint cursor differs")
        return progress, payload["integrity_sha256"]

    warm_steps = range(11, 21) if mode == "continuous" else ()
    completed = []
    with (output / "optimizer-journal.jsonl").open("x", encoding="utf-8") as journal:
        def record(step, event):
            journal.write(json.dumps({"global_step": step, "event": event}) + "\n")
            journal.flush()
            os.fsync(journal.fileno())

        if warm_steps:
            progress, _ = restore(10)
        for step in warm_steps:
            deadline()
            data, identity = sample(dataset, rows, (step - 1) % 20)
            need(identity == old[step - 1]["input_identity"], "History input changed")
            witness = Witness(torch, named.values(), deadline, lambda event: record(step, event))
            with captures.observe_native_loss(native, witness):
                result = update(model, optimizer, data, recipe["seed"] + step, 1., observer=witness)
            record(step, "optimizer_returned")
            witness.bind_return(result)
            progress = common.advance_optimizer_progress(progress, model, optimizer)
            completed.append(step)

    _, restored_sha = restore(20)
    data, identity = sample(dataset, rows, 0)
    need(identity == old[20]["input_identity"], "Step 21 input changed")
    optimizer.zero_grad(set_to_none=True)
    versions = {name: parameter._version for name, parameter in named.items()}
    optimizer_before = common.tree_sha256(optimizer.state_dict())
    prior_calls = attention["forced_calls"]

    def unexpected(_event):
        raise RuntimeError("Measured step 21 started an optimizer update")

    witness = ForwardWitness(torch, named.values(), deadline, unexpected)
    with (output / "forward-boundary-progress.jsonl").open("x", encoding="utf-8") as stream:
        def persist_boundary(row):
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        with observe(model.pipe.dit, torch.Tensor, data_hash, deadline, persist_boundary) as trace:
            with captures.observe_native_loss(native, witness):
                try:
                    update(model, optimizer, data, recipe["seed"] + 21, 1., observer=witness)
                except ForwardCaptured:
                    pass
                else:
                    raise RuntimeError("Measured native forward did not stop before backward")
    validate(trace, model.pipe.dit)
    need(witness.stages == ["prepared", "forward"] and attention["forced_calls"] > prior_calls,
         "Incomplete grad-enabled native forward")
    need(all(p.grad is None and p._version == versions[name] for name, p in named.items())
         and all(p.grad is None and p._version == version for p, version in frozen.values())
         and common.tree_sha256(optimizer.state_dict()) == optimizer_before,
         "Measured forward mutated gradients, parameters or optimizer")
    need({k: v for k, v in witness.operands.items() if k != "prediction_SHA256"}
         == {k: v for k, v in old[20]["operands"].items() if k != "prediction_SHA256"},
         "Measured native operands changed")
    evidence = {"mode": mode, "warm_updates": completed, "restored_integrity_SHA256": restored_sha,
                "input_identity": identity, "operands": witness.operands, "loss": witness.loss,
                "matches_original_continuous_prediction":
                    witness.operands["prediction_SHA256"] == old[20]["operands"]["prediction_SHA256"],
                "trace": trace, "measured_optimizer_updates": 0}
    with (output / "forward-trace.json").open("x", encoding="utf-8") as target:
        json.dump(evidence, target, allow_nan=False, sort_keys=True)
    return {"warm_updates": completed, "optimizer_updates": len(completed),
            "measured_optimizer_updates": 0, "native_forward_boundaries": len(trace)}
