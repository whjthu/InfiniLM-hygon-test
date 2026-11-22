"""Record the actual backward, clipping and AdamW boundaries for five paired updates."""

import json
import re
import time

from observe_update import Witness


OBSERVED_STEPS = range(21, 26)
STATE_FIELDS = ("step", "exp_avg", "exp_avg_sq")
CHECKS = ("preclip", "postclip", "LoRA", *STATE_FIELDS)


class TensorWitness(Witness):
    def __init__(self, torch, named, deadline, append, first_result, first_norm, digest, enabled):
        super().__init__(torch, named.values(), deadline, append, first_result, first_norm)
        self.named = named
        self.digest = digest
        self.enabled = enabled
        self.gradients_by_boundary = {}

    def gradients(self, stage, norm=None):
        super().gradients(stage, norm)
        if self.enabled:
            self.gradients_by_boundary[stage] = {name: self.digest(p.grad) for name, p in self.named.items()}


def after_update(witness, optimizer, step):
    if not witness.enabled:
        return None
    witness.deadline()
    if step not in OBSERVED_STEPS or set(witness.gradients_by_boundary) != {"preclip", "postclip"}:
        raise ValueError("Incomplete native backward or clipping boundary")
    record = dict(witness.gradients_by_boundary)
    record.update({field: {} for field in ("LoRA", *STATE_FIELDS)})
    for name, parameter in witness.named.items():
        state = optimizer.state[parameter]
        if set(state) != set(STATE_FIELDS) or float(state["step"].item()) != step:
            raise ValueError("Incomplete or unexpected AdamW state")
        record["LoRA"][name] = witness.digest(parameter)
        for field in STATE_FIELDS:
            record[field][name] = witness.digest(state[field])
    witness.deadline()
    return record


def paired_boundaries(left, right, expected_parameters=476):
    if ([row["global_step"] for row in left] != list(OBSERVED_STEPS)
            or [row["global_step"] for row in right] != list(OBSERVED_STEPS)):
        raise ValueError("Incomplete paired tensor-boundary denominator")
    differences = []
    identity = None
    for a, b in zip(left, right, strict=True):
        aa, bb = a["tensor_boundaries"], b["tensor_boundaries"]
        if set(aa) != set(CHECKS) or set(bb) != set(CHECKS):
            raise ValueError("Missing tensor boundary")
        if identity is None:
            identity = set(aa["LoRA"])
        for record in (aa, bb):
            for mapping in record.values():
                if (len(identity) != expected_parameters or set(mapping) != identity
                        or any(not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
                               for digest in mapping.values())):
                    raise ValueError("Trainable tensor names or digests changed")
        counts = {key: sum(aa[key][name] != bb[key][name] for name in identity) for key in CHECKS}
        differences.append({"global_step": b["global_step"], "changed_tensors": counts})
    first = {key: next((row["global_step"] for row in differences if row["changed_tensors"][key]), None)
             for key in CHECKS}
    return {"paired_updates": len(differences), "tensors_per_boundary": len(identity),
            "steps": differences, "first_difference_by_boundary": first}


def wait_for_worker_files(root, deadline_monotonic, sleep=time.sleep):
    ready = (
        ("continuous", "steps.jsonl"), ("resume20", "steps.jsonl"),
        ("continuous", "checkpoint-receipts.json"), ("resume20", "checkpoint-receipts.json"),
        ("continuous", "worker-result.json"), ("resume20", "worker-result.json"),
        ("resume20", "restore.json"),
    )
    while True:
        if all((root / directory / name).is_file() for directory, name in ready):
            for directory in ("continuous", "resume20"):
                worker = json.loads((root / directory / "worker-result.json").read_bytes())
                if worker["passed"] is not True:
                    raise ValueError("Worker terminal receipt failed")
            return
        remaining = deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Original worker phase files not visible within the launch deadline")
        sleep(min(1.0, remaining))
