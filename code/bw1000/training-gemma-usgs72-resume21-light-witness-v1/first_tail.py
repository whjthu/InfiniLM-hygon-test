"""One bounded metadata witness at the first forward of Gemma update 21."""

import hashlib
import json
import os


OBSERVED_RANKS = (0, 1)
STEP_BEFORE = 20
MICRO_IN_UPDATE = 1


def need(condition, code):
    if not condition:
        raise ValueError(code)


def rng_hashes(rng, fingerprint, cpu):
    state = rng["numpy"]
    numpy = dict(algorithm=state[0], keys=state[1].tolist(), position=int(state[2]),
                 has_gauss=int(state[3]), cached_gaussian=float(state[4]))
    need(len(rng["cuda"]) == 1, "EXACT_ASSIGNED_HIP_RNG_ONLY")
    values = dict(python=rng["python"], numpy=numpy, cpu=rng["cpu"], assigned_HIP=rng["cuda"][0])
    return {name: fingerprint(cpu(value))["SHA256"] for name, value in values.items()}


class FirstTail:
    def __init__(self, rank, arm, model, rng, context_snapshot, input_layouts):
        self.rank, self.arm, self.model = rank, arm, model
        self.rng, self.context_snapshot, self.input_layouts = rng, context_snapshot, input_layouts
        self.entered = self.returned = self.written = False

    def snapshot(self):
        rng = self.rng()
        context = self.context_snapshot(self.model, device_type="cuda")
        # Keep the aggregate and identity digest from the existing provider.
        context.pop("module_training")
        return dict(RNG=rng, context=context)

    def before(self, step, micro, argv, kwargs, input_record):
        if (step, micro) != (STEP_BEFORE, MICRO_IN_UPDATE):
            return
        need(not self.entered, "ONE_EXACT_FIRST_FORWARD_ENTRY")
        before = self.snapshot()
        before["input_layout"] = self.input_layouts(argv, kwargs)
        self.data = dict(rank=self.rank, arm=self.arm, global_step_before=step, micro_in_update=micro,
                         input_ordinal=input_record["ordinal"], full_input=input_record["full_input"],
                         before=before, after=None, loss=None)
        self.entered = True

    def after(self, step, micro):
        if (step, micro) != (STEP_BEFORE, MICRO_IN_UPDATE):
            return
        need(self.entered and not self.returned, "ONE_EXACT_FIRST_FORWARD_RETURN")
        self.data["after"] = self.snapshot()
        self.returned = True

    def write(self, step, micro, output_record, path):
        if (step, micro) != (STEP_BEFORE, MICRO_IN_UPDATE):
            return
        need(self.returned and not self.written and output_record["ordinal"] == self.data["input_ordinal"],
             "MATCHED_FIRST_FORWARD_INPUT_AND_LOSS")
        self.data["loss"] = {key: output_record[key] for key in ("loss", "loss_scalar")}
        raw = (json.dumps(self.data, sort_keys=True, allow_nan=False) + "\n").encode()
        need(len(raw) <= 64 << 10, "BOUNDED_FIRST_FORWARD_METADATA")
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.SHA256 = hashlib.sha256(raw).hexdigest()
        self.written = True


def for_rank(rank, arm, model, torch, rng_values, fingerprint, cpu):
    if rank not in OBSERVED_RANKS:
        return None
    from .forward_context_state import context_snapshot, input_layouts

    need(torch.version.hip and not torch.version.cuda and torch.cuda.current_device() == rank,
         "EXACT_ASSIGNED_HIP_RANK")
    return FirstTail(rank, arm, model, lambda: rng_hashes(
        rng_values(lambda: [torch.cuda.get_rng_state(rank)]), fingerprint, cpu), context_snapshot, input_layouts)
