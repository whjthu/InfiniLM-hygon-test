"""Exercise the candidate through true two-rank DDP and a synchronized final backward."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

from grouped_accumulation import grouped_local_backward

WORLD = 2
UPDATES = 2
MICROBATCHES = 4


def check(ok, detail):
    if not ok:
        raise AssertionError(detail)


def run_fixture(rank, root):
    import torch
    from torch import nn
    from torch import distributed as dist

    torch.set_num_threads(1)
    check(not os.environ.get("SLURM_STEP_GPUS")
          and all(os.environ.get(key) == "-1" for key in
                  ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")),
          "CPU-only fixture device mask differs")

    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.active = nn.ParameterList([nn.Parameter(torch.randn(8, 8)) for _ in range(129)])
            self.intermittent = nn.Parameter(torch.randn(8, 8))
            self.frozen = nn.Parameter(torch.randn(8, 8), requires_grad=False)

        def forward(self, value, use_head):
            total = sum((weight * value).sum() for weight in self.active)
            return total + ((self.intermittent * value).sum() if use_head else 0) + (self.frozen * value).sum()

    def run_arm(candidate):
        torch.manual_seed(539)
        model = nn.parallel.DistributedDataParallel(SmallModel(), gradient_as_bucket_view=False)
        trainable = tuple(value for value in model.parameters() if value.requires_grad)
        optimizer = torch.optim.AdamW(trainable, lr=1e-3, foreach=False, fused=False)
        merged = []
        records = []
        failures = 0
        for update in range(UPDATES):
            optimizer.zero_grad(set_to_none=True)
            for microbatch in range(MICROBATCHES):
                final = microbatch == MICROBATCHES - 1
                with (nullcontext() if final else model.no_sync()):
                    def backward():
                        value = torch.arange(64, dtype=torch.float32).view(8, 8)
                        value = (value + 1 + rank * 2 + (microbatch + update) * 3) / 128
                        # Exercise a new gradient in microbatch 1 or a retained gradient
                        # while the optional head is unused in the middle microbatch.
                        loss = model(value, microbatch > 0 and microbatch != 2) / MICROBATCHES
                        loss.backward()
                        return loss.detach()

                    if candidate and microbatch in (1, 2):
                        loss, count = grouped_local_backward(torch, model, trainable, backward)
                        merged.append(count)
                    else:
                        if candidate and final:
                            try:
                                grouped_local_backward(torch, model, trainable, lambda: None)
                            except ValueError:
                                failures += 1
                            else:
                                raise AssertionError("Synchronized DDP was accepted")
                        loss = backward()
                    check(torch.isfinite(loss).item(), "Nonfinite native loss")
                records.append(tuple(None if p.grad is None else p.grad.clone() for p in trainable))
            optimizer.step()
            records.append(tuple(x.detach().clone() for x in trainable))
            records.append(tuple(optimizer.state[p][key].clone() for p in trainable
                                 for key in ("step", "exp_avg", "exp_avg_sq")))
        check(model.module.frozen.grad is None and len(trainable) == 130, "Frozen boundary or shape denominator differs")
        return records, merged, failures

    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=WORLD)
    try:
        baseline, _, _ = run_arm(False)
        actual, calls, rejected = run_arm(True)
        equal = 0
        for expected, observed in zip(baseline, actual, strict=True):
            for left, right in zip(expected, observed, strict=True):
                check((left is None and right is None) or (left is not None and right is not None
                      and torch.equal(left, right)), "DDP gradient or AdamW update changed")
                equal += 1
        check(calls == [129, 129, 129, 129] and rejected == 2, "Expected middle and final boundary coverage differs")
        payload = {"rank": rank, "world": WORLD, "combined_tensors_per_candidate_microbatch": calls,
                   "candidate_synchronized_calls_rejected": rejected, "exact_tensor_or_none_comparisons": equal,
                   "updates_per_arm": UPDATES, "microbatches_per_update": MICROBATCHES,
                   "torch_version": str(torch.__version__), "backend": dist.get_backend()}
        with (root / f"rank-{rank}.json").open("x", encoding="utf-8") as result:
            json.dump(payload, result, allow_nan=False)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing

    check(os.environ["SLURM_JOB_ID"] == "1103606" and os.environ["SLURM_STEP_ID"].isdigit(),
          "CPU fixture must run in the claimed Slurm scope")
    torch.multiprocessing.spawn(run_fixture, args=(Path(sys.argv[1]).resolve(),), nprocs=WORLD, join=True)
