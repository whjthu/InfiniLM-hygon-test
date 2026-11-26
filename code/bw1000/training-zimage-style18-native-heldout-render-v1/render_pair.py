"""Render frozen Style18 heldout pairs with the original strict two-arm loading."""

from hashlib import sha256
from math import isfinite
from pathlib import Path
import json
from heldout_contract import execute, load_claim_plan


TRAINED = Path("/recovery-checkpoint/continuous")


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = sha256()
    with path.open("rb") as f:
        for data in iter(lambda:f.read(4<<20),b""):
            h.update(data)
    return h.hexdigest()


def tensor_hash(torch, value):
    return sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def render(model,torch,output,claim,deadline):
    from gencuda import hy_resume_checkpoint as common
    from gencuda import zimage_training_resume_checkpoint as checkpoint

    for entry in (common,checkpoint):
        path=Path(entry.__file__).resolve()
        pin=claim["framework_checkpoint_modules"][path.name]
        need(path.parent==Path("/common-framework/generation/src/gencuda")
             and path.stat().st_size==pin["bytes"] and digest(path)==pin["SHA256"],
             "Imported checkpoint reader differs from the common framework")
    samples=load_claim_plan(claim)
    named={name:p for name,p in model.named_parameters() if p.requires_grad}
    initial={name:p.detach().cpu().clone() for name,p in named.items()}
    before=TRAINED/"INITIAL_ADAPTER_HASHES.private.json"
    need(digest(before)==claim["initial_training_hashes"]["SHA256"],"Original training initial adapter receipt differs")
    initial_hashes=json.loads(before.read_bytes())
    need(len(named)==476 and set(initial_hashes)=={n.removeprefix("pipe.dit.") for n in named},
         "Initial adapter parameter set differs")
    need(all(tensor_hash(torch,p)==initial_hashes[n.removeprefix("pipe.dit.")] for n,p in initial.items()),
         "Recreated initial adapter differs from the actual continuous run")
    path=TRAINED/"checkpoint-25.private.pt"
    need(path.stat().st_size==claim["continuous25_checkpoint"]["bytes"] and digest(path)==claim["continuous25_checkpoint"]["checkpoint_SHA256"],
         "Original continuous checkpoint body differs")
    raw=checkpoint.load_checkpoint(path)
    need(raw["metadata"]["global_step"]==25 and raw["metadata"]["next_data_cursor"]==1,
         "Candidate checkpoint is not the actual completed step 25")
    after=raw["state"]["trainable"]
    need(len(after)==476 and set(after)==set(initial) and common.tree_sha256(after)==claim["continuous25_checkpoint"]["adapter_SHA256"],
         "Trained adapter extent or identity differs")
    del raw
    states={"before":initial,"after":after}
    return execute(model,torch,output,samples,named,states,deadline,tensor_hash)
