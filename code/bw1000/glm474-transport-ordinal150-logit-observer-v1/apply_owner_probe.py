"""Emit the exact opt-in probe overlay for the qualified Socket/SHCA owner."""
import argparse
import ast
import hashlib
from pathlib import Path


BEFORE = "    if token in eos:break\n"
AFTER = """    # Transport observation follows the original greedy broadcast, before another forward.
    if rank == 0 and request['ordinal'] == 150 and len(request['generated_token_ids']) == 43:
     attach(request, transport_pin, rank=rank, arm=q['Native_arm'], past=past, vocab=VOCAB,
            read_logits=lambda: logits[0].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    if token in eos:break
"""
SETUP = "  from scoring import grade\n"
PROBE_SETUP = """  from transport_observer import attach, load_pin, preserve_missing
  transport_pin=load_pin(__file__)
"""
COMPLETED = "   ids=request['generated_token_ids'];natural="
COMPLETE_HOOK = """   if rank == 0 and request['ordinal'] == 150:preserve_missing(request, transport_pin, rank)
"""


def patch_owner(raw):
    expected = "ba045e33d91875ec7fca6aab852ea69159eb1cfe8b412e2df18afcd619d50750"
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("original Socket/SHCA owner source was changed")
    source = raw.decode()
    if any(source.count(anchor) != 1 for anchor in (BEFORE, COMPLETED, SETUP)):
        raise ValueError("original greedy and completion anchors changed")
    patched = source.replace(SETUP, SETUP + PROBE_SETUP, 1)
    patched = patched.replace(BEFORE, AFTER, 1).replace(COMPLETED, COMPLETE_HOOK + COMPLETED, 1)
    ast.parse(patched)
    return patched.encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("original", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    with args.output.open("xb") as stream:
        stream.write(patch_owner(args.original.read_bytes()))


if __name__ == "__main__":
    main()
