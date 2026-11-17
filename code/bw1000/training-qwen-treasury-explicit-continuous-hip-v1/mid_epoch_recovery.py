"""Materialize the shared cursor/witness extensions against the frozen framework."""

import argparse
import hashlib
import json
from pathlib import Path


CURSOR = "vlm/runtime/resume_cursor.py"
WITNESS = "vlm/runtime/resume_witness.py"
BASE_PINS = {
    CURSOR: "ab49439782518c0abd91bb3515d0f102fe056087ccd250effd681eadb9e640d0",
    WITNESS: "8dcdf503002ecfbb7c0b933196639ac86044abe955924643b5ecc1ba6168af15",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def replace_once(raw, old, new):
    old, new = old.encode(), new.encode()
    if raw.count(old) != 1:
        raise ValueError("framework recovery patch anchor differs")
    return raw.replace(old, new, 1)


def patch_cursor(raw):
    raw = replace_once(raw, "    dataloader_batches_per_rank: int,\n",
                       "    dataloader_batches_per_rank: int, allow_mid_epoch: bool = False,\n")
    raw = replace_once(raw,
        "    Only complete accumulation groups and an exact epoch boundary are supported.\n"
        "    Mid-epoch checkpoints need their own data-skip and RNG witness before use.\n",
        "    Only complete accumulation groups are supported. Epoch boundaries remain\n"
        "    the default. An opted-in caller must witness native skip, input order and RNG.\n")
    raw = replace_once(raw,
        "    if checkpoint_step % updates_per_epoch:\n"
        "        raise ValueError(\"mid-epoch resume is outside this cursor contract\")\n"
        "    expected_epoch = checkpoint_step // updates_per_epoch\n",
        "    if type(allow_mid_epoch) is not bool:\n"
        "        raise ValueError(\"mid-epoch opt-in must be boolean\")\n"
        "    complete_epochs, completed_updates = divmod(checkpoint_step, updates_per_epoch)\n"
        "    skip_batches = completed_updates * gradient_accumulation_steps\n"
        "    if skip_batches and not allow_mid_epoch:\n"
        "        raise ValueError(\"mid-epoch resume is outside this cursor contract\")\n"
        "    expected_epoch = complete_epochs + skip_batches / dataloader_batches_per_rank\n")
    raw = replace_once(raw,
        '        "expected_epochs_trained": expected_epoch,\n'
        '        "expected_batches_to_skip_in_resumed_epoch": 0,\n',
        '        "expected_epochs_trained": complete_epochs,\n'
        '        "expected_batches_to_skip_in_resumed_epoch": skip_batches,\n')
    return replace_once(raw,
        '        "qualification": "Epoch-boundary cursor plan only; no model or checkpoint state loaded.",\n',
        '        "qualification": ("Mid-epoch cursor plan only; native data skip and restored state unverified."\n'
        '                          if skip_batches else "Epoch-boundary cursor plan only; no model or checkpoint state loaded."),\n')


def patch_witness(raw):
    raw = replace_once(raw, "                 boundary_modules=None):\n",
                       "                 boundary_modules=None, allow_mid_epoch=False):\n")
    return replace_once(raw, "                dataloader_batches_per_rank=len(loader))\n",
                       "                dataloader_batches_per_rank=len(loader), allow_mid_epoch=allow_mid_epoch)\n")


def candidate_sources(framework):
    patched = {}
    for name, patch in ((CURSOR, patch_cursor), (WITNESS, patch_witness)):
        raw = (framework / name).read_bytes()
        if digest(raw) != BASE_PINS[name]:
            raise ValueError(f"frozen framework file differs: {name}")
        patched[name] = patch(raw)
        compile(patched[name], name, "exec")
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new candidate overlay directory")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.framework.resolve()):
        raise ValueError("candidate overlay must be outside the original framework")
    sources = candidate_sources(args.framework)
    args.output.mkdir(mode=0o700, exist_ok=False)
    for name, raw in sources.items():
        target = args.output / name
        target.parent.mkdir(mode=0o700, exist_ok=True, parents=True)
        target.write_bytes(raw)
    print(json.dumps({name: digest(raw) for name, raw in sources.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
