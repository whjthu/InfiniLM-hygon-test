"""Render an optional metadata-only first-slot diagnostic from exact native v3."""

import argparse
import ast
from hashlib import sha256
from pathlib import Path


ORIGINAL = "a0af5ad1f92c79f369a710032e78bcfceb47e5f7925e902a7819cb2422f20517"


def derive(raw):
    if sha256(raw).hexdigest() != ORIGINAL:
        raise ValueError("EXACT_ORIGINAL_NATIVE_V3")
    source = raw.decode()
    pairs = (
        ("from contract import ARMS, CAP,", "from generation_boundary import BoundaryTrace\n\nfrom contract import ARMS, CAP,"),
        ('        return generate(*args, **kwargs)\n',
         '        trace = current.get("boundary")\n'
         '        if trace is not None:\n'
         '            with trace.attach_model(engine.model):\n'
         '                return current["boundary_generate"](*args, **kwargs)\n'
         '        return generate(*args, **kwargs)\n'),
        ('        ids = get_ids(*args, **kwargs)\n',
         '        ids = current.get("boundary_ids", get_ids)(*args, **kwargs)\n'),
        ('            current.update(phase="generation", arm=arm, slot=slot, input=None, eos=None, ids=None)\n',
         '            current.update(phase="generation", arm=arm, slot=slot, input=None, eos=None, ids=None)\n'
         '            if arm == "base" and slot == 0:\n'
         '                trace = BoundaryTrace()\n'
         '                current.update(boundary=trace,\n'
         '                    boundary_generate=trace.wrapper("template.generate", generate),\n'
         '                    boundary_ids=trace.wrapper("template.get_generate_ids", get_ids))\n'),
        ('            record(args.output, row)\n    current["phase"] = "asset_continuity"\n',
         '            record(args.output, row)\n'
         '            if "boundary" in current:\n'
         '                facts["first_generation_boundary"] = current.pop("boundary").project()\n'
         '                current.pop("boundary_generate")\n'
         '                current.pop("boundary_ids")\n'
         '    current["phase"] = "asset_continuity"\n'),
        ('        if current["phase"] == "load_base" and "base_diagnostic" in current:\n',
         '        if "boundary" in current:\n'
         '            failure["first_generation_boundary"] = current["boundary"].project()\n'
         '        if current["phase"] == "load_base" and "base_diagnostic" in current:\n'),
    )
    for original, replacement in pairs:
        if source.count(original) != 1:
            raise ValueError("EXACT_GENERATION_SOURCE_ANCHOR")
        source = source.replace(original, replacement, 1)
    ast.parse(source)
    return source.encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-v3", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = derive(args.native_v3.read_bytes())
    with args.out.open("xb") as out:
        out.write(raw)
    print(sha256(raw).hexdigest())
