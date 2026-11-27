"""Derive a private tensor witness from the qualified LLaVA CLEVR CPU probe."""

import argparse
from hashlib import sha256
from pathlib import Path


PROBE_SHA256 = "446d4d0b27d9eff6488a6ebe07bce959fdf6e81b4f708e805c7c6cd237183aaf"
FINGERPRINT_SHA256 = "816ef694537273ac04d8f7d40df5694a16e8a869797d2d2f65664d21e6a0ea49"


def render(probe):
    if sha256(probe).hexdigest() != PROBE_SHA256:
        raise ValueError("The original CLEVR supervisor differs")
    source = probe.decode("utf-8")
    load = "        result[\"providers\"].update(encode=identity(template.encode),"
    canonical = (
        "        from vlm.runtime import first_update_audit as witness_audit\n"
        "        audit_path = args.framework / 'vlm/runtime/first_update_audit.py'\n"
        "        require(Path(witness_audit.__file__).resolve() == audit_path and\n"
        f"                sha(read(audit_path, 65536)) == {FINGERPRINT_SHA256!r},\n"
        "                'EXACT_CANONICAL_TENSOR_AUDIT_PROVIDER')\n"
        "        result['fingerprint_source_sha256'] = sha(read(audit_path, 65536))\n"
    )
    record = '                record.update(metrics, status="passed", image_sha256=pin["sha256"])'
    measure = (
        "                integers = {key: collated[key].detach().to(torch.int64).contiguous()\n"
        "                            for key in ('input_ids', 'labels', 'attention_mask')}\n"
        "                witness = witness_audit.fingerprint(\n"
        "                    dict(integers, pixel_values=collated['pixel_values']))\n"
        "                require(witness['finite'] is True and witness['floating_elements'] > 0,\n"
        "                        'FINITE_COLLATOR_WITNESS')\n"
        "                record['input_sha256'] = witness['SHA256']\n"
    )
    for anchor, addition in ((load, canonical), (record, measure)):
        if source.count(anchor) != 1:
            raise ValueError("The tensor audit insertion point differs")
        source = source.replace(anchor, addition + anchor)
    source = source.replace("CLEVR_TRAIN_ONLY_NATIVE_CPU_SUPERVISION", "CLEVR_TRAIN_ONLY_NATIVE_COLLATOR_WITNESS")
    compile(source, "collated_witness.py", "exec")
    return source.encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    witness = render(args.probe.read_bytes())
    with args.output.open("xb") as stream:
        stream.write(witness)
    print(sha256(witness).hexdigest())


if __name__ == "__main__":
    main()
