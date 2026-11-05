"""Freeze deterministic GLM rollback cases from the original private token plan."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

PARENT = "299c00c3234db357defe1aab2e5e5eb3f004c70658c719c31d71c6b52e9d9824"
CASES = (("below", 4, 0), ("above", 8, 8), ("crossing_prompt_slice", 8, 4))
VOCAB = 154880
EOS = {154820, 154827, 154829}


def encode(value):
    return (json.dumps(value, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def replacement(original):
    token = (original + 1) % VOCAB
    while token in EOS:
        token = (token + 1) % VOCAB
    return token


def freeze(raw):
    if sha256(raw).hexdigest() != PARENT:
        raise ValueError("unexpected original private GLM plan")
    source = json.loads(raw)["cases"]
    if [row["geometry"] for row in source] != [name for name, _, _ in CASES]:
        raise ValueError("unexpected original input order")
    rows = []
    for sequence, ((geometry, width, accepted), row) in enumerate(zip(CASES, source, strict=True), start=1):
        proposal = row["proposal"][:width]
        # Use a deliberately different token at each replaced rejected slot.
        # Full acceptance uses two fixed diagnostic values from the same rule.
        positions = (accepted, accepted + 1) if accepted < width else (width - 2, width - 1)
        continuation = [replacement(proposal[i]) for i in positions]
        prompt, chunks = row["prompt"], row["chunk_widths"]
        if (len(prompt) + max(width, accepted + len(continuation)) > 32768
                or len(proposal) != width or set(proposal) & EOS or sum(chunks) != len(prompt)):
            raise ValueError("invalid continuation source geometry")
        rows.append(dict(geometry=geometry, sequence=sequence, prompt=prompt, chunk_widths=chunks,
                         proposal=proposal, accepted=accepted, continuation=continuation))
    return encode(dict(parent_private_plan_SHA256=PARENT, cases=rows))


def summary(raw):
    plan = json.loads(raw)
    cases = [dict(geometry=row["geometry"], sequence=row["sequence"], prompt_tokens=len(row["prompt"]),
                  chunks=len(row["chunk_widths"]), width=len(row["proposal"]), accepted=row["accepted"],
                  continuation_tokens=len(row["continuation"]),
                  first_continuation_past=len(row["prompt"]) + row["accepted"],
                  CAPI_forwards_per_owner=2 * len(row["chunk_widths"]) + row["accepted"]
                                         + 1 + 2 * len(row["continuation"]))
             for row in plan["cases"]]
    return dict(private_plan_SHA256=sha256(raw).hexdigest(),
                parent_private_plan_SHA256=plan["parent_private_plan_SHA256"], cases=cases,
                CAPI_forwards_per_owner=sum(row["CAPI_forwards_per_owner"] for row in cases),
                context_loads_per_owner=1, measured_acceptance=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    args = parser.parse_args()
    raw = freeze(args.parent.read_bytes())
    print(json.dumps(summary(raw), indent=2))

