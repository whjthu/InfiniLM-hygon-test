"""Bind a new one-card comparison to the 36 originally capped arithmetic prompts."""

import hashlib


LABEL = "training-qwen-treasury-dev-arith128-v1"
ORIGINAL_GENERATION_SHA256 = "1edbba41a164dd862057124011fbacc957914993f132eb12c180bd02b30903d1"
ORIGINAL_CONTROLLER_SHA256 = "393ddf6d57f76625c7c8da80251a8b663f6bd2711de6b0496cbffa31a60f4947"
CASE_IDS_SHA256 = "9afa5fb0d15ab1f6b797a75348546545e93daeab3993d2e4f34264c65b1055ad"


def rerender(raw, expected, replacements):
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("original Treasury source pin differs")
    text = raw.decode("utf8")
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError(f"expected exactly one source anchor: {old!r}")
        text = text.replace(old, new, 1)
    return text.encode("utf8")


def generation(raw):
    return rerender(raw, ORIGINAL_GENERATION_SHA256, {
        "MAX_NEW_TOKENS = 64": "MAX_NEW_TOKENS = 128",
        "    requests, _ = bound_inputs(root, args.inventory)\n": (
            "    requests, _ = bound_inputs(root, args.inventory)\n"
            '    requests = [row for row in requests if row["id"].endswith("_signed_difference")]\n'
            '    identity = ("\\n".join(row["id"] for row in requests) + "\\n").encode("ascii")\n'
            f'    require(len(requests) == 36 and hashlib.sha256(identity).hexdigest() == "{CASE_IDS_SHA256}",\n'
            '            "arithmetic subset identity differs")\n'
        ),
        "adapter_name=ADAPTER_NAME, planned_rows_per_arm=108, base=base, domain=domain)": (
            'adapter_name=ADAPTER_NAME, planned_rows_per_arm=36, base=base, domain=domain,\n'
            '                   topic="signed_difference", case_ids_SHA256=hashlib.sha256(identity).hexdigest())'
        ),
    })


def controller(raw):
    return rerender(raw, ORIGINAL_CONTROLLER_SHA256, {
        'RUN = RESULTS / "training-qwen-treasury-dev-pair-hip-v2"': f'RUN = RESULTS / "{LABEL}"',
        "budget <= 2700": "budget <= 3000",
        "1110 if gpu else 60": "1710 if gpu else 60",
        '("18" if gpu else "2")': '("28" if gpu else "2")',
        "time.time() + 1020": "time.time() + 1560",
        'receipt["base"]["rows"] == receipt["domain"]["rows"] == 108': (
            'receipt.get("max_new_tokens") == 128\n'
            '                    and receipt.get("topic") == "signed_difference"\n'
            f'                    and receipt.get("case_ids_SHA256") == "{CASE_IDS_SHA256}"\n'
            '                    and receipt["base"]["rows"] == receipt["domain"]["rows"] == 36'
        ),
    })
