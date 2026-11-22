"""Check the sealed Z-Image tokenizer on the exact staged 12 training prompts."""

import argparse
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import struct


TOKENS = ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")
CSV_SHA = "f71b51feb39af32e1d79b7750c9c4c437d930a35f69b60e52f80389ba6084350"
MANIFEST_SHA = "88b9886c9c4aeee3c5468fa935539f0166bf9d1b8a37eb30cef0f55a1f7bf13b"


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def digest(raw):
    return sha256(raw).hexdigest()


def checked_tokenizer_files(token_root, manifest):
    raw = manifest.read_bytes()
    need(digest(raw) == MANIFEST_SHA, "runtime manifest differs")
    pin = {row["filename"]: row for row in json.loads(raw)["assets"]
           if row["target_relative_path"].startswith("base/tokenizer/")}
    need(set(pin) == set(TOKENS), "tokenizer file set differs")
    for name, value in pin.items():
        p = token_root / name
        need(p.is_file() and not p.is_symlink() and p.stat().st_size == value["bytes"]
             and digest(p.read_bytes()) == value["SHA256"], "tokenizer component differs")
    return len(pin)


def run(token_root, metadata, manifest):
    member_count = checked_tokenizer_files(token_root, manifest)
    raw = metadata.read_bytes()
    need(len(raw) == 3601 and digest(raw) == CSV_SHA, "original staged 12-row CSV differs")
    reader = csv.DictReader(StringIO(raw.decode("utf-8")))
    rows = list(reader)
    need(reader.fieldnames == ["image", "prompt"] and len(rows) == 12
         and len({row["image"] for row in rows}) == 12
         and all(row["prompt"] and not row["prompt"].isspace() for row in rows), "original training row roster differs")

    from transformers import AutoTokenizer, __version__

    tokenizer = AutoTokenizer.from_pretrained(str(token_root), local_files_only=True, trust_remote_code=False)
    prompts = [row["prompt"] for row in rows]
    batch = tokenizer(prompts, add_special_tokens=True, return_attention_mask=True, truncation=False)
    receipts = []
    for index, (prompt, ids, mask) in enumerate(zip(prompts, batch["input_ids"], batch["attention_mask"], strict=True)):
        solo = tokenizer(prompt, add_special_tokens=True, return_attention_mask=True, truncation=False)
        need(ids == solo["input_ids"] and mask == solo["attention_mask"] and len(ids) == len(mask) > 0
             and all(type(i) is int and 0 <= i < 2**32 for i in ids) and all(x == 1 for x in mask),
             "batch and single-prompt tokenizer results differ")
        receipts.append({"ordinal": index + 1, "tokens": len(ids),
                         "iterable_u32le_SHA256": digest(struct.pack(f"<{len(ids)}I", *ids))})
    return {"status": "TARGET_TOKENIZER_SMOKE_PASS", "scope": "Direct offline tokenizer on unchanged raw training prompts",
            "transformers": __version__, "tokenizer_class": type(tokenizer).__name__,
            "checked_tokenizer_files": member_count, "prompts_checked": len(receipts), "rows": receipts,
            "trained_updates": 0, "A100_token_reference_compared": False,
            "native_DiffSynth_prompt_template_verified": False, "model_loaded": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", required=True, type=Path)
    p.add_argument("--metadata", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(run(args.tokenizer, args.metadata, args.manifest), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
