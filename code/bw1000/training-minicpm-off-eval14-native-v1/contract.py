"""Frozen inputs and small metadata gates for the answerless packaging pilot."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
INPUT = ROOT / "incoming/training-minicpm-off-eval14-30f70022a75f"
PROMPTS = "prompts.answerless.jsonl"
PROMPT_SHA = "30f70022a75ffba0b0b2088619f06083ab3423a27af72e7bbc85718b291039af"
MODEL = ROOT / "results/training-minicpm-original-model-v1/MiniCPM-V-4.6"
MODEL_MANIFEST_SHA = "ddcb8756dc66474f348bd81bb1730874ba78af8b46567e6a99596cedf5594ae6"
TRAIN = ROOT / "results/training-minicpm-off-native2-c6dab1cb2089"
CP2 = TRAIN / "continuous/train/checkpoint-2"
CP_ARCHIVE_SHA = "1e17cd6c34eb8007d925504d8f1f4ed771cfaa474291dbd103bf6c26eef996ce"
CP_INVENTORY_SHA = "11207b254a8c417d0729927500ecd1c23ed80462f16dfa925f3d4b2ec019fb95"
CP_RELOAD_SHA = "03bbd7fe6047f6db2e93d4987fa1fa67f5d2130ae4fdb1e0148f4b58f0650220"
CP_FILES = {
    "adapter_config.json": {"bytes": 1230, "SHA256": "331a767f39acd369e92dc24f87286b1d8eb00b3c3525ed91f96a96bc6ea00e31"},
    "adapter_model.safetensors": {"bytes": 43346432, "SHA256": "49a98355eaaef21714e5de035a0a6245c7d88bb3dbc32872e1140acabd309fe9"},
}
CP_ADAPTER = {"SHA256": "81b448fd3b6dcaad1de9db89aabbd419be5b9dafc610d53f611ba08aea61c939",
              "tensors": 372, "elements": 10822656, "floating_elements": 10822656,
              "floating_l2": 31.483360356546246, "finite": True}
RELOAD_ROW = ["1103606.4486", "training-minicpm-off-native2-c6dab1cb2089-cp2-reload-9ce5f64a",
              "COMPLETED", "0:0", "7", "f13r1n19", "cpu=4,mem=8G,node=1",
              "2026-11-27T09:24:59", "2026-11-27T09:25:06"]
ARMS = ("base", "adapter")
SLOTS = tuple(f"pkg-eval-{i:02d}" for i in range(1, 15))
CAP = 32
HEX = re.compile(r"[0-9a-f]{64}")
HEX32 = re.compile(r"[0-9a-f]{32}")


def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def put(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"bytes": len(raw), "SHA256": sha(raw)}


def stat_key(st):
    return [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def verify_file(path, limit, expected=None, collect=False):
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and path.resolve(strict=True) == path.absolute() and 0 < before.st_size <= limit,
         "FILE_TYPE_PATH_OR_EXTENT")
    chunks = []
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        start = os.fstat(stream.fileno())
        need(os.name == "nt" or (start.st_dev == before.st_dev and stat_key(start) == stat_key(before)), "FILE_OPEN_IDENTITY")
        count = 0
        for part in iter(lambda: stream.read(8 * 1024**2), b""):
            count += len(part)
            need(count <= limit, "FILE_EXCEEDED_EXTENT")
            digest.update(part)
            if collect:
                chunks.append(part)
        need(start.st_dev == os.fstat(stream.fileno()).st_dev and stat_key(start) == stat_key(os.fstat(stream.fileno())),
             "OPEN_FILE_CHANGED")
    after = path.lstat()
    need(before.st_dev == after.st_dev and stat_key(before) == stat_key(after) and count == before.st_size, "FILE_CHANGED")
    result = {"bytes": count, "SHA256": digest.hexdigest()}
    if expected is not None:
        need(result["SHA256"] == expected, "FILE_SHA")
    return (b"".join(chunks) if collect else result), stat_key(before)


def small(path, size=256*1024, digest=None):
    return verify_file(path, size, digest, collect=True)[0]


def inputs(root, strict_tree=True, expected=PROMPT_SHA):
    raw = small(root / PROMPTS, 16384, expected)
    rows = [json.loads(line) for line in raw.splitlines()]
    need(len(rows) == 14 and all(type(row) is dict and set(row) == {"image", "image_SHA256", "group", "slot", "question"}
                                for row in rows) and tuple(r["slot"] for r in rows) == SLOTS, "ANSWERLESS_PROMPT_SCHEMA")
    count = Counter(r["group"] for r in rows)
    need(len(count) == 7 and set(count.values()) == {2}, "SEVEN_PAIRS_REQUIRED")
    media = {}
    for row in rows:
        name, pin = row["image"], row["image_SHA256"]
        need(type(name) is str and re.fullmatch(r"[0-9]+-[0-9]+\.jpg", name) and type(pin) is str
             and HEX.fullmatch(pin) and type(row["question"]) is str and 0 < len(row["question"]) <= 1000
             and not any(x in row["question"] for x in ("<image>", "<|im_start|>", "<|im_end|>")), "PROMPT_OR_IMAGE_SCOPE")
        need(name not in media or media[name] == pin, "PROMPT_IMAGE_PIN_DISAGREEMENT")
        media[name] = pin
    need(len(media) == len(set(media.values())) == 11, "ELEVEN_JPEGS_REQUIRED")
    if strict_tree:
        need({x.name for x in root.iterdir()} == {PROMPTS} | set(media), "ANSWERLESS_EXACT_DIRECTORY")
    for name, pin in media.items():
        verify_file(root / name, 16*1024**2, pin)
    return rows


def archive(raw):
    need(sha(raw) == CP_ARCHIVE_SHA, "CP2_ORIGINAL_ARCHIVE_PIN")
    data = json.loads(raw)
    need(data["scope"]["request"] == "c6dab1cb20894c45a2bf140a15fb0f2f"
         and data["actual"]["adapter"] == data["independent_CP2_CPU_reload"]["adapter"] == CP_ADAPTER
         and data["original_post_clean"] is True and data["independent_CP2_CPU_reload"]["actual_GPU"] == 0
         and data["reload_accounting_UTC"][:4] == ["1103606.4486", "training-minicpm-off-native2-c6dab1cb2089-cp2-reload-9ce5f64a", "COMPLETED", "0:0"],
         "ORIGINAL_CP2_CPU_RELOAD")
    return data["independent_CP2_CPU_reload"]["adapter"]


def actual_reload(raw, rows):
    value = json.loads(raw)
    need(rows == [RELOAD_ROW] and value["request"] =="c6dab1cb20894c45a2bf140a15fb0f2f"
         and value["adapter"] == CP_ADAPTER and value["actual_GPU"] == 0
         and value["actual_optimizer_updates"] == value["planned_optimizer_updates"] == 2
         and value["native_forwards"] == value["planned_native_forwards"] == 64
         and value["checkpoint_members"] == 14 and value["rng_members_hashed"] == 8
         and value["optimizer"] == {"slots":372,"steps":[2]} and value["scheduler"] == {"last_epoch":2}
         and value["device_restore_verified"] is value["full_training_resume_run"] is False
         and value["state"] == "OFF_NATIVE_CP2_INDEPENDENT_CPU_RELOAD_PASS", "ACTUAL_INDEPENDENT_CP2_ZERO_GRES_RELOAD")
    return {"result_SHA256": sha(raw), "accounting_step": RELOAD_ROW[0], "adapter": value["adapter"]}


def generation(ids, eos, pad):
    need(type(ids) is list and 0 < len(ids) <= CAP and all(type(x) is int and x >= 0 for x in ids), "GENERATION_ID_BOUNDS")
    for i, token in enumerate(ids):
        if token in eos:
            need(all(x == pad for x in ids[i+1:]), "POST_EOS_NONPAD")
            return "eos", i+1
    need(len(ids) == CAP, "NATIVE_GENERATION_INCOMPLETE")
    return "cap", len(ids)


def seal(root, request, rows, witnesses, failure, facts):
    pins, full = {}, []
    for arm in ARMS:
        native_rows = rows[arm]
        need(set(native_rows) <= set(SLOTS), "PRIMARY_SLOT_SCOPE")
        primary = []
        for slot in SLOTS:
            item = native_rows.get(slot)
            if item is None:
                full.append({"arm": arm, "slot": slot, "state": "missing"})
                continue
            need(set(item) == {"slot", "answer", "finish_reason"} and item["slot"] == slot
                 and (item["finish_reason"] in {"eos", "cap"} and type(item["answer"]) is str
                      or item["finish_reason"] == "error" and item["answer"] is None), "PRIMARY_ROW_CONTRACT")
            primary.append(item)
            full.append({"arm": arm, "slot": slot, "state": item["finish_reason"],
                         "input": witnesses[arm].get(slot)})
        pins[arm] = put(root / (arm+".jsonl"), b"".join(encode(row) for row in primary))
    counts = {arm: dict(Counter(x["state"] for x in full if x["arm"] == arm)) for arm in ARMS}
    complete = (failure is None and all(len(rows[arm]) == 14 for arm in ARMS)
                and witnesses["base"] == witnesses["adapter"] and len(witnesses["base"]) == 14
                and facts.get("actual_live_adapter") == CP_ADAPTER and facts.get("model_loads") == 1
                and facts.get("actual_native_calls") == 28)
    result = dict(request=request, passed=complete, planned=28, prompt_SHA256=PROMPT_SHA,
                  counts=counts, failure=failure, gold_read=False, source_facts=facts, arms=pins,
                  full_ledger=put(root/"ledger.private.json", encode(full)))
    put(root/"SEALED.json", encode(result))
    return result
