"""Build the Gemma question-only training view from the qualified target artifact."""

import hashlib
import json
import os
from pathlib import Path
import stat


RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
ORIGINAL = RESULTS / "training-gemma-usgs72-question-only-native-CPU-v1/audit"
OUTPUT = RESULTS / "training-gemma-usgs72-question-only-view800-v1"
SOURCE_BYTES = 43014
SOURCE_SHA = "21637fd30cab21ef005c64d6eddb09657e6933fd7817c75cba75622a1aaa79fa"
OUTCOME_SHA = "8f869945c8e97ad32ff8a1ec22e3c00f18256723d917b9b351ffe6e2852f1251"
SOURCE_ROWS = 72
RANKS, BATCH, ACCUMULATION, UPDATES = 8, 1, 4, 25


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_pinned(path, expected, limit):
    def identity(s):
        return s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "invalid source file extent")
    with path.open("rb") as stream:
        require(identity(os.fstat(stream.fileno())) == identity(before), "source file changed at open")
        raw = stream.read(before.st_size + 1)
        require(identity(os.fstat(stream.fileno())) == identity(before), "source file changed during read")
    require(identity(path.lstat()) == identity(before) and len(raw) == before.st_size and sha(raw) == expected,
            "frozen source file differs")
    return raw


def exposure_plan(raw):
    rows = raw.splitlines(keepends=True)
    require(len(rows) == SOURCE_ROWS and all(line.endswith(b"\n") for line in rows), "exact 72 complete rows required")
    seen = set()
    for line in rows:
        row = json.loads(line)
        key = row["id"]
        messages = row["messages"]
        require(type(key) is str and key and key not in seen, "unique original source ID required")
        require(not set(row) & {"images", "videos", "audios", "objects", "labels"}
                and [message["role"] for message in messages] == ["user", "assistant"]
                and all(type(message["content"]) is str and message["content"] for message in messages)
                and not any(tag in messages[0]["content"] for tag in ("<image>", "<video>", "<audio>"))
                and messages[1]["content"].startswith("FINAL: "), "question-only source contract differs")
        seen.add(key)
    total = RANKS * BATCH * ACCUMULATION * UPDATES
    order = [i % SOURCE_ROWS for i in range(total)]
    view = b"".join(rows[i] for i in order)
    receipt = dict(source_rows=SOURCE_ROWS, source_SHA256=sha(raw), source_bytes=len(raw),
                   source_row_SHA256=[sha(row) for row in rows], source_index_by_physical_row=order,
                   exposure_count_by_source_index=[order.count(i) for i in range(SOURCE_ROWS)],
                   physical_rows=total, world_size=RANKS, per_rank_batch=BATCH,
                   gradient_accumulation=ACCUMULATION, planned_updates=UPDATES,
                   expected_per_rank_microbatches=ACCUMULATION * UPDATES,
                   view_SHA256=sha(view), view_bytes=len(view), new_independent_examples=0,
                   native_rank_sampler_verified=False, actual_updates=0, training_ready=False)
    return view, receipt


def materialize():
    outcome = json.loads(read_pinned(ORIGINAL / "result.json", OUTCOME_SHA, 65536))
    require(outcome["request"] == "eef6ad53dab54c3994d70b3adf25a4fd" and outcome["step"] == "1103606.1197"
            and outcome["passed"] is True and outcome["completed"] == SOURCE_ROWS
            and outcome["derived_train"] == dict(bytes=SOURCE_BYTES, SHA256=SOURCE_SHA),
            "original native qualification differs")
    raw = read_pinned(ORIGINAL / "train72-question-only-private.jsonl", SOURCE_SHA, SOURCE_BYTES)
    require(len(raw) == SOURCE_BYTES, "original source length differs")
    view, receipt = exposure_plan(raw)
    receipt.update(original_CPU_step=outcome["step"], original_CPU_result_SHA256=OUTCOME_SHA)
    OUTPUT.mkdir(mode=0o700)
    ledger = (json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode()
    for name, body in (("train800-swift.jsonl", view), ("exposure-ledger-private.json", ledger)):
        with (OUTPUT / name).open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    return {**{k: receipt[k] for k in ("source_rows", "source_SHA256", "physical_rows", "view_SHA256", "view_bytes",
                                       "expected_per_rank_microbatches", "native_rank_sampler_verified",
                                       "new_independent_examples", "actual_updates", "training_ready")},
            "ledger_SHA256": sha(ledger), "original_CPU_step": outcome["step"]}


if __name__ == "__main__":
    print(json.dumps(materialize(), sort_keys=True, allow_nan=False))
