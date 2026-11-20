"""Target-local exact-line exposure view; no encoding, gold collection or network."""
import hashlib
import json
from pathlib import Path

SOURCE_SHA256="0a8560e6046439aa7450a3c1080ff5e537be17a56a8d4d332ff501d302b6bc62"
SOURCE_ROWS=88
EXPOSURES=800


def schedule():
    # Frozen source order; no answers, loss, or model output used for selection.
    return [i % SOURCE_ROWS for i in range(EXPOSURES)]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def build(source, target):
    source,target=Path(source),Path(target)
    if source.is_symlink() or not source.is_file() or target.exists():
        raise ValueError("Expected pinned source and fresh view namespace")
    before=source.stat()
    raw=source.read_bytes()
    after=source.stat()
    fields=("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns")
    if (len(raw)>2*1024**2 or digest(raw)!=SOURCE_SHA256
            or any(getattr(before,k)!=getattr(after,k) for k in fields)):
        raise ValueError("Original qualified 88-row train source changed")
    lines=raw.splitlines(keepends=True)
    if len(lines)!=SOURCE_ROWS or any(not x.endswith(b"\n") or not x.strip() for x in lines):
        raise ValueError("Expected exactly 88 complete original JSONL lines")
    for line in lines:
        value=json.loads(line)
        if (not isinstance(value,dict) or len(value.get("messages",[]))!=2
                or [x.get("role") for x in value["messages"]]!=["user","assistant"]
                or len(value.get("images",[]))!=1):
            raise ValueError("Original train-only message/image interface differs")
    order=schedule()
    view=b"".join(lines[i] for i in order)
    target.mkdir(mode=0o700)
    (target/"training800.jsonl").write_bytes(view)
    ledger=dict(schema="qwen-usgs-fixed88-to800-exposure-view-v1",source_SHA256=SOURCE_SHA256,
                source_rows=SOURCE_ROWS,physical_row_count=EXPOSURES,
                original_index_by_physical_row=order,view_SHA256=digest(view),
                original_row_SHA256=[digest(x) for x in lines],
                exposure_counts=[order.count(i) for i in range(SOURCE_ROWS)],
                original_line_bytes_retained=True,new_independent_examples=0,
                new_labels=False,dev_or_blind_read=False)
    out=(json.dumps(ledger,sort_keys=True,separators=(",",":"))+"\n").encode()
    (target/"exposure-ledger-private.json").write_bytes(out)
    return dict(passed=True,original_unique_rows=88,physical_rows=800,unique_media=22,
                view={"bytes":len(view),"SHA256":digest(view)},ledger={"bytes":len(out),"SHA256":digest(out)},
                exposures_min=9,exposures_max=10,new_independent_examples=0,
                intended_world_size=8,micros_per_rank=100,accumulation=4,complete_updates=25,
                actual_encoded_samples=0,actual_optimizer_updates=0,dev_or_blind_read=False)
