"""Read hidden journals only; raw native journals are a separate producer.

Filename schema is exact. A raw snapshot journal must never be interpreted as
hidden records merely because both producers use the same stage/rank prefix.
"""
import pathlib,json,re

def read_boundary_records(trace_root):
    records=[]
    if not trace_root:return records
    for p in sorted(pathlib.Path(trace_root).glob('stage*-rank*.jsonl')):
        if re.fullmatch(r'stage\d+-rank\d+\.jsonl',p.name)is None:continue
        for line in p.read_text().splitlines():
            v=json.loads(line)
            assert v['packed_target_only']and len(v['requests'])==1
            records.append(v)
    return records
