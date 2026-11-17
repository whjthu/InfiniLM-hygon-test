"""Read-only final gate: saved whole-file proofs plus exact current stats, no body read.

Caller supplies PLAN and original asset_plan_SHA256/endpoint_SHA256. A PASS is file
byte identity only; no initial adapter, processor, runtime or training qualification.
"""
import json
import os
from pathlib import Path
import stat


def gate(plan, plan_sha, endpoint_sha):
    base = Path(plan['target_namespace'])
    rows = []
    for name, pin in plan['files'].items():
        path = base/'whole-verifications'/name/'outcome.json'
        for item in [*reversed(path.parents), path]:
            if item.is_symlink():
                raise ValueError('whole proof symlink')
        if not path.is_file() or not 0 < path.stat().st_size <= 65536:
            raise ValueError('whole proof missing/bound')
        with path.open('rb', buffering=0) as stream:
            before = os.fstat(stream.fileno()); raw = stream.read(65537); after = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or len(raw) != before.st_size:
            raise ValueError('whole proof changed')
        proof = json.loads(raw)
        model = base/'model'/name
        if model.is_symlink() or not model.is_file():
            raise ValueError('final model member missing/link')
        st = model.stat(); current = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
        if not (proof['passed'] is True and proof['whole_file_qualified'] is True
                and proof['member'] == name and proof['bytes'] == pin['bytes'] and proof['SHA256'] == pin['SHA256']
                and proof['asset_plan_SHA256'] == plan_sha and proof['endpoint_SHA256'] == endpoint_sha
                and proof['uid'] == st.st_uid == os.getuid() and proof['assembled_stat'] == current
                and stat.S_ISREG(st.st_mode) and proof['finished_epoch'] <= proof['deadline']):
            raise ValueError('whole proof/actual stable file differs')
        rows.append(dict(name=name, bytes=st.st_size, SHA256=pin['SHA256']))
    if len(rows) != 18 or sum(row['bytes'] for row in rows) != 17070557079:
        raise ValueError('final file denominator differs')
    return dict(case='T04.sft', whole_shards_verified=4, metadata_verified=14,
                complete_model_file_byte_identity_qualified=True, verified_files=rows,
                gate_body_reads=0, second_disk_body_read=False, initial_trainables_qualified=False,
                native_runtime_qualified=False, numerical_qualified=False, GPU_GO=False)
