"""One explicit long-pair residency; the external draft is outside target WORLD."""
from pathlib import Path
import hashlib
import json
import math
import re

TARGET_NODES = ['a14r2n08', 'a14r2n09', 'a14r2n10', 'a14r2n12']
DRAFT_NODE = 'a02r3n15'
CPU_NODE = 'a14r2n20'
NATIVE_SHA = '4dfdc9a7809b367ef91b476280f12bca6da0c12511eb14dae679147382246505'
WORKER_SHA = '49fa1d269f18f78820c7589496011300a7648fe340933e3c7a0e9105958b93c2'
GPU_SECONDS, NATIVE_SECONDS, CPU_SECONDS = 29400, 28800, 32400
PORTS = [32947, 39947, 45947, 46947]
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_plan(plan, *, execution=False):
    if execution and plan.get('execute_once') is not True:
        raise ValueError('Concrete source-published one-attempt plan required')
    assert plan['schema'] == 1 and re.fullmatch('[a-z0-9][a-z0-9-]{7,39}', plan['run'])
    assert plan['target_parent'] == 910380 and plan['target_nodes'] == TARGET_NODES
    assert plan['draft_parent'] == 910371 and plan['draft_node'] == DRAFT_NODE
    assert plan['CPU_parent'] == 910407 and plan['CPU_node'] == CPU_NODE
    for key, value in dict(target_cards=32, draft_cards=1, TP=16, PP=2, context=32768,
                           requests=4, output_budget=128, GPU_max_s=GPU_SECONDS,
                           Native_max_s=NATIVE_SECONDS, CPU_max_s=CPU_SECONDS).items():
        assert type(plan[key]) is int and plan[key] == value
    assert plan['request_order'] == [['retained8635', 'AR'], ['near32k', 'AR'],
                                     ['near32k', 'spec'], ['retained8635', 'spec']]
    assert plan['prompt_tokens'] == [8635, 32638] and plan['ports'] == PORTS
    assert plan['Native_SHA256'] == NATIVE_SHA and plan['worker_manifest_SHA256'] == WORKER_SHA
    assert plan['publication_branch'] == 'bw1000-experiments'
    assert re.fullmatch('[0-9a-f]{40}', plan['publication_commit'])
    for key in ['controller_manifest_SHA256', 'target_manifest_SHA256', 'CPU_receipt_SHA256',
                'target_CPU_receipt_SHA256']:
        assert re.fullmatch('[0-9a-f]{64}', plan[key])
    assert plan['retry'] is False and plan['reload'] is False and plan['profile'] is False
    return plan


def marker_identity(run, nonce_hash, epoch):
    assert re.fullmatch('[a-z0-9][a-z0-9-]{7,39}', run)
    assert re.fullmatch('[0-9a-f]{64}', nonce_hash) and math.isfinite(epoch) and epoch > 0
    return dict(schema=1, run=run, nonce_SHA256=nonce_hash, original_GPU_start_unix=epoch,
                target_Native_SHA256=NATIVE_SHA, worker_manifest_SHA256=WORKER_SHA,
                requested_by_node=0)


def read_start(path, expected):
    """Writer uses exclusive creation; incomplete first write is never a launch."""
    path = Path(path)
    if not path.exists(): return False
    raw = path.read_bytes()
    if not raw.endswith(b'\n'): return False
    actual = json.loads(raw)
    if actual != expected or any(type(actual[k]) is not type(v) for k, v in expected.items()):
        raise ValueError('Draft start identity mismatch; no draft submission')
    return True


def publish_once(path, record):
    import os
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(record, f, separators=(',', ':'), allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def verify_bundle(directory, expected):
    directory = Path(directory)
    assert sha(directory/'SOURCE_MANIFEST.json') == expected
    manifest = json.loads((directory/'SOURCE_MANIFEST.json').read_bytes())
    assert manifest['files_SHA256']
    for name, digest in manifest['files_SHA256'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert sha(directory/name) == digest, name
    return manifest
