"""Bind the private W4 entry to its tested client and unchanged parent loader."""
from pathlib import Path
import hashlib
import json
from dso_binding import verify_dso

CPU628_SHA = '303bab995e13b03c1c3fe072333fcff21ea9e9533b051aece14d8ce478e71c16'
PARENT_SHA = 'afde8ff72bdbfd32fbf174691f4cc5069fc356cf743ac577813b2a68b8e5a9ae'
OWNER_SHA = '3508075aa744f3bfb8f3631af73b1112f8905b0c810800b7ec0819885b718545'
CPU_PACKAGE_SHA = '2f8e8bcb03033e13960404c83b714d9bccdffae0b2084cf4106b0623e46e864d'
CHANGED = {'glm_prefill_execute_r236.py', 'owner_r290.py', 'client_driver_r290.py',
           'client_paired_r290.py', 'GPU_inside_r290.sh'}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_candidate(root, directory, source):
    root, source = Path(root), Path(source)
    parent = root / 'incoming/GLM-latest-full-owner-r299'
    assert sha(parent / 'SOURCE_MANIFEST.json') == PARENT_SHA
    assert sha(source / 'SOURCE_MANIFEST.json') == OWNER_SHA
    old = json.loads((parent / 'SOURCE_MANIFEST.json').read_bytes())
    new = json.loads((source / 'SOURCE_MANIFEST.json').read_bytes())
    assert new['parent_manifest_SHA256'] == PARENT_SHA
    assert set(new['changed_members']) == CHANGED
    assert set(new['files_SHA256']) == set(old['files_SHA256'])
    for name, digest in old['files_SHA256'].items():
        assert sha(parent / name) == digest
        if name not in CHANGED:
            assert new['files_SHA256'][name] == digest and sha(source / name) == digest
    assert (source / 'GPU_inside_r290.sh').read_text() == (parent / 'GPU_inside_r290.sh').read_text().replace(
        'GLM-latest-full-owner-r299', 'GLM-prefill-W2-W4-owner-r307')
    # The current model loader/DSO audit and all dependency/math files are exact
    # parent bytes. CPU621 evidence retains its original, unchanged scope.
    verify_dso(root, parent, 'GLM')
    package = root / 'incoming/GLM-W2-W4-CPU-r307b'
    assert sha(package / 'SOURCE_MANIFEST.json') == CPU_PACKAGE_SHA
    cpu_source = json.loads((package / 'SOURCE_MANIFEST.json').read_bytes())
    assert all(sha(package / n) == h for n, h in cpu_source['files_SHA256'].items())
    receipt = root / 'candidates/GLM-W2-W4-CPU-r307b/CPU-inside.json'
    assert sha(receipt) == CPU628_SHA
    q = json.loads(receipt.read_bytes())
    assert q['completed'] and q['passed'] and q['CPU_only'] and q['actual_CPU'] == '910407.628'
    assert q['source_manifest_SHA256'] == OWNER_SHA
    assert q['CPU_source_SHA256'] == sha(package / 'CPU_main.py')
    assert all(q[k] == 0 for k in ['GPU', 'Model', 'Native_dlopen', 'production_HTTP'])
    assert q['real_driver_and_client_main_executed'] and q['actual_frozen_request_read']
    assert [(x['mode'], x['synthetic_HTTP_attempts'], x['returncode']) for x in q['cases']] == [
        ('positive', 6, 0), ('wrong-step', 0, 1), ('wrong-window', 3, 1), ('wrong-tool', 1, 1)]
    return {'CPU628_SHA256': CPU628_SHA, 'parent299_SHA256': PARENT_SHA,
            'unchanged_Native_math_and_loader': True, 'new_runtime_qualification': False}


def require_baseline_terminal(root, go):
    assert go['baseline_result_consumed'] is True
    assert isinstance(go['baseline_result_record'], str) and go['baseline_result_record'].startswith('results/bw1000/')
    p = Path(root) / 'results/GLM-latest99-full-Model-r302-watch.json'
    w = json.loads(p.read_bytes())
    assert w['actual_GPU'] == '910376.816' and w['completed'] is True and w['post_empty'] is True
    assert w['terminal'][0] == '910376.816'
    assert w['terminal'][2].split()[0].rstrip('+') in {
        'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
        'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
    return {'baseline_actual': '910376.816', 'baseline_terminal': w['terminal'],
            'baseline_watch_SHA256': sha(p), 'root_consumed_result': go['baseline_result_record']}
