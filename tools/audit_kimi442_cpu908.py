"""Reparse saved CPU908 artifacts. No imports of candidate code or device calls."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def audit(path):
    raw = path.read_bytes()
    archive = json.loads(raw)
    assert archive['readonly'] and archive['new_jobs'] == 0
    files = archive['files']
    for name, item in files.items():
        b = item['text'].encode('utf-8')
        assert len(b) == item['bytes'] and sha(b) == item['SHA256'], name
    assert files['result.json']['SHA256'] == 'ec69cf11db161291006e683f94f61c8fa6baa05b9a1fe58f105b2e72220eeab1'
    q = json.loads(files['result.json']['text'])
    assert q['actual_CPU'] == '910407.908' and q['completed'] and q['passed']
    assert not any(q[k] for k in ['Native_dlopen', 'Runtime_executed', 'GPU', 'Model_READY', 'compile', 'link', 'Model_GO'])
    expected = {role + suffix for role in ['Native', 'Common_extension', 'Candidate'] for suffix in ['-dynamic', '-ldd']} | {'old-exports', 'new-exports'}
    assert {c['name'] for c in q['commands']} == expected and len(q['commands']) == 8
    for c in q['commands']:
        assert c['rc'] == 0
        for stream in ['stdout', 'stderr']:
            assert c[stream + '_SHA256'] == files[c['name'] + '.' + stream]['SHA256']
        assert files[c['name'] + '.stderr']['text'] == ''
    proof = json.loads(files['deployment/DEPLOYMENT.json']['text'])
    assert q['consumer']['proof'] == proof
    assert not proof['GPU_GO'] and not proof['Model_READY']
    for rel, key in [('deployment/config.json', 'config_SHA256'),
                     ('deployment/parent/SOURCE_MANIFEST.json', 'parent_MF_SHA256'),
                     ('deployment/text/SOURCE_MANIFEST.json', 'text_MF_SHA256')]:
        assert files[rel]['SHA256'] == proof[key]
    assert proof['Runtime_SHA256'] == '5b8df2e6dd334b88e3a1493b928dce1a25fe2a7878c2eccdf141a74aaa3ad36b'
    runtime = proof['runtime_directory'] + '/libinfinilm_runtime.so'
    parsed = {}
    for role in ['Native', 'Common_extension']:
        dynamic = files[role + '-dynamic.stdout']['text']
        needed = re.findall(r'\(NEEDED\).*?\[(.*?)\]', dynamic)
        assert [n for n in needed if 'infinilm_runtime' in n] == ['libinfinilm_runtime.so']
        text = files[role + '-ldd.stdout']['text']
        assert 'not found' not in text and 'undefined symbol' not in text
        providers = []
        for line in text.splitlines():
            match = re.match(r'\s*(\S+)\s+=>\s+(/\S+)\s', line)
            direct = re.match(r'\s*(/\S+)\s+\(', line)
            if match:
                providers.append((match.group(1), match.group(2)))
            elif direct:
                providers.append((Path(direct.group(1)).name, direct.group(1)))
        runtime_rows = [(n, p) for n, p in providers if 'infinilm_runtime' in n or 'infinilm_runtime' in p]
        assert runtime_rows == [('libinfinilm_runtime.so', runtime)], runtime_rows
        parsed[role] = {k: [p for n, p in providers if n == k]
                        for k in q['actual_trace_providers'][role]}
        assert all(len(v) == 1 for v in parsed[role].values())
        for name, observation in q['actual_trace_providers'][role].items():
            assert parsed[role][name] == [observation['path']]
    common = files['Common_extension-dynamic.stdout']['text']
    assert '(RPATH)' not in common and '$ORIGIN/infinilm/_lib' in common
    assert '(SONAME)' not in files['Candidate-dynamic.stdout']['text']
    def strong(text):
        return {r[-1] for line in text.splitlines() if len(r := line.split()) == 3 and r[1] in 'TDBRSG'}
    old = strong(files['old-exports.stdout']['text'])
    new = strong(files['new-exports.stdout']['text'])
    assert len(old) == 149 and old <= new and 'modern_cpu_copy_row_hits_r436' in new
    consumer = q['consumer']
    assert consumer['default_OFF'] and not any(consumer[k] for k in ['Runtime_executed', 'Native_dlopen', 'GPU', 'Model_READY'])
    roles = consumer['actual_source_configure_roles']
    assert [r['node'] for r in roles] == list(range(4))
    assert [r['PP_stage'] for r in roles] == [0, 0, 1, 1]
    assert [r['TP_node_rank'] for r in roles] == [0, 1, 0, 1]
    assert all(r['local_Native_internal_workers'] == 8 and r['global_cards'] == 32 for r in roles)
    assert sorted(consumer['changed_original_members']) == sorted([
        'public_kimi_native/owner.py', 'public_kimi_native/runtime/native_binding_r9143.py', 'public_kimi_text/cli.py'])
    g = Path(__file__).resolve().parents[1]
    source = g / 'code/bw1000/Kimi-private-loading-Runtime-integration-r442'
    mf = source / 'SOURCE_MANIFEST.json'
    assert sha(mf.read_bytes()) == 'b37c4b4c59e87c449d6c4e87800ab5711da77b94f156a0699cfe7178d36245e4'
    for name, h in json.loads(mf.read_bytes())['files_SHA256'].items():
        assert sha((source / name).read_bytes()) == h, name
    return dict(state='ROOT_EXISTING_CPU908_ELF_AND_SOURCE_HANDOFF_AUDIT_PASS',
        original_CPU='910407.908', archive_SHA256=sha(raw), artifact_files=len(files),
        original_receipt_SHA256=files['result.json']['SHA256'], source_manifest_SHA256=sha(mf.read_bytes()),
        sole_Runtime_trace_path=runtime, Native_and_Common_both_resolve_candidate=True,
        strong_exports_retained=len(old), source_roles=roles, CPU_work_wall_s=q['wall_s'],
        qualification='ELF trace and changed source/config recording leaves only; no device constructors executed',
        new_CPU_runs=0, new_GPU_runs=0, new_Model_loads=0, actual_Runtime_copy_hit=None,
        original_Native_and_SDK_identity='Retained by pinned CPU receipt; ldd names independently reparsed, live device maps still required',
        next='Reuse at next otherwise necessary Kimi residency; verify live maps/dladdr and actual loading counter delta before claiming Model loading gain',
        performance_gain_new=None, Model_GO=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('archive', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.archive)
    assert not a.output.exists()
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
