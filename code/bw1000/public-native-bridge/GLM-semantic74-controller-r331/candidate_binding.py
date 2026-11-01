"""Reuse the qualified parent dependency chain and bind the remaining74 CPU entry."""
from pathlib import Path
import hashlib
import json
from parent_pair_binding import verify_candidate as verify_parent_pair
from parent_pair_binding import require_baseline_terminal, CPU628_SHA

CPU641_SHA = '7a9357413371851a2ba17102dd91823131144cb8fa20d91bca2f1f7545250447'
PARENT_SHA = '3508075aa744f3bfb8f3631af73b1112f8905b0c810800b7ec0819885b718545'
OWNER_SHA = 'dc63a67bcacc3444a8da7718063cd5d3d78c0f9f5858fc548642a53ab4d58c72'
CPU_PACKAGE_SHA = '5c0de7714f7a46fabf35bc4cee2267b7d4ea07a111958337b41fd574076a64e7'
CHANGED = {'GPU_inside_r290.sh','PREPARATION.json','client_driver_r290.py',
           'client_multicall_r290.py','client_paired_r290.py','client_quality_r290.py',
           'content_json.py','owner_r290.py','scope_plan_r328.py'}
ADDED = {'content_json.py','scope_plan_r328.py'}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_candidate(root, directory, source):
    root, source = Path(root), Path(source)
    parent = root/'incoming/GLM-prefill-W2-W4-owner-r307'
    inherited = verify_parent_pair(root, directory, parent)
    assert sha(parent/'SOURCE_MANIFEST.json') == PARENT_SHA
    assert sha(source/'SOURCE_MANIFEST.json') == OWNER_SHA
    old = json.loads((parent/'SOURCE_MANIFEST.json').read_bytes())
    new = json.loads((source/'SOURCE_MANIFEST.json').read_bytes())
    assert new['parent_manifest_SHA256'] == PARENT_SHA and set(new['changed_members']) == CHANGED
    assert set(new['files_SHA256']) == set(old['files_SHA256']) | ADDED
    assert new['Native_SHA256'] == old['Native_SHA256']
    assert new['HTTP'] == new['Native'] == 74 and new['phase_Native_end'] == {'multicall':4,'quality':68,'paired':74}
    for name, digest in old['files_SHA256'].items():
        assert sha(parent/name) == digest
        if name not in CHANGED:
            assert new['files_SHA256'][name] == digest and sha(source/name) == digest
    assert (source/'GPU_inside_r290.sh').read_text() == (parent/'GPU_inside_r290.sh').read_text().replace(
        'GLM-prefill-W2-W4-owner-r307','GLM-semantic74-owner-r328')
    parser = root/'incoming/Deep-semantic-content-r316/content_json.py'
    # The published parser may only live in the private Deep320 source snapshot.
    if not parser.exists(): parser = root/'incoming/Deep-semantic68-owner-r320/content_json.py'
    assert sha(source/'content_json.py') == sha(parser)
    package = root/'incoming/GLM-semantic74-CPU-r329b'
    assert sha(package/'SOURCE_MANIFEST.json') == CPU_PACKAGE_SHA
    cm = json.loads((package/'SOURCE_MANIFEST.json').read_bytes())
    assert cm['owner_source_manifest_SHA256'] == OWNER_SHA
    assert all(sha(package/n) == h for n,h in cm['files_SHA256'].items())
    receipt = root/'candidates/GLM-semantic74-CPU-r329b/CPU-result.json'
    assert sha(receipt) == CPU641_SHA
    q = json.loads(receipt.read_bytes())
    assert q['completed'] and q['passed'] and q['CPU_only'] and q['actual_CPU'] == '910407.641'
    assert q['source_manifest_SHA256'] == OWNER_SHA and q['CPU_source_SHA256'] == sha(package/'CPU_main.py')
    assert all(q[k] == 0 for k in ['GPU','Model','Native_dlopen','production_HTTP'])
    assert q['real_supervisor_and_client_mains_and_wire_parser'] and q['synthetic_owner_receipts_and_gold_answers']
    expected = [('positive-bare',74,0),('positive-fence',74,0),('wrong-manifest',0,1),('wrong-step',0,1),
                ('fence-prefix',2,1),('fence-suffix',2,1),('duplicate-key',2,1),('nonfinite',2,1),
                ('wrong-nonce',2,1),('bool-sum',2,1),('multicall-index',2,1),('quality-index',5,1),
                ('quality-clock',5,1),('quality-cap',68,1),('pair-window',71,1),('pair-prompt',70,1),('pair-tool',69,1)]
    assert [(c['name'],c['synthetic_HTTP'],c['returncode'])for c in q['cases']] == expected
    assert all(c['passed'] for c in q['cases'])
    return dict(inherited,CPU641_SHA256=CPU641_SHA,source_manifest_SHA256=OWNER_SHA,
                whole_client_entry_cases=17,new_runtime_qualification=False)
