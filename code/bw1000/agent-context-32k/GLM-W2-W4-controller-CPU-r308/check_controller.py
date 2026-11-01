"""Pure CPU scope, handle binding and active-predecessor rejection checks."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile


def rejected(fn):
    try: fn()
    except (AssertionError, PermissionError, ValueError, KeyError, TypeError): return True
    raise AssertionError('Invalid scope/evidence accepted')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    a = p.parse_args()
    repo = a.repo.resolve()
    d = repo / 'code/bw1000/public-native-bridge/GLM-prefill-W2-W4-controller-r308'
    mf = hashlib.sha256((d / 'SOURCE_MANIFEST.json').read_bytes()).hexdigest()
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location('controller308_CPU', d / 'controller_r308.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    import candidate_binding
    g = json.loads((d / 'execution-GO.template.json').read_bytes())
    assert g['GPU_authorized'] is False and g['baseline_result_consumed'] is False
    rejected(lambda: module.validate_scope(g))
    g.update(GPU_authorized=True, publication_commit='a' * 40,
             controller_manifest_SHA256=mf, baseline_result_consumed=True,
             baseline_result_record='results/bw1000/CPU_fixture_not_execution.json')
    module.validate_scope(g)
    invalid = {'GPU_authorized': False, 'publication_branch': 'main', 'publication_commit': 'x',
               'controller_manifest_SHA256': 'x', 'owner_manifest_SHA256': 'a' * 64,
               'GPU_parent': 910374, 'GPU_nodes': list(reversed(g['GPU_nodes'])),
               'CPU_keeper_parent': 910380, 'CPU_keeper_node': 'a14r2n08',
               'cards': 16, 'TP': 4, 'PP': 2, 'context': 16384, 'ordinary_budget': 2048,
               'thinking_budget': 4096, 'max_HTTP': 7, 'max_Native': 5,
               'GPU_max_s': 29401, 'Native_max_s': 28801, 'CPU_keeper_max_s': 32401,
               'one_attempt': False, 'no_retry_reload': False,
               'DSO_CPU621_receipt_SHA256': 'a' * 64,
               'pair_CPU628_receipt_SHA256': 'a' * 64, 'wrapper_SHA256': 'a' * 64,
               'namespace_CPU_receipt_SHA256': 'a' * 64,
               'baseline_result_consumed': False, 'baseline_result_record': '/tmp/unpublished'}
    for key, value in invalid.items():
        bad = copy.deepcopy(g); bad[key] = value
        rejected(lambda: module.validate_scope(bad))
    for key in ['max_HTTP', 'max_Native', 'cards', 'GPU_parent']:
        bad = copy.deepcopy(g); bad[key] = float(g[key])
        rejected(lambda: module.validate_scope(bad))
    expected = ['910376.999', module.LABEL, 'RUNNING', '0:0', 'a02r4n[12-15]']
    module.validate_account_row(expected, module.NODES)
    for row, nodes in [(['910374.999', *expected[1:]], module.NODES),
                       ([expected[0], 'other', *expected[2:]], module.NODES),
                       (expected, module.NODES[:-1]), (expected, [*module.NODES[:-1], 'a02r4n01'])]:
        rejected(lambda: module.validate_account_row(row, nodes))
    local = repo / 'local-only'; local.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='controller308-CPU-', dir=local) as temporary:
        root = Path(temporary).resolve()
        assert root.is_relative_to(local.resolve())
        path = root / 'results/GLM-latest99-full-Model-r302-watch.json'
        path.parent.mkdir()
        baseline = {'actual_GPU': '910376.816', 'completed': True, 'post_empty': True,
                    'terminal': ['910376.816', 'GLM-latest99-full-Model-r302', 'COMPLETED', '0:0']}
        path.write_text(json.dumps(baseline)); candidate_binding.require_baseline_terminal(root, g)
        for change in [{'completed': False}, {'post_empty': False}, {'actual_GPU': '910376.999'},
                       {'terminal': ['910376.816', 'GLM-latest99-full-Model-r302', 'RUNNING', '0:0']}]:
            bad = dict(baseline, **change); path.write_text(json.dumps(bad))
            rejected(lambda: candidate_binding.require_baseline_terminal(root, g))
    print(json.dumps({'passed': True, 'CPU_only': True, 'controller_manifest_SHA256': mf,
                      'scope_negative_cases': len(invalid) + 4, 'accounting_negative_cases': 4,
                      'predecessor_running_wrong_handle_or_unclean_negative_cases': 4,
                      'template_remains_false': True, 'fixture_not_saved_as_GO': True,
                      'GPU': 0, 'Native': 0, 'jobs_submitted': 0}))


if __name__ == '__main__': main()
