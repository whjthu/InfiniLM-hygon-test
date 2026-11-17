"""Complete local source/claim closure; no transport/SDK/Native/model execution."""
import ast
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
H = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    package = json.loads((D/'PACKAGE_FILES.json').read_bytes())
    for name, digest in package.items():
        assert Path(name).name == name and H(D/name) == digest
        raw = (D/name).read_bytes()
        if name.endswith('.py'):
            ast.parse(raw)
        assert all(line.rstrip(b' \t\r') == line.rstrip(b'\r') for line in raw.split(b'\n'))
    tree = ast.parse((D/'shared.py').read_bytes())
    required = next(node.value.args[0] for node in tree.body if isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'REQUIRED_FILES')
    required = set(ast.literal_eval(required))
    worker = json.loads((D/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256']
    host = json.loads((D/'HOST_MANIFEST.json').read_bytes())['files_SHA256']
    assert set(worker) == required
    assert all(H(D/name) == digest for manifest in [worker, host] for name,digest in manifest.items())
    for name, digest in {'INPUT_BINDING.json':'81f6b2d68228aafb79993306847b0edb04c1ba101a35ab21bb822ac96ca2ba5e',
                         'ENV.sh':'aa8d9619534a2395492d62dd7a5c6aa828539b28a389d00b98a389f80ab39d67',
                         'typed_binding.py':'ea014b57f1ac1b450ca5acc8cb28e41283d041b80f82e88141377fb005b86fb7',
                         'observer_trial.py':'4d00164482c7daa532f05cc15d5c89b3e0df98d1d8e166b30f8acc3cf1fcac56',
                         'slurm_accounting.py':'15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b'}.items():
        assert H(D/name) == digest
    claim = json.loads((D/'GPU_RESOURCE_CLAIM.json').read_bytes())
    assert claim['source_manifest_SHA256'] == H(D/'SOURCE_MANIFEST.json')
    assert claim['host_manifest_SHA256'] == H(D/'HOST_MANIFEST.json')
    assert claim['native_arms'] == {'socket':'02b0b1e8cf529fcb73ad6bd429c78d158beb6d19d4577866977abe97f686136d'}
    assert claim['model_context'] == 32768 and claim['memory_floor_bytes'] == 8*1024**3
    assert claim['whole_seconds'] == 60000 and claim['GPU_seconds'] == 57600
    assert claim['planned_requests_per_arm'] == 66 and claim['max_new_tokens'] == 4096
    assert claim['nodes'] == ['f13r1n05','f13r1n07'] and claim['parent'] == '1103606'
    assert claim['planned_requests_per_math_arm'] == 33 and claim['planned_screen_per_math_arm'] == 32
    from resident_trial import schedule, phases
    rows=schedule();trial=json.loads((D/'TRIAL.json').read_bytes())
    assert len(rows)==66 and trial['schedule']==rows and trial['phases']==phases()
    assert [r['arm'] for r in rows]==['OFF','ON']*33
    assert [i for i,r in enumerate(rows) if r['diagnostic']]==[1]
    assert sum(r['original_diagnostic'] for r in rows)==2
    assert len(set(phases()))==len(phases())==136
    for arm in ('OFF','ON'):
        assert sorted(r['screen_index'] for r in rows if r['arm']==arm and r['screen_index'] is not None)==list(range(32))
    inputs=json.loads((D/'INPUT_BINDING.json').read_bytes())
    assert all(p['prompt_tokens']+4096<=32768 for p in inputs['private_plan']['requests'])
    providers = json.loads((D/'PROVIDER_BINDING.json').read_bytes())
    assert len(providers['providers']) == providers['count'] == 71
    event = json.loads((D/'EVENT_BUDGET.json').read_bytes())
    assert sum(row['max_new_marks'] for row in event['new_mark_rows']) == 162
    assert event['old_caller_upper_bound']+162 == event['future_caller_upper_bound'] == 440 < 512
    ledger = json.loads((D/'RESOURCE_BINDING.json').read_bytes())
    assert ledger['group'] == 'I02' and set(claim['nodes']) <= set(ledger['record']['allocations']['1103606'])
    print(json.dumps(dict(passed=True, package_files=len(package), worker_files=len(worker), host_files=len(host),
                          package_manifest_SHA256=H(D/'PACKAGE_FILES.json'), actual_target_calls=0)))


if __name__ == '__main__':
    main()
