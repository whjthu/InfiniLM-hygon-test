"""Real private VIEW bytes; synthetic installation metadata, no target/native execution."""
import argparse
import json
from pathlib import Path
import time

import stage_view
from stage_view import FILES, TREE, copy_view, pin, read_exact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--synthetic-only', action='store_true')
    args = p.parse_args()
    origin = time.monotonic()
    root = args.output.resolve()
    root.mkdir(mode=0o700)
    if args.synthetic_only:
        source = root / 'synthetic-source'
        source.mkdir()
        bodies = {'train800-swift.jsonl': b'{"synthetic_fixture":true}\n' * 800,
                  'exposure-ledger-private.json': b'{"synthetic_fixture":true}\n'}
        FILES.clear()
        FILES.update({name: pin(raw) for name, raw in bodies.items()})
        stage_view.FILES = FILES
        for name, raw in bodies.items():
            (source / name).write_bytes(raw)
    else:
        source = (args.repo / 'local-only/training-gemma-render72-exposure800-local-v1-original').resolve(strict=True)
    before = {n: pin(read_exact(source / n, expected)) for n, expected in FILES.items()}
    receipt = dict(original_base_files_read=393, installed_files_verified=396,
                   candidate_framework_tree=TREE, training_ready=False)
    tests = []
    def run(name, fn, rejected=False):
        try:
            value = fn()
        except (ValueError, FileExistsError) as error:
            tests.append(dict(name=name, passed=rejected, observed=type(error).__name__, reason=str(error)[:160]))
        else:
            tests.append(dict(name=name, passed=not rejected, observed='RETURNED'))
            return value
    target = root / 'real-view-copy-fixture'
    target.mkdir()
    run('synthetic_only_copy' if args.synthetic_only else 'real602794byte_view_and8196byte_ledger_copy',
        lambda: copy_view(source, target, receipt, lambda: None))
    run('existing_attempt_or_view_rejected', lambda: copy_view(source, target, receipt, lambda: None), True)
    bad = root / 'wrong-framework-fixture'
    bad.mkdir()
    run('wrong_framework_receipt_rejected', lambda: copy_view(source, bad, {**receipt, 'installed_files_verified': 393}, lambda: None), True)
    expired = root / 'expired-clock-fixture'
    expired.mkdir()
    def stop():
        raise ValueError('fixture_expired_original_clock')
    run('clock_exhausted_before_first_source_body', lambda: copy_view(source, expired, receipt, stop), True)
    changed = root / 'changed-source-fixture'
    changed.mkdir()
    for n in FILES:
        raw = read_exact(source / n, FILES[n])
        (changed / n).write_bytes((b'X' + raw[1:]) if n.endswith('jsonl') else raw)
    badcopy = root / 'changed-source-output'
    badcopy.mkdir()
    run('corrupt_view_fullSHA_rejected', lambda: copy_view(changed, badcopy, receipt, lambda: None), True)
    after = {n: pin(read_exact(source / n, expected)) for n, expected in FILES.items()}
    tests.append(dict(name='actual_source_unchanged', passed=before == after, observed='FULL_SHA_EQUAL'))
    result = dict(state='GEMMA_SYNTHETIC_IO_BOUNDARIES' if args.synthetic_only else 'GEMMA_REAL_VIEW_LOCAL_IO_BRIDGE_CHECK',
                  passed=all(t['passed'] for t in tests), tests=tests, tested_files=before,
                  synthetic_source=args.synthetic_only, actual_training_view_body_read=not args.synthetic_only,
                  local_view_copy=True,
                  synthetic_installation_metadata=True, actual_target_stage_calls=0,
                  actual_native_encodes=0, actual_updates=0, training_READY=False,
                  wall_seconds=time.monotonic() - origin)
    raw = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
    (root / 'result.json').write_bytes(raw)
    print(json.dumps(dict(passed=result['passed'], checks=len(tests), result=pin(raw)), sort_keys=True))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
