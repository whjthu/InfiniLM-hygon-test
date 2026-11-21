"""Compare two private, real Deep CSA component captures without loading the Native library."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re


REQUEST = 'e828d67b3b94d2ad8338fd19a338e4b3'
FLAG = 'INFINILM_DEEP_CSA_IDENTITY_SWA_COMMIT'
WINDOW = 128
CONTEXT = 32768
SWA_BYTES = (256 * 584 + 575) // 576 * 576
STATE_FIELDS = ('swa', 'compressed', 'compressor_state', 'index_cache', 'indexer_state', 'selection', 'output')
IDENTITY_FIELDS = ('checkpoint', 'runtime_config', 'input_plan', 'capture_driver', 'other_environment')
HEX = re.compile(r'[0-9a-f]{64}')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def plan():
    steps = []

    def add(name, *requests):
        steps.append(dict(name=name, requests=[list(request) for request in requests],
                          capture=name in ('warm-t1', 'wrap-and-mixed', 'cropped-129', 'after-cropped', 'max-t1'),
                          direct=all(tokens <= WINDOW for _, _, tokens in requests)))

    def advance(past, end):
        while past < end:
            tokens = min(end - past, 512)
            add(f'prefill-{past}', (0, past, tokens))
            past += tokens

    advance(0, 8496)
    add('warm-t1', (0, 8496, 1))
    add('wrap-and-mixed', (1, 0, 1), (0, 8497, 128))
    add('cropped-129', (0, 8625, 129))
    add('after-cropped', (0, 8754, 1))
    advance(8755, CONTEXT - 1)
    add('max-t1', (0, CONTEXT - 1, 1))
    return dict(component='CsaAttention', layer=2, window=WINDOW, context=CONTEXT,
                dtype='BF16', hidden_size=4096, cache_rows=2, swa_page_bytes=SWA_BYTES,
                steps=steps, toggle=FLAG)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def record(path):
    need(path.is_file() and path.resolve(strict=True) == path, 'noncanonical receipt path')
    with path.open('rb') as stream:
        raw = stream.read(128 * 1024 + 1)
    need(len(raw) <= 128 * 1024, 'receipt exceeds size limit')
    value = json.loads(raw)
    need(type(value) is dict, 'receipt is not an object')
    return raw, value


def original_compile(path):
    raw, result = record(path)
    row = result.get('row')
    need(result.get('request') == REQUEST and result.get('state') == 'COMPILE_VERIFIED'
         and result.get('independent_observation') is True and result.get('replay_allowed') is False
         and isinstance(row, list) and len(row) == 7 and re.fullmatch(r'1087656\.[0-9]+', row[0])
         and row[1] == 'Deep-CSA-identity-SWA-commit-CPU-v1' and row[2:4] == ['COMPLETED', '0:0']
         and row[4].isascii() and row[4].isdecimal() and 0 < int(row[4]) <= 300
         and row[5] == 'f12r3n17'
         and set(row[6].split(',')) in ({'cpu=4', 'mem=8G', 'node=1'}, {'cpu=4', 'mem=8192M', 'node=1'})
         and all(HEX.fullmatch(result[key]) for key in ('library_SHA256', 'receipt_SHA256', 'object_SHA256')),
         'original TU07 zero-GRES compile has no independently verified output')
    return dict(compile_reconcile_SHA256=digest(raw), library_SHA256=result['library_SHA256'])


def snapshot(arm, index, field, reference):
    name = f'{index:03d}-{field}.bin'
    path = arm / name
    need(path.is_file() and path.resolve(strict=True) == path, f'{name} missing or noncanonical')
    with path.open('rb') as stream:
        count = path.stat().st_size
        need(type(reference['bytes']) is int and 0 < count == reference['bytes'] <= 32 * 1024**2,
             f'{name} bytes differ')
        if field == 'swa':
            need(count == SWA_BYTES, 'raw SWA page byte count differs')
        result = sha256()
        remaining = count
        while remaining:
            chunk = stream.read(min(remaining, 1024**2))
            need(bool(chunk), f'{name} shortened while reading')
            result.update(chunk)
            remaining -= len(chunk)
        need(not stream.read(1) and result.hexdigest() == reference['SHA256'], f'{name} digest differs')
    return count, result.hexdigest()


def compare(compile_path, off_path, on_path):
    spec = plan()
    steps = spec['steps']
    result = dict(passed=False, planned_arms=2, planned_steps_per_arm=len(steps),
                  steps_compared=0, raw_state_comparisons=0, actual_GPU_launches=0,
                  noninferiority_qualified=False, performance_qualified=False, Model_GO=False)
    try:
        binding = original_compile(compile_path)
        arm_paths = (off_path, on_path)
        need(off_path != on_path, 'capture paths must be distinct')
        records = [record(arm / 'capture.json')[1] for arm in arm_paths]
        for flag, arm in enumerate(records):
            need(all(arm.get(key) == value for key, value in binding.items())
                 and arm.get('resolved_forward_provider_SHA256') == binding['library_SHA256']
                 and arm.get('plan_SHA256') == digest(canonical(spec))
                 and arm.get('opt_in') == {FLAG: str(flag)}, 'arm binding or active symbol provider differs')
            need(arm.get('completed') is True and arm.get('normal_destroy') is True
                 and isinstance(arm.get('steps'), list) and len(arm['steps']) == len(steps),
                 'arm is incomplete')
            need(all(HEX.fullmatch(arm[key]) for key in IDENTITY_FIELDS)
                 and re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', arm['PCI'])
                 and re.fullmatch(r'[0-9]+\.[0-9]+', arm['slurm_step'])
                 and type(arm['pid']) is int and arm['pid'] > 0 and type(arm['pid_startticks']) is int
                 and arm['pid_startticks'] > 0, 'GPU, input identity or owner is malformed')
        off, on = records
        need(all(off[key] == on[key] for key in (*IDENTITY_FIELDS, 'node', 'PCI'))
             and (off['pid'], off['pid_startticks']) != (on['pid'], on['pid_startticks']),
             'arms must reuse the device and inputs in distinct processes')
        result['arms_bound'] = True
        for index, unit in enumerate(steps):
            observed = [arm['steps'][index] for arm in records]
            for flag, step in enumerate(observed):
                expected_selects = int(not (flag and unit['direct']))
                need(step['name'] == unit['name'] and step['requests'] == unit['requests']
                     and step.get('passed') is True and HEX.fullmatch(step['input_SHA256']),
                     f"{unit['name']} missing or input identity differs")
                if unit['capture']:
                    need(type(step['commit_IndexSelect_calls']) is int
                         and step['commit_IndexSelect_calls'] == expected_selects
                         and HEX.fullmatch(step['commit_callsite_trace_SHA256']),
                         f"{unit['name']} missing observed commit branch")
                    need(set(step['raw']) == set(STATE_FIELDS), f"{unit['name']} state evidence incomplete")
            need(observed[0]['input_SHA256'] == observed[1]['input_SHA256'], f"{unit['name']} inputs differ")
            if unit['capture']:
                for field in STATE_FIELDS:
                    left, right = (snapshot(path, index, field, step['raw'][field])
                                   for path, step in zip(arm_paths, observed, strict=True))
                    need(left == right, f"{unit['name']} {field} bytes differ")
                    result['raw_state_comparisons'] += 1
            result['steps_compared'] += 1
        result['passed'] = True
    except (OSError, TypeError, ValueError, KeyError, AttributeError) as error:
        result.update(error_type=type(error).__name__, error=str(error)[:160])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    for name in ('compile-reconcile', 'off', 'on'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    paths = (args.compile_reconcile, args.off, args.on)
    if args.plan and not any(paths):
        spec = plan()
        print(json.dumps(dict(plan=spec, plan_SHA256=digest(canonical(spec))), indent=2))
        return 0
    if args.plan or not all(paths):
        parser.error('use --plan alone, or pass all three receipt paths')
    result = compare(*(path.absolute() for path in paths))
    print(json.dumps(result, sort_keys=True))
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
