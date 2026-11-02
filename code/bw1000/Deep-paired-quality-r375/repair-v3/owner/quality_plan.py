"""Finite64+64 case plan, read existing frozen data without new raw copies."""
import hashlib
import json
import os
import pathlib

D = pathlib.Path(__file__).resolve().parent
R = pathlib.Path('/refactor')
NATIVE = '976f97b51e9f4b11dfb2439d2c97f2a1e7788e4de05ee8a8a0138f33e38b6ed1'
TOTAL = 66
WORK_SECONDS = 7080
MEMORY_FLOOR_BYTES = 16 * 1024**3
SYSTEM = {'GSM8K': 'Solve this math problem accurately. Explain your work and end with #### followed by your numeric answer.',
          'MMLU': 'Answer the multiple choice question accurately. Explain your choice and end with #### followed by exactly one of A, B, C, D.'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_data():
    refs = json.loads((D / 'CASE_REFS.json').read_bytes())
    sources = {}
    for source in refs['sources']:
        p = R / source['remote_relative']
        raw = p.read_bytes()
        assert digest(raw) == source['SHA256'], str(p)
        sources[source['repo_path']] = json.loads(raw)['cases']
    cases = {}
    for meta in refs['cases']:
        row = sources[meta['source_file']][meta['source_row']]
        assert digest(canonical(row)) == meta['row_canonical_SHA256']
        assert row['source_index'] == meta['source_index']
        cases[meta['case_id']] = dict(row, suite=meta['suite'], case_id=meta['case_id'])
        assert digest(canonical(messages(cases[meta['case_id']]))) == meta['messages_canonical_SHA256']
    assert len(cases) == 128
    warm = sources[refs['warm']['source_file']][refs['warm']['source_row']]
    assert digest(canonical(warm)) == refs['warm']['row_canonical_SHA256']
    cases['warm'] = dict(warm, suite='GSM8K', case_id='warm')
    return refs, cases


def messages(case):
    user = case['question']
    if case['suite'] == 'MMLU':
        user += '\n' + '\n'.join(chr(65 + i) + '. ' + c for i, c in enumerate(case['choices']))
    return [{'role': 'system', 'content': SYSTEM[case['suite']]}, {'role': 'user', 'content': user}]


def body(case):
    return {'messages': messages(case), 'temperature': 0, 'max_tokens': 4096,
            'stream': False, 'tool_choice': 'none', 'parallel_tool_calls': False,
            'chat_template_kwargs': {'enable_thinking': False}}


def selected_shard():
    text = os.environ['DEEP_R375_REPLICA']
    if text not in ['0', '1', '2', '3']:
        raise ValueError('Exact replica0..3 required')
    return int(text)


def plan(shard=None):
    shard = selected_shard() if shard is None else shard
    if type(shard) is not int or shard not in range(4):
        raise ValueError('Exact replica index required')
    refs = json.loads((D / 'CASE_REFS.json').read_bytes())
    rows = [{'case_id': 'warm', 'suite': 'GSM8K', 'arm': arm, 'warmup': True,
             'shard': shard, 'planned_index': arm} for arm in [0, 1]]
    rows += refs['shards'][shard]
    assert len(rows) == TOTAL
    for i, row in enumerate(rows):
        assert row['planned_index'] == i and row['shard'] == shard
    return rows


def descriptor_for_index(base, index):
    if type(index) is not int or index not in range(TOTAL):
        raise ValueError('Finite66 Native plan exhausted')
    arm = plan()[index]
    assert base['library_SHA256'] == NATIVE
    assert base['rows'] == 1 and base['capacity'] == 32768
    assert base['hc_epilogue_modes_supported'] == [0, 1]
    return dict(base, prefill_chunk=1024, hc_epilogue_mode=arm['arm'],
                prefill_arm_index=index, output_budget=4096, planned_case=arm)


def encoding():
    p = pathlib.Path(os.environ['DEEP_R375_ENCODING_PATH'])
    assert p.is_relative_to(R / 'candidates/Deep-paired-quality-r375')
    raw = p.read_bytes()
    assert digest(raw) == os.environ['DEEP_R375_ENCODING_SHA']
    value = json.loads(raw)
    assert value['completed'] and value['passed'] and value['actual_prepared_ingress']
    assert value['GPU'] == value['Model'] == 0 and not value['weights_loaded']
    assert value['CASE_REFS_SHA256'] == digest((D / 'CASE_REFS.json').read_bytes())
    assert value['comparison_contract']['output_budget'] == 4096
    assert value['comparison_contract']['context_capacity'] == 32768
    assert len(value['cases']) == 129
    return value
