"""Read-only stdlib audit of the original CPU846 semantic diagnostic archive."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

RAW_SHA = 'ae1f1b4643b94e47c1db7cc423b8439d5c3352c9737698d847fd4788aae96fee'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def floats(record):
    return [struct.unpack('<f', struct.pack('<I', b))[0] for b in record['F32_bits']]


def category(bits):
    exponent = (bits >> 23) & 255
    fraction = bits & 0x7fffff
    if exponent == 255:
        return 'NaN' if fraction else 'infinity'
    return ('subnormal' if fraction else 'zero') if exponent == 0 else 'normal'


def audit_bits(value):
    count = 0
    if isinstance(value, dict):
        if 'F32_bits' in value:
            assert value['dtype'] == 'float32'
            assert math.prod(value['shape']) == len(value['F32_bits'])
            payload = b''.join(struct.pack('<I', b) for b in value['F32_bits'])
            assert sha(payload) == value['raw_SHA256']
            count += 1
        for child in value.values():
            count += audit_bits(child)
    elif isinstance(value, list):
        count += sum(audit_bits(child) for child in value)
    return count


def analyze(path):
    archive_bytes = Path(path).read_bytes()
    archive = json.loads(archive_bytes)
    for item in archive['files']:
        payload = item['text'].encode('utf-8')
        assert len(payload) == item['bytes'] and sha(payload) == item['SHA256']
    item = next(x for x in archive['files'] if x['path'].endswith('/diagnostic.json'))
    assert item['SHA256'] == RAW_SHA and item['bytes'] == 914366
    q = json.loads(item['text'])
    assert q['completed'] is True and q['diagnostic_execution_passed'] is True
    assert q['original_numerical_predicates_passed'] is False
    assert q['actual_CPU'] == '910407.846'
    for flag in ('new_compile', 'Native_dlopen', 'Runtime_created', 'GPU_initialized', 'weights_opened', 'GPU_qualification', 'Model_qualification'):
        assert q[flag] is False or q[flag] == 0
    assert archive['accounting'] == ['910407.846|Deep419-semantic-diag-CPU-v2|COMPLETED|0:0|00:00:04|a14r2n20']
    assert len(q['cases']) == 14 and len({x['case'] for x in q['cases']}) == 14
    failed = [x['case'] for x in q['cases'] if not x['original_passed']]
    assert failed == ['negative_underflow']
    arrays_checked = audit_bits(q)
    rows = []
    for c in q['cases']:
        if c['margin']['separated']:
            assert c['margin']['same_expert_set']
        rows.append({'case': c['case'], 'original_passed': c['original_passed'], 'margin': c['margin'],
                     'weights_maxabs_at_same_candidate_IDs': c['maxabs_weights_at_same_candidate_IDs']})
    c = next(x for x in q['cases'] if x['case'] == 'negative_underflow')
    ids = c['NumPy_candidate']['ids']
    assert ids == c['reference']['ids'] == [50, 89, 155, 60, 220, 195]
    assert c['NumPy_candidate']['selection']['F32_bits'] == c['reference']['selection']['F32_bits']
    selected = c['selected_logit_and_subnormal_diagnostics']
    assert selected['NumPy_standalone_exp_probe_at_candidate_IDs']['F32_bits'][3] == 1
    assert selected['Torch_standalone_exp_probe_at_candidate_IDs']['F32_bits'][3] == 1
    index = ids[3]
    assert c['softplus_NumPy_candidate']['F32_bits'][index] == 1
    assert c['softplus_reference']['F32_bits'][index] == 0
    weights = floats(c['NumPy_candidate']['weights'])
    expected = floats(c['normalization_expected_at_candidate_IDs'])
    assert all(math.isfinite(x) for x in weights + expected)
    maxabs = max(abs(a-b) for a, b in zip(weights, expected))
    assert maxabs == c['maxabs_weights_at_same_candidate_IDs'] == 8.151907240971923e-05
    probe = q['FTZ_probe_only_not_CPU_register_read']
    assert probe['input_F32_bits'] == probe['numpy_times_one_bits'] == probe['Torch_times_one_bits']
    return {
        'id': 'Deep419-CPU846-semantic-diagnostic', 'reader_passed': True,
        'diagnostic_execution_passed': True, 'original_numerical_predicates_passed': False,
        'original_CPU845_failure_preserved': True, 'actual_CPU': q['actual_CPU'],
        'accounting': archive['accounting'], 'collection_seconds': q['seconds'],
        'raw_diagnostic': {k: item[k] for k in ('path', 'bytes', 'SHA256')},
        'archive_SHA256': sha(archive_bytes),
        'publication_commit': q['publication_commit'],
        'original_source_manifest_SHA256': q['original_source_manifest_SHA256'],
        'diagnostic_source_manifest_SHA256': q['diagnostic_source_manifest_SHA256'],
        'F32_raw_arrays_hash_checked': arrays_checked, 'case_count': 14, 'original_pass_count': 13,
        'failing_cases': failed, 'cases': rows,
        'underflow': {
            'same_IDs': ids, 'selection_bits_equal': True, 'margin_6_7': c['margin']['baseline_6_7'],
            'selected_logit': floats(selected['candidate_selected_logits'])[3],
            'standalone_exp_both_bits': 1, 'Torch_CPU_softplus_bits': 0, 'NumPy_softplus_bits': 1,
            'Torch_CPU_softplus_class': category(0), 'NumPy_softplus_class': category(1),
            'reference_sum6': floats(c['sum_reference_at_candidate_IDs'])[0],
            'NumPy_sum6': floats(c['sum_NumPy_candidate_tree'])[0],
            'reference_weights_at_candidate_IDs': expected, 'NumPy_weights': weights,
            'weights_maxabs': maxabs, 'all_weights_finite': True},
        'CPU_fegetround_readonly': q['CPU_fegetround_readonly'], 'tiny_multiply_probe_bits': probe['input_F32_bits'],
        'interpretation': 'The original CPU-only failure is localized to tiny Softplus/libmath results and normalization, with identical margin-separated routes. Standalone exp is a diagnostic, not proof of the internal Torch Softplus primitive. This does not establish global FTZ or ATen GPU/libdevice error.',
        'unchanged_original_predicates': True, 'new_compile': 0, 'SDK_default_active_metadata_executed': False,
        'actual_CXX_math_qualification': False, 'GPU_qualification': False, 'Model_qualification': False,
        'next_action': 'Parent review of this evidence before any changed acceptance/compile claim; do not replay CPU845/846 or infer device qualification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive')
    parser.add_argument('--out')
    args = parser.parse_args()
    output = json.dumps(analyze(args.archive), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.out:
        Path(args.out).write_text(output, encoding='utf-8')
    else:
        print(output, end='')
