"""Audit answer syntax on the independently verified GLM906 prefix in place."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import runpy

R = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
RESULT = R / 'results/GLM474-original906-failed-prefix-consumption-v2/private-result.json'
OWNER = R / 'results/GLM-global-TP16-C1-observer-screen32-r474-v1/owner-00.json'
PLAN = R / 'results/GLM-r474-C1-observer-screen32-v1-private-plan.json'
SCORER = R / 'incoming/GLM-C1-layer-observer-r474/full-model-screen32-v1/scoring.py'
PREFIX_SHA = '9e60449833c85a04fe4d1cdfbb87439745d3f23a5b4e861ad77becbb543e754a'
SCORER_SHA = 'c9c39a5fc185ed6bfc5f48c2b547d4ef36a4cfed351626835793fcf0ab5b521c'
PATTERNS = {
    'answer_marker_with_markdown': re.compile(
        r'(?i)\s*(?:\*\*)?(?:final answer|answer)(?:\*\*)?\s*:\s*(?:\*\*)?[\[(]?([ABCD])[\])]?(?:\*\*)?\s*[.!]?\s*'),
    'answer_is_line': re.compile(
        r'(?i)\s*(?:\*\*)?(?:the )?(?:final )?answer\s+is\s*[:：]?\s*(?:\*\*)?[\[(]?([ABCD])[\])]?(?:\*\*)?\s*[.!]?\s*'),
    'bare_markdown_letter': re.compile(r'\s*\*\*([A-Da-d])\*\*\s*[.!]?\s*'),
}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path, pin):
    raw = path.read_bytes()
    if isinstance(pin, dict):
        assert len(raw) == pin['bytes']
        pin = pin['SHA256']
    assert sha(raw) == pin
    return json.loads(raw)

def shadow_last_line(text):
    last = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), '')
    if not last:
        return 'empty', None
    for name, pattern in PATTERNS.items():
        match = pattern.fullmatch(last)
        if match:
            return name, match[1].upper()
    if re.search(r'(?i)\b(?:final answer|answer)\b', last):
        return 'answer_cue_without_recognized_single_letter', None
    return 'no_terminal_answer_cue', None

def main():
    report = read(RESULT, PREFIX_SHA)
    assert report['independently_validated_prefix'] == 31 and not report['cross_errors']
    assert report['engineering_full_entry_passed'] is False
    owner_pin, = [p for p in report['original_artifacts'] if p['name'] == OWNER.name]
    owner = read(OWNER, owner_pin)
    plan = read(PLAN, report['private_plan_SHA256'])
    assert len(plan['requests']) == 32
    raw = SCORER.read_bytes()
    assert sha(raw) == SCORER_SHA
    scorer = runpy.run_path(str(SCORER), run_name='frozen_glm906_scorer')
    planned = [i for i, q in enumerate(plan['requests']) if q['suite'] == 'mmlu']
    assert len(planned) == 16
    observed = {r['screen_index']:r for r in report['root_requests'] if r['suite']=='mmlu' and not r['diagnostic']}
    assert set(observed) <= set(planned) and len(observed) == 14
    counts = Counter()
    signatures = Counter()
    for i in planned:
        row = observed.get(i)
        if row is None:
            counts['missing_or_unvalidated'] += 1
            continue
        request = owner['requests'][i+1]
        assert row['index']==i+1 and row['ordinal']==request['ordinal']==plan['requests'][i]['ordinal']
        assert row['output_IDs_SHA256']==request['generated_token_ids_SHA256']
        content = request['output_text']
        grade = scorer['grade'](plan['requests'][i]['case'],content,
                    natural_EOS=request['natural_EOS'],cap_hit=request['cap_hit'],unexpected_tool=False)
        assert grade==request['score']
        assert all(grade[k]==row[k] for k in ('correct','parse_failure','cap_hit','natural_EOS'))
        counts['validated'] += 1
        counts['original_correct'] += int(grade['correct'])
        counts['cap_hit'] += int(grade['cap_hit'])
        if not grade['parse_failure']:
            counts['parsed_but_wrong_or_incomplete'] += int(not grade['correct'])
            continue
        counts['parse_failure'] += 1
        counts['parse_failure_and_cap'] += int(grade['cap_hit'])
        kind, choice = shadow_last_line(content)
        terminal = 'cap' if grade['cap_hit'] else 'EOS' if grade['natural_EOS'] else 'other'
        signatures[terminal+'/'+kind] += 1
        if choice is not None:
            counts['shadow_answer_present'] += 1
            counts['shadow_gold_match_with_natural_EOS'] += int(
                choice == plan['requests'][i]['case']['gold_choice']
                and grade['natural_EOS'] and not grade['cap_hit'])
    assert (counts['original_correct'], counts['parse_failure'],counts['cap_hit']) == (7,6,2)
    print(json.dumps(dict(planned_MMLU=16, validated=14, original_score=dict(counts),
        parse_failure_last_line_by_stop=dict(signatures), source_pins=dict(prefix=PREFIX_SHA,
            owner=owner_pin['SHA256'],plan=report['private_plan_SHA256'],scorer=SCORER_SHA),
        official_score_changed=False,raw_text_exported=False,GPU_steps=0),sort_keys=True))

if __name__ == '__main__':
    main()
