"""Freeze new GSM8K questions and identical W2/W4 request bodies; no device access."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
from fractions import Fraction

SOURCE_SHA = '3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14'
SEED = 20261103
COUNT = 64
SYSTEM = 'Solve this math problem accurately. Explain your work and end with #### followed by your numeric answer.'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


def freeze(workspace, destination):
    base = Path(__file__).resolve().parent
    prior = json.loads((base / 'PRIOR_INPUTS.json').read_bytes())
    raw = (workspace / 'gsm8k-test-source-r9100.jsonl').read_bytes()
    assert digest(raw) == SOURCE_SHA
    source = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(source) == 1319 and len({r['question'] for r in source}) == 1319
    by_question = {r['question']: i for i, r in enumerate(source)}
    excluded, evidence = set(), []
    for item in prior:
        data = (workspace / item['workspace_relative_path']).read_bytes()
        assert digest(data) == item['SHA256'], item['workspace_relative_path']
        parsed = json.loads(data)
        cases = parsed.get('cases', [])
        matched = {by_question[c['question']] for c in cases if c.get('question') in by_question}
        excluded.update(matched)
        evidence.append(dict(**item, matching_official_test_questions=len(matched)))
    order = list(range(len(source)))
    random.Random(SEED).shuffle(order)
    selected = [i for i in order if i not in excluded][:COUNT]
    assert len(selected) == len(set(selected)) == COUNT and not set(selected) & excluded
    # Selection is complete before examining any answers or question lengths.
    cases, requests, schedule = [], [], []
    for j, i in enumerate(selected):
        row = source[i]
        final = re.findall(r'####\s*([-+]?[0-9][0-9,.]*(?:/[0-9]+)?)', row['answer'])
        assert len(final) == 1
        gold = final[0].replace(',', '')
        Fraction(gold)
        case_id = 'gsm8k-test-%d' % i
        cases.append(dict(id=case_id, source_index=i, question=row['question'],
                          gold_answer=row['answer'], gold_numeric=gold))
        body = dict(model='GLM-5.3', messages=[dict(role='system', content=SYSTEM),
                    dict(role='user', content=row['question'])], temperature=0,
                    max_tokens=4096, stream=False, tool_choice='none', parallel_tool_calls=False,
                    chat_template_kwargs=dict(enable_thinking=False, clear_thinking=False))
        assert body['messages'][1]['content'] == source[i]['question']
        requests.append(dict(case_id=case_id, body=body, compact_body_SHA256=digest(compact(body))))
        # Counterbalance order by shuffled ordinal, independently of answers.
        arms = [2, 4] if j % 2 == 0 else [4, 2]
        schedule.extend(dict(index=2 * j + k, case_id=case_id, window=w,
                             compact_body_SHA256=requests[-1]['compact_body_SHA256'])
                        for k, w in enumerate(arms))
    assert [x['index'] for x in schedule] == list(range(2 * COUNT))
    assert sum(x['window'] == 2 for x in schedule) == COUNT
    assert sum(x['window'] == 4 for x in schedule) == COUNT
    destination.mkdir(parents=True, exist_ok=False)
    files = {
        'gsm8k-new64.json': dict(suite='GSM8K official test disjoint frozen64 custom0shot paired screen',
            source_revision='3101c7d5072418e28b9008a6636bde82a006892c', source_SHA256=SOURCE_SHA,
            source_count=len(source), seed=SEED,
            selection='Random(seed).shuffle all1319 indices; first64 outside exact prior-question exclusions; no answer or length filtering',
            excluded_source_indices=sorted(excluded), exclusion_evidence=evidence, cases=cases,
            exclusion_scope='Only exact question matches from the enumerated, SHA-bound prior fixtures; not a claim to have audited every historical request.',
            full_benchmark=False, no_regression_proven=False),
        'requests-and-pairs.json': dict(requests=requests, comparison_schedule=schedule,
            engine_requests=128, warmups_in_this_file=0, GPU_GO=False,
            policy='Same complete question/template/greedy/output4096 body for both windows; experimental arms live outside request bodies. Real checkpoint encoding/profile consumer and resource claim are still required.',
            thinking='Requested enable_thinking=False; existing GLM checkpoint template may still open reasoning. Never claim thinking-toggle qualification.',
            quality='Score final content numeric answer; report naturalEOS, parse failures, truncation, completed coverage, paired accuracy difference and uncertainty. A length cap is not natural EOS. Do not stop merely because an answer is wrong.',
            timing='Same input across W2/W4; report each complete request, backend emitter TTFT/first-last separately; buffered HTTP is not client incremental SLO.'),
    }
    hashes = {}
    for name, value in files.items():
        data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
        (destination / name).write_bytes(data)
        hashes[name] = digest(data)
    (destination / 'MANIFEST.json').write_bytes((json.dumps(dict(
        source_SHA256=SOURCE_SHA, freezer_SHA256=digest(Path(__file__).read_bytes()),
        prior_inputs_SHA256=digest((base / 'PRIOR_INPUTS.json').read_bytes()),
        files_SHA256=hashes, GPU_GO=False), indent=2) + '\n').encode())
    return dict(selected=COUNT, excluded=len(excluded), requests=len(schedule), files_SHA256=hashes)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(freeze(args.workspace, args.output)))
