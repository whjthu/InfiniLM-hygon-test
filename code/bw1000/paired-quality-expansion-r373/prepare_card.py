"""Freeze metadata and finite paired plans from existing public dataset snapshots.

This CPU-only preparation does not tokenize, import a model, submit a job, or score
results. Raw questions/gold remain in their existing source files. r374 owns the
paired quality/coverage consumer.
"""
import argparse
import hashlib
import json
import pathlib
import random
import statistics

G = pathlib.Path(__file__).resolve().parents[3]
SEED = 20261102
SOURCES = [
    ('GSM8K', 'evidence/bw1000/quality-expanded-r9231/gsm8k-disjoint256.json',
     '405733dac31124148fdce613780e87090667ebf1b26585d1eabcdc89bff10294'),
    ('MMLU', 'code/bw1000/public-native-bridge/GLM-latest-full-owner-r290/benchmark/mmlu-fixed32.json',
     'b2bd725774a5ee3ccdb897e287c06c6831abcf0eabd338b6701c4e5d8698bac2'),
    ('MMLU', 'evidence/bw1000/confirmed-components-and-concurrency/GLM-real-concurrency-private-r9132/quality-bench-mmlu-r9132.json',
     '89cb7e9cefb849c61a91966ada5070a675ae3f3654a6881d0eba2e2d30ecd2ac'),
]
SYSTEM = {
    'GSM8K': 'Solve this math problem accurately. Explain your work and end with #### followed by your numeric answer.',
    'MMLU': 'Answer the multiple choice question accurately. Explain your choice and end with #### followed by exactly one of A, B, C, D.',
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def pin(path):
    p = G / path
    return {'path': path, 'SHA256': digest(p.read_bytes())}


def prepare():
    suites = {'GSM8K': [], 'MMLU': []}
    sources = []
    for suite, name, expected in SOURCES:
        raw = (G / name).read_bytes()
        assert digest(raw) == expected, name
        d = json.loads(raw)
        rows = d['cases'][:64] if suite == 'GSM8K' else d['cases']
        assert len(rows) == (64 if suite == 'GSM8K' else 32)
        sources.append({'path': name, 'SHA256': expected, 'selected_count': len(rows),
                        'selection': 'first64 in existing frozen order' if suite == 'GSM8K' else 'all32 in existing frozen order',
                        'source_selection': d['selection'], 'seed': d['seed'],
                        'repository_revision': d.get('source_revision', d.get('repository_revision'))})
        for i, row in enumerate(rows):
            question = row['question']
            user = question if suite == 'GSM8K' else question + '\n' + '\n'.join(chr(65 + j) + '. ' + c for j, c in enumerate(row['choices']))
            messages = [{'role': 'system', 'content': SYSTEM[suite]}, {'role': 'user', 'content': user}]
            suites[suite].append({'suite': suite, 'source_index': row['source_index'],
                                  'case_id': suite + ':' + str(row['source_index']),
                                  'source_file': name, 'source_row': i,
                                  'row_canonical_SHA256': digest(canonical(row)),
                                  'question_SHA256': digest(question.encode()),
                                  'messages_canonical_SHA256': digest(canonical(messages)),
                                  'subject': row.get('subject'),
                                  'gold_canonical_SHA256': digest(canonical(row.get('gold_numeric', row.get('answer'))))})
    for rows in suites.values():
        assert len(rows) == 64 and len({r['source_index'] for r in rows}) == 64
    historical = json.loads((G / 'results/bw1000/2026-10-28/deep1168-real-batch-frontier.json').read_text())
    used = {r['id'] for r in historical['by_batch']['32']['user_metrics']}
    known_overlap = [r['source_index'] for r in suites['GSM8K'] if 'gsm8k-test-' + str(r['source_index']) in used]
    plans = {}
    for model, replicas in [('GLM', 2), ('Deep', 4)]:
        shards = [[] for _ in range(replicas)]
        for task_number, (suite, rows) in enumerate(suites.items()):
            order = list(rows)
            random.Random(SEED + task_number).shuffle(order)
            for shard in range(replicas):
                cases = order[shard::replicas]
                labels = [False, True] * (len(cases) // 2)
                random.Random(SEED + 100 + task_number * 10 + shard).shuffle(labels)
                for row, on_first in zip(cases, labels):
                    for arm in ([1, 0] if on_first else [0, 1]):
                        shards[shard].append({'case_id': row['case_id'], 'suite': suite,
                                              'arm': arm, 'messages_canonical_SHA256': row['messages_canonical_SHA256']})
        for shard, rows in enumerate(shards):
            for i, row in enumerate(rows):
                row.update(shard=shard, planned_index=i + 2, warmup=False)
            assert len(rows) == 256 // replicas
            for i in range(0, len(rows), 2):
                assert rows[i]['case_id'] == rows[i + 1]['case_id']
                assert {rows[i]['arm'], rows[i + 1]['arm']} == {0, 1}
            for suite in suites:
                pairs = [r for i, r in enumerate(rows) if i % 2 == 0 and r['suite'] == suite]
                assert sum(r['arm'] for r in pairs) * 2 == len(pairs)
        assert len({r['case_id'] for rows in shards for r in rows}) == 128
        plans[model] = {'replicas': replicas, 'cards_per_replica': 32 if model == 'GLM' else 8,
                        'eval_HTTP': 256, 'warm_HTTP_per_replica': 2,
                        'total_HTTP_including_warm': 256 + 2 * replicas,
                        'warmup': 'Fixed GSM source0 OFF then ON per replica, excluded from scores; not implemented/tokenized yet',
                        'shards': shards}
    costs = {}
    for model, f in [('GLM', 'GLM819-terminal68-requests-two-quality-caps.json'),
                     ('Deep', 'Deep1221-semantic68-lifecycle-and-phases.json')]:
        name = 'evidence/bw1000/' + f
        d = json.loads((G / name).read_text())
        q = d['cause']['supervisor']['phases']['quality']['result'] if model == 'GLM' else d['phases']['quality']['result']
        tasks = {}
        for suite, result in q['suites'].items():
            rows = result['rows']
            tasks[suite] = {'pilot_n': len(rows), 'pilot_HTTP_s': sum(r['client_HTTP_wall_s'] for r in rows),
                            'pilot_output_tokens_mean': statistics.mean(r['generated_tokens'] for r in rows),
                            'pilot_prompt_minmax': [min(r['prompt_tokens'] for r in rows), max(r['prompt_tokens'] for r in rows)],
                            'pilot_caps': result['caps'], 'pilot_parse_failures': result['parse_failures']}
        eval_s = sum(x['pilot_HTTP_s'] for x in tasks.values()) * 4
        replicas = plans[model]['replicas']
        loading_s = 238.1685797846876 if model == 'GLM' else 375.32815242325887
        warm_s = 2 * tasks['GSM8K']['pilot_HTTP_s'] / 32
        costs[model] = {'source': pin(name), 'tasks': tasks, 'scaling': '64 per task *2 arms /32 pilot =4; extrapolation, not a run guarantee',
                        'estimated_total_serial_eval_s': eval_s, 'replicas': replicas,
                        'historical_loading_per_replica_s': loading_s,
                        'estimated_replica_wall_including_two_warm_and_load_s': eval_s / replicas + loading_s + warm_s,
                        'estimated_GPU_hours_including_load_and_warm': (eval_s + replicas * (loading_s + warm_s)) * plans[model]['cards_per_replica'] / 3600,
                        'proposed_GPU_ceiling_per_replica_s': 21600 if model == 'GLM' else 7200,
                        'cleanup_margin_in_estimate_s': 0, 'worst_case_4096_cap_completion_guaranteed': False}
    paths = [
        'code/bw1000/GLM-W4-cross-workload-r349/owner/owner_r290.py',
        'code/bw1000/GLM-W4-cross-workload-r349/owner/prepared_family_entry_r268.py',
        'code/bw1000/GLM-W4-cross-workload-r349/CPU_encode_fixtures_v3.py',
        'code/bw1000/public-native-bridge/GLM-prefill-W2-W4-owner-r307/client_quality_r290.py',
        'code/bw1000/Deep-HC-T1-Model-r366/owner-v2/worker_r206b.py',
        'code/bw1000/Deep-HC-T1-Model-r366/owner-v2/native_engine_bridge_r268.py',
        'code/bw1000/Deep-HC-T1-Model-r366/owner-v2/hc_model_contract.py',
        'code/bw1000/Deep-HC-T1-Model-r366/owner-v2/chunk_contract.py',
        'code/bw1000/Deep-HC-T1-Model-r366/owner-v2/NATIVE_BUILD_PIN.json',
        'code/bw1000/Deep-HC-T1-Model-r366/owner-v2/prepared_family_entry_r268.py',
        'code/bw1000/Deep-HC-T1-Model-r366/owner-v2/wire_client_r206b.py',
        'code/bw1000/public-native-bridge/Deep-semantic68-owner-r320/client_quality_r281.py',
    ]
    return {'schema': 1, 'card': 'paired-quality-expansion-r373', 'completed_preparation': True,
            'GPU_GO': False, 'GPU_submitted': False, 'CPU_jobs_submitted': 0,
            'scope': 'One proposed custom zero-shot64 GSM8K +64 MMLU paired sample per model; not official full benchmark or unseen heldout',
            'sources': sources, 'cases': suites, 'plans': plans, 'costs': costs,
            'license_evidence': {'GSM8K': pin('evidence/bw1000/quality-expanded-r9231/LICENSE-gsm8k.txt'),
                                 'MMLU': 'Original32 metadata says MIT per publisher dataset card; no independent local license text found; no new raw redistribution in this card'},
            'contract': {'temperature': 0, 'max_tokens': 4096, 'context': 32768,
                         'precision_and_checkpoint': 'unchanged current actual owner per model',
                         'stream': False, 'thinking_requested': False,
                         'GLM_actual_template': 'always opens think; requestFalse not a thinking toggle qualification',
                         'selection': 'Existing file order and seed only, no answer/length/score conditioning; no padding/truncation',
                         'off_on': {'GLM': 'explicit per-request W2/W4, decode unchanged', 'Deep': '976f HC mode0/1; fixed prefill1024; tokens==1 eligibility also includes prefillT1 tail'},
                         'counts': 'planned/attempted/completed start0; immutable arm/global/shard indices; no retries; unfinished in denominator',
                         'score_consumer': 'root r374; preserve naturalEOS-required original scorer and report parse/cap/error/missing separately',
                         'timing': 'Per-request HTTP and Native-emitter full token timeline including prefill; buffered HTTP not incremental/GPUbusy',
                         'performance_scope': 'GLM pilot prompts84..529 do not exercise W2/W4 differences; actual current encoding and effective W4 coverage still required'},
            'known_history': {'GSM_first64_overlap_Deep1168_actual64_count': len(known_overlap),
                              'GSM_overlap_source_indices': known_overlap,
                              'MMLU_original32': 'GLM819/Deep1221 actual0shot4096 pilots',
                              'MMLU_second32': 'Existing r9132 concurrency fixture; prior model usage not exhaustively audited',
                              'not_claimed_unseen': True},
            'reuse_sources': [pin(p) for p in paths],
            'readiness_gaps': [
                'No full14042 MMLU source found in bounded local/repo/remote inventory; 64 frozen recorded rows only. No network retrieval.',
                'MMLU source revision null; response content SHA frozen. Original32 says MIT per publisher card; no independent local MMLU license text found.',
                'Current per-model prepared processor/engine encoding for128 rows absent; old r9110 token IDs are not current qualified rows.',
                'New immutable finite-plan owner/client adapters and actual whole-entry CPU closure not implemented; fixed6/18 clients cannot directly run this card.',
                'Additional independent replicas require exact loader/provider/namespace CPU qualification and fresh resource claim; no GPU reservations or submissions.',
                'Historical pilot/loader wall is an estimate; new questions and ordinary4096 caps can exceed proposed deadline; stop/partial reporting must preserve full denominator.'
            ]}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('output', type=pathlib.Path)
    args = p.parse_args()
    result = prepare()
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps({'completed': True, 'GPU_GO': False, 'case_count': 128,
                      'output': str(args.output), 'SHA256': digest(args.output.read_bytes())}))
