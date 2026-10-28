"""Build data-only existing-entry queues; never submit or run GPU work.

A=C8W8 closed cohort, B=two independent C4W8 closed cohorts.
B is NOT two live microbatches, has no PP overlap, and resets between cohorts.
"""
import copy
import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent/'Kimi-current-model-sparse-profile-launch-private-r136b/queue'


def main():
    sources = sorted(BASE.glob('q*.json'))
    originals = [json.loads(p.read_text(encoding='utf-8')) for p in sources]
    groups = [q for q in originals if q.get('action') == 'case' and q['id'].endswith('OFFbefore')]
    assert len(groups) == 4 and all(q['C'] == 8 and q['W'] == 8 and len(q['requests']) == 8 for q in groups)
    plans = {}
    for name, selected in [('first8-minimal', groups[:1]), ('original32-optional', groups)]:
        directory = HERE/name
        directory.mkdir(exist_ok=True)
        commands = []
        for arm in ('A1', 'B1', 'B2', 'A2'):
            for group_index, original in enumerate(selected):
                slices = [original['requests']] if arm.startswith('A') else [original['requests'][:4], original['requests'][4:]]
                # Reverse B2 subset order to counter the first/last-four placement effect.
                if arm == 'B2': slices = slices[::-1]
                for part, requests in enumerate(slices):
                    q = copy.deepcopy(original)
                    q.update(id=f'r154-{arm}-g{group_index}-p{part}', C=len(requests), requests=requests, model_profile=False)
                    q.pop('service_id', None); q.pop('service_part', None)
                    commands.append(q)
        # Profile OFF/ON/OFF brackets stay separate from throughput ABBA measurements.
        for C in (8, 4):
            for observer in ('OFFbefore', 'ON', 'OFFafter'):
                q = copy.deepcopy(selected[0]);q.update(id=f'r154-profile-C{C}W8-{observer}', C=C, requests=selected[0]['requests'][:C], model_profile=observer == 'ON')
                commands.append(q)
        commands.append({'schema': 1, 'action': 'stop', 'reason': 'r154 bounded shape-demand comparison complete'})
        for index, q in enumerate(commands):
            (directory/f'q{index:04d}.json').write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding='utf-8')
        plans[name] = {'commands': len(commands), 'throughput_cases': len(commands)-7, 'unique_requests': 8*len(selected), 'profile_ON_windows': 2, 'profile_bracket_cases': 6, 'scope': 'independent closed cohorts only; no two-micro live service claim'}
    receipt = {'Native': '3057262312679f86bcdb181ce198b9b1f1d2fccc59fc204b4adbc197132b60dd', 'plans': plans, 'original_queue_files': [{'name':p.name,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources], 'submitted': False}
    (HERE/'shape-queue-receipt-r154.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(plans, ensure_ascii=False))


if __name__ == '__main__':
    main()
