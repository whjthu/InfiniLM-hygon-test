"""Actual finite66 HTTP client; quality failures are results, not early aborts."""
import argparse
import copy
import hashlib
import json
import pathlib
import re
import time
from fractions import Fraction
from source_pin_r281 import load_wire, verify, sha
from client_owner_r281 import bind_identity, row, backend_metrics
from quality_plan import load_data, plan, messages, encoding, TOTAL, selected_shard


def numeric(text):
    values = re.findall(r'####[ *$]*([-+]?(?:[0-9][0-9,]*(?:[.][0-9]+)?|[.][0-9]+)(?:/[0-9]+)?)', text)
    try:
        return Fraction(values[-1].replace(',', '')) if values else None
    except (ValueError, ZeroDivisionError):
        return None


def letter(text):
    values = re.findall(r'####\s*([ABCD])\b', text)
    return ord(values[-1]) - 65 if values else None


def main():
    p = argparse.ArgumentParser()
    for name in ['base-url', 'model', 'output-dir', 'identity', 'service-report', 'manifest-sha', 'expected-step']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    p.add_argument('--timeout', type=float, default=7200)
    args = p.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(exist_ok=False)
    q = dict(completed=False, passed=False, CPU_only=False, planned_HTTP=TOTAL,
             actual_HTTP_attempts=0, actual_HTTP_completed=0, actual_Native_completed=0,
             no_retries=True, full_benchmark=False, no_regression_proven=False,
             requests=[], arms={str(a): {'suites': {s: {'rows': []} for s in ['GSM8K', 'MMLU']}} for a in [0, 1]},
             scope='Custom0shot64+64 whole questions, one fourth of pairs; normal4096; no quality-score early stop')

    def save():
        (out / 'result.json').write_text(json.dumps(q, ensure_ascii=False, indent=2) + '\n')

    def remaining():
        left = args.deadline_unix - time.time()
        if left <= 90:
            raise TimeoutError('Original residency deadline exhausted; no retry')
        return left

    save()
    try:
        identity = bind_identity(args.identity, args.manifest_sha, args.service_report)
        assert identity['actual_step'] == args.expected_step
        enc = encoding()
        assert identity['prepared_entry_identity'] == enc['prepared_entry_identity']
        assert identity['comparison_contract'] == enc['comparison_contract']
        assert identity['replica'] == selected_shard()
        refs, cases = load_data()
        frozen = plan()
        wire = load_wire(args.manifest_sha)
        verify(args.manifest_sha)
        q.update(comparison_contract=enc['comparison_contract'], replica=selected_shard(), identity_SHA256=sha(args.identity))
        for index, entry in enumerate(frozen):
            remaining()
            case = cases[entry['case_id']]
            where = out / ('request-%03d' % index)
            where.mkdir()
            client = wire.WireClient(args.base_url, args.model, where, 4096, min(args.timeout, remaining() - 60))
            client.session = 'Deep375-r%d-i%d-mode%d' % (selected_shard(), index, entry['arm'])
            prompt = messages(case)
            original = copy.deepcopy(prompt)
            q['actual_HTTP_attempts'] += 1
            save()
            message, finish, usage, event = client.request(prompt, 'none', False, None, max_tokens=4096, thinking=False)
            assert original == prompt and event['HTTP_status'] == 200
            assert finish in ['stop', 'length'] and not message.get('tool_calls')
            body = json.loads((where / 'request-001.request.json').read_bytes())
            assert body['messages'] == prompt and body['temperature'] == 0 and body['max_tokens'] == 4096
            assert body['chat_template_kwargs']['enable_thinking'] is False
            raw = json.loads((where / 'request-001.wire').read_bytes())
            r = row(pathlib.Path(args.service_report), args, raw['id'], index, client.session, 4096, remaining)
            expected = enc['cases'][entry['case_id']]
            assert usage['prompt_tokens'] == r['prompt_tokens'] == expected['prompt_tokens']
            assert r['admission']['history_render_audit']['Native_prompt_token_ids_SHA256'] == expected['prompt_IDs_SHA256']
            assert r['prefill_binding']['planned_case'] == entry and r['prefill_binding']['prefill_chunk'] == 1024
            assert r['HC_mode'] == entry['arm']
            coverage = r['HC_forward_coverage']
            assert len(coverage) == sum(r['Native_forward_host_summary'][k + '_calls'] for k in ['prefill', 'decode'])
            sizes = [x['tokens'] for x in r['Native_continuation'] if x['kind'] == 'prefill']
            sizes += [1] * r['Native_forward_host_summary']['decode_calls']
            assert [x['input_tokens'] for x in coverage] == sizes
            assert all(x['mode'] == entry['arm'] and x['eligible_T1_BF16'] == (size == 1) and
                       bool(x['candidate_completed_branch']) == (size == 1 and entry['arm'] == 1)
                       for x, size in zip(coverage, sizes))
            natural = r['native_finish_reason'] == 'eos_token'
            assert (finish == 'stop') == natural
            value = numeric(message.get('content') or '') if case['suite'] == 'GSM8K' else letter(message.get('content') or '')
            gold = Fraction(case['gold_numeric'].replace(',', '')) if case['suite'] == 'GSM8K' else case['answer']
            result = dict(entry, source_index=case['source_index'], subject=case.get('subject'),
                          question_SHA256=hashlib.sha256(case['question'].encode()).hexdigest(),
                          correct=bool(value is not None and value == gold and natural),
                          parse_failure=value is None, natural_EOS=natural, cap_hit=not natural,
                          output_budget=4096, generated_tokens=r['actual_generated_tokens'], prompt_tokens=r['prompt_tokens'],
                          request_SHA256=sha(where / 'request-001.request.json'), wire_SHA256=sha(where / 'request-001.wire'),
                          client_HTTP_wall_s=event['elapsed_s'], HC_T1_eligible_forwards=sum(size == 1 for size in sizes),
                          **backend_metrics(r))
            q['requests'].append(result)
            if not entry['warmup']:
                q['arms'][str(entry['arm'])]['suites'][case['suite']]['rows'].append(result)
            q['actual_HTTP_completed'] += 1
            q['actual_Native_completed'] += 1
            save()
        assert q['actual_HTTP_attempts'] == q['actual_HTTP_completed'] == q['actual_Native_completed'] == TOTAL
        for arm in q['arms'].values():
            arm['comparison_contract'] = enc['comparison_contract']
            assert all(len(s['rows']) == 16 for s in arm['suites'].values())
        q['passed'] = True
        q['passed_scope'] = 'All66 transport/identity/Native/HC receipts completed; quality scores retained without coarse-screen gating'
    except BaseException as error:
        q['error'] = repr(error)
    finally:
        q['completed'] = True
        save()
    return 0 if q['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
