"""Audit the completed step 1103606.146 screen using its original artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from summarize import CLAIM_COMMIT, GPU_STEP, REQUEST, RUN, require

INPUT_SHA = '81f6b2d68228aafb79993306847b0edb04c1ba101a35ab21bb822ac96ca2ba5e'
SCORER_SHA = 'c9c39a5fc185ed6bfc5f48c2b547d4ef36a4cfed351626835793fcf0ab5b521c'
OWNER_ARTIFACT = f'GLM474-indexer-{RUN}-socket/owner-00.json'
SUITES = ('gsm8k', 'mmlu')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path, limit, digest=None):
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and (digest is None or sha(raw) == digest),
            f'{path.name} failed its size or SHA-256 binding')
    return raw


def json_sha(value):
    return sha(json.dumps(value, separators=(',', ':')).encode())


def audit_request(row, plan, expected, number, eos, special, grade):
    diagnostic = number == 0
    require(all(row[field] == expected[field] for field in (
        'ordinal', 'case_id', 'suite', 'prompt_tokens', 'prompt_IDs_SHA256', 'body_SHA256'
    )) and row['diagnostic'] is diagnostic and row['screen_index'] == (None if diagnostic else number - 1)
        and all(row[key] is True for key in ('completed', 'reset_before', 'reset_after')),
        f'occurrence {number} has a different request identity or did not complete')
    ids = row['generated_token_ids']
    ticks = row['sampled_ns']
    begin = row['request_enter_ns']
    require(row['planned_max_new_tokens'] == 4096 and 0 < len(ids) == len(ticks) <= 4096
            and all(type(token) is int and 0 <= token < 154880 for token in ids)
            and row['generated_token_ids_SHA256'] == json_sha(ids)
            and not eos.intersection(ids[:-1]), 'invalid generated token sequence')
    require(type(begin) is int and begin > 0 and all(type(tick) is int and tick >= begin for tick in ticks)
            and all(b > a for a, b in zip(ticks, ticks[1:])), 'invalid sampled timeline')
    natural = ids[-1] in eos
    cap = not natural and len(ids) == 4096
    require((natural or cap) and row['natural_EOS'] is natural and row['cap_hit'] is cap,
            'invalid completion status')
    visible = [tick for token, tick in zip(ids, ticks) if token not in special]
    ttft = (visible[0] - begin) / 1e9 if visible else None
    decode = (len(visible) - 1) * 1e9 / (visible[-1] - visible[0]) if len(visible) > 1 else None
    require(row['visible_nonEOS_nonspecial'] == len(visible)
            and row['backend_sampled_TTFT_s'] == ttft and row['backend_decode_tok_s'] == decode,
            'saved backend clocks disagree')
    whole = row['whole_backend_s']
    require(type(whole) in (int, float) and math.isfinite(whole) and whole >= (ticks[-1] - begin) / 1e9,
            'whole request time does not enclose generation')
    score = grade(plan['case'], row['output_text'], natural_EOS=natural, cap_hit=cap)
    require(score == row['score'], 'frozen scorer disagrees with saved score')
    intervals = [b - a for a, b in zip(visible, visible[1:])]
    return dict(ordinal=expected['ordinal'], suite=expected['suite'], diagnostic=diagnostic,
                generated_including_EOS=len(ids), visible_tokens=len(visible), natural_EOS=natural, cap_hit=cap,
                correct=score['correct'], answer_matches_gold=score['answer_matches_gold'],
                parse_failure=score['parse_failure'], backend_TTFT_s=ttft, backend_decode_tok_s=decode,
                backend_ITL_P95_s=percentile(intervals, .95), backend_ITL_P99_s=percentile(intervals, .99))


def percentile(intervals, fraction):
    return sorted(intervals)[math.ceil(fraction * len(intervals)) - 1] / 1e9 if intervals else None


def summarize_rows(rows):
    rates = [r['backend_decode_tok_s'] for r in rows if r['backend_decode_tok_s'] is not None]
    return dict(planned=len(rows), measured_backend_decode=len(rates),
                strict_backend_decode_gt15=sum(rate > 15.0 for rate in rates),
                median_backend_decode_tok_s=median(rates) if rates else None,
                **{key: sum(r[key] for r in rows) for key in (
                    'correct', 'answer_matches_gold', 'parse_failure', 'natural_EOS', 'cap_hit')})


def audit(terminal_bytes, owner_path, source, private_plan_path):
    terminal = json.loads(terminal_bytes)
    row = terminal['actual_accounting']['gate']
    require(terminal['run_id'] == RUN and terminal['original_request_id'] == REQUEST
            and terminal['claim_commit'] == CLAIM_COMMIT and terminal['launcher_rc'] == 0
            and terminal['planned_owner_ranks'] == 16 and terminal['planned_requests_per_rank'] == 33
            and terminal['independent_post_clean'] is True and terminal['valid_ffn_evidence'] is True
            and row[0] == GPU_STEP and row[2:4] == ['COMPLETED', '0:0'], 'original terminal gate did not pass')
    artifacts = terminal['raw_artifact_SHA256']
    require(OWNER_ARTIFACT in artifacts, 'terminal audit did not bind rank zero')
    raw_owner = load(owner_path, 32 * 1024**2, artifacts[OWNER_ARTIFACT])
    owner = json.loads(raw_owner)
    require(owner['global_rank'] == 0 and owner['actual_GPU'] == GPU_STEP and owner['completed'] is True
            and owner['passed'] is True and len(owner['requests']) == 33, 'invalid rank-zero terminal record')
    binding = json.loads(load(source / 'INPUT_BINDING.json', 65536, INPUT_SHA))
    plan_pin = binding['private_plan']
    raw_plan = load(private_plan_path, plan_pin['bytes'], plan_pin['SHA256'])
    require(len(raw_plan) == plan_pin['bytes'], 'wrong private plan length')
    plans = json.loads(raw_plan)['requests']
    pins = plan_pin['requests']
    require(len(plans) == len(pins) == len({r['case_id'] for r in pins}) == 32
            and [p['ordinal'] for p in pins] == binding['selected_ordinals']
            and all(sum(p['suite'] == suite for p in pins) == 16 for suite in SUITES), 'wrong frozen cohort')
    for plan, pin in zip(plans, pins):
        require(all(plan[key] == value for key, value in pin.items())
                and plan['case']['suite'] == pin['suite'], 'private plan row differs from the frozen contract')
    scorer = {}
    exec(compile(load(source / 'scoring.py', 32768, SCORER_SHA), 'scoring.py', 'exec'), scorer)
    eos = set(binding['eos_token_ids'])
    special = eos | set(binding['tokenizer']['special_ids'])
    rows = [audit_request(request, plans[max(0, i - 1)], pins[max(0, i - 1)], i, eos, special, scorer['grade'])
            for i, request in enumerate(owner['requests'])]
    screen = rows[1:]
    return dict(run_id=RUN, GPU_step=GPU_STEP, private_terminal_SHA256=sha(terminal_bytes),
                rank_zero_SHA256=sha(raw_owner), private_plan_SHA256=plan_pin['SHA256'], planned_total_requests=33,
                screen=summarize_rows(screen), by_suite={s: summarize_rows([r for r in screen if r['suite'] == s])
                                                       for s in SUITES}, requests=rows,
                timing_scope='Rank-zero backend sampled timeline. The diagnostic repetition is excluded from screen metrics.',
                independent_tokenizer_replay=False, broad_quality_qualified=False, client_SLO_qualified=False)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ('terminal', 'owner', 'source', 'private_plan', 'output'):
        cli.add_argument(arg, type=Path)
    args = cli.parse_args()
    result = audit(load(args.terminal, 2 * 1024**2), args.owner, args.source, args.private_plan)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
