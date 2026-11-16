"""Summarize GLM indexer clocks from the independently audited original run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

RUN = '100cc6c7a0154cd392f4e0f2d694dce4'
REQUEST = 'ca2b58ed31cd42cab20f14224db9fdc5'
CLAIM_COMMIT = 'b1161bc2815f550af81dc9da613cfad912d6ae94'
GPU_STEP = '1103606.146'
NAMES = ('features', 'scored_selector')
CLOCKS = ('host_s', 'enclosing_stream_s')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clock_value(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def summarize_clock(samples):
    if not samples:
        return {'samples': 0, 'min_s': None, 'median_s': None, 'max_s': None}
    return dict(samples=len(samples), min_s=min(samples), median_s=median(samples), max_s=max(samples))


def summarize(report):
    require(report['run_id'] == RUN and report['original_request_id'] == REQUEST
            and report['claim_commit'] == CLAIM_COMMIT, 'different original run')
    gate = report['actual_accounting']['gate']
    require(gate[0] == GPU_STEP and gate[2:4] == ['COMPLETED', '0:0']
            and report['launcher_rc'] == 0 and report['independent_post_clean'] is True
            and report['valid_ffn_evidence'] is True, 'independent terminal evidence did not pass')
    source = report['ffn_observation']
    require(source['planned_ranks'] == 16 and source['planned_frames'] == 48
            and len(source['owners']) == 16, 'unexpected owner denominator')
    frames = []
    for rank, owner in enumerate(source['owners']):
        windows = owner['indexer_windows']
        require(owner['rank'] == rank and type(owner['observed_frames']) is int
                and 0 <= len(windows) == owner['observed_frames'] <= 3
                and owner['full_coverage'] is (len(windows) == 3)
                and len(owner['detailed_layers']) == len(windows), 'invalid rank coverage')
        previous_past = None
        for frame, window in enumerate(windows):
            require(set(window) == {'layer', 'past', *NAMES}, 'invalid row fields')
            layer, past = window['layer'], window['past']
            require(type(layer) is int and 0 <= layer < 78
                    and layer == owner['detailed_layers'][frame][0]
                    and type(past) is int and 0 < past < 32768
                    and (previous_past is None or past == previous_past + 1), 'invalid full-layer or position')
            previous_past = past
            require(window['scored_selector'] is None or window['features'] is not None,
                    'selector clock requires feature marks')
            for name in NAMES:
                value = window[name]
                if value is not None:
                    require(set(value) == set(CLOCKS) and clock_value(value['host_s'])
                            and (value['enclosing_stream_s'] is None
                                 or clock_value(value['enclosing_stream_s'])), 'invalid split clock')
            frames.append(dict(rank=rank, frame=frame, **window))
    require(len(frames) == source['observed_frames']
            and source['full_coverage'] is (len(frames) == 48)
            and report['full_ffn_coverage'] is source['full_coverage'], 'rank-frame denominator differs')
    totals = {}
    for name in NAMES:
        measured = [row[name] for row in frames if row[name] is not None]
        require(len(measured) == source['measured_indexer_windows'][name], 'split measurement count differs')
        totals[name] = dict(
            measured_rank_frames=len(measured),
            **{clock: summarize_clock([row[clock] for row in measured if row[clock] is not None])
               for clock in CLOCKS},
        )
    return dict(run_id=RUN, GPU_step=GPU_STEP, observed_rank_frames=len(frames), planned_rank_frames=48,
                ranks_with_frames=len({row['rank'] for row in frames}),
                positions=sorted({row['past'] for row in frames}), totals=totals,
                clock_scope='Host elapsed and same-stream event envelopes, including enqueue gaps and dependencies.',
                missing_window='No qualified measurement for the split in this frame.',
                formal_inference_speed_qualified=False)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('private_terminal', type=Path)
    cli.add_argument('summary', type=Path)
    args = cli.parse_args()
    raw = args.private_terminal.read_bytes()
    report = summarize(json.loads(raw))
    report['private_terminal_SHA256'] = hashlib.sha256(raw).hexdigest()
    with args.summary.open('x', encoding='utf8', newline='\n') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')


if __name__ == '__main__':
    main()
