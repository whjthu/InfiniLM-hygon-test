"""Evaluate actual one-clock request events; never invent client timestamps.

CLI: python online_service_metrics_r8453.py TRACE.json REPORT.json
TRACE: {events: [...], window: [start,end], cards: N, boundary: 'emitter'|'client',
        target_tok_s: 15, max_TTFT_s: optional, max_ITL_p95_s: optional}.
All timestamps including arrivals must be in this boundary's one monotonic
clock domain. Client reports require observed client token events.
"""
import json, math, statistics, sys


def distribution(xs):
    xs = sorted(xs)
    if not xs:
        return None
    def q(p):
        a = (len(xs)-1)*p
        i = int(a)
        return xs[i] + (xs[min(i+1,len(xs)-1)]-xs[i])*(a-i)
    return dict(count=len(xs), mean=statistics.mean(xs), p50=q(.5), p95=q(.95), p99=q(.99), min=xs[0], max=xs[-1])


def evaluate(trace):
    start, end = trace['window']
    assert math.isfinite(start) and math.isfinite(end) and 0 <= start < end
    cards, boundary = trace['cards'], trace['boundary']
    assert type(cards) is int and cards > 0 and boundary in ('emitter', 'client')
    requests, last = {}, -1
    for e in trace['events']:
        t, kind = e['t'], e['kind']
        assert math.isfinite(t) and t >= last >= -1
        last = t
        if 'request' not in e:
            continue
        k = e['request']
        if kind == 'arrival':
            assert k not in requests
            requests[k] = dict(arrival=t, tokens=[], status='unfinished')
            continue
        r = requests[k]
        if kind == 'admit':
            assert 'admit' not in r and r['status'] == 'unfinished'
            r['admit'] = t
        elif kind == 'token':
            assert 'admit' in r and r['status'] == 'unfinished'
            assert e['boundary'] == boundary and e['index'] == len(r['tokens'])
            assert not r['tokens'] or (not r['tokens'][-1]['eos'] and t > r['tokens'][-1]['t'])
            assert type(e['eos']) is bool
            assert not e.get('pad', False), 'padding must not be recorded as user output'
            r['tokens'].append(e)
        elif kind == 'terminal':
            assert r['status'] == 'unfinished'
            assert e['status'] in ('completed', 'cancelled', 'rejected', 'failed')
            r['status'], r['end'] = e['status'], t
            if e['status'] == 'completed':
                assert r['tokens']
    rows, decode_tokens, good_tokens = [], 0, 0
    intervals, ttfts, complete_rates = [], [], []
    prefix_intervals, prefix_ttfts, prefix_rates = [], [], []
    for k, r in requests.items():
        ts = r['tokens']
        gaps = [b['t']-a['t'] for a,b in zip(ts, ts[1:])]
        useful_after_first = sum(not e['eos'] for e in ts[1:])
        rate = useful_after_first/(ts[-1]['t']-ts[0]['t']) if len(ts)>1 else None
        ttft = ts[0]['t']-r['arrival'] if ts else None
        decode_in_window = sum(start < e['t'] <= end and not e['eos'] for e in ts[1:])
        decode_tokens += decode_in_window
        gaps_dist = distribution(gaps)
        completed = r['status']=='completed'
        censored = r['status'] in ('unfinished','cancelled','failed')
        eligible = completed and rate is not None
        good = eligible and rate >= trace.get('target_tok_s', 15)
        if 'max_TTFT_s' in trace:
            good = good and ttft <= trace['max_TTFT_s']
        if 'max_ITL_p95_s' in trace:
            good = good and gaps_dist['p95'] <= trace['max_ITL_p95_s']
        if good:
            good_tokens += decode_in_window
        if completed:
            intervals.extend(gaps)
            if ttft is not None:
                ttfts.append(ttft)
            if rate is not None:
                complete_rates.append(rate)
        elif censored:
            prefix_intervals.extend(gaps)
            if ttft is not None:
                prefix_ttfts.append(ttft)
            if rate is not None:
                prefix_rates.append(rate)
        rows.append(dict(request=k,status=r['status'],TTFT_s=ttft,
                         queue_to_admit_s=r.get('admit',r.get('end',last))-r['arrival'],
                         user_decode_tok_s=rate if completed else None,
                         observed_prefix_decode_tok_s=rate if censored else None,
                         ITL_s=gaps_dist,rate_window='completed' if completed else 'censored_prefix',
                         useful_decode_tokens_in_window=decode_in_window,goodput_qualified=good,
                         complete_request_window=eligible))
    changes = {}
    for r in requests.values():
        for t, dq, da in [(r['arrival'],1,0),
                           *(([(r['admit'],-1,1)]) if 'admit' in r else []),
                           *(([(r['end'],0,-1) if 'admit' in r else (r['end'],-1,0)]) if 'end' in r else [])]:
            changes.setdefault(t,[0,0])
            changes[t][0] += dq
            changes[t][1] += da
    queued = active = 0
    previous = start
    queue_area = active_area = 0.0
    timeline = []
    for t, (dq, da) in sorted(changes.items()):
        if t > end:
            break
        if t >= start:
            queue_area += queued*(t-previous)
            active_area += active*(t-previous)
            previous = t
        queued, active = queued+dq, active+da
        assert queued >= 0 and active >= 0
        if t >= start:
            timeline.append(dict(t=t,queued=queued,admitted=active))
    queue_area += queued*(end-previous)
    active_area += active*(end-previous)
    return dict(scope='ACTUAL_EVENT_EVALUATION_ONLY',boundary=boundary,
                client_delivery='MEASURED' if boundary=='client' else 'NOT_MEASURED',
                window=[start,end],cards=cards,requests=rows,
                statuses={s:sum(r['status']==s for r in requests.values()) for s in ('completed','unfinished','cancelled','rejected','failed')},
                aggregate_decode_tok_s=decode_tokens/(end-start),
                decode_tok_s_per_card=decode_tokens/(end-start)/cards,
                goodput_decode_tok_s_per_card=good_tokens/(end-start)/cards,
                completed_request_metrics=dict(TTFT_s=distribution(ttfts),ITL_s=distribution(intervals),user_decode_tok_s=distribution(complete_rates)),
                censored_prefix_metrics=dict(TTFT_observed_s=distribution(prefix_ttfts),ITL_observed_s=distribution(prefix_intervals),observed_decode_tok_s=distribution(prefix_rates),SLO_qualification=False),
                occupancy_events=timeline,mean_queued_requests=queue_area/(end-start),
                mean_admitted_requests=active_area/(end-start),
                accounting='Exclude ordinal0/prefill token and EOS from decode numerator; include every real intertoken wait including EOS decision. Completed-request distributions and censored prefixes are separate; only completed full windows qualify SLO goodput. Aggregate counts observed useful tokens from all statuses in explicit observation window. No padding tokens accepted.')


if __name__ == '__main__':
    with open(sys.argv[1], encoding='utf8') as f:
        result = evaluate(json.load(f))
    with open(sys.argv[2], 'x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
