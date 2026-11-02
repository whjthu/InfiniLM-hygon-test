"""Client-clock contract checks; injected clocks are not real PD measurements."""
import json

from token_client import TokenClient


def event(seq, token=42):
    return dict(request_nonce='CPUtoken0123456789', epoch=3, kind='token', seq=seq, token_id=token)


def finish(count, reason='eos'):
    return dict(request_nonce='CPUtoken0123456789', epoch=3, kind='finish', generated_tokens=count, reason=reason)


def client(ticks, mode='PD', budget=128):
    values = iter(ticks)
    return TokenClient('CPUtoken0123456789', 3, mode, 'D', budget,
                       admitted=10, deadline=100, clock=lambda: next(values))


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Invalid client event accepted')


def main():
    checks = []
    c = client([11, 15, 16, 17])
    c.frame('P', event(0)); c.frame('D', event(1)); c.frame('D', event(2, 1)); c.frame('D', finish(3))
    s = c.summary()
    assert s['completed'] and s['visible_tokens'] == 2 and s['client_token_ID_TTFT_s'] == 1
    assert s['client_token_ID_decode_tok_s'] == .25 and s['client_token_ID_ITL_p95_s'] == 4
    assert s['client_token_ID_ITL_max_s'] == 4
    assert s['token_ID_completion_wall_s'] == 7 and not s['out_of_order_buffered']
    checks.append('Pfirst token immediate; four-secondhandoff gap retained; EOSexcluded')
    c = client([11, 12, 15, 16, 17])
    c.frame('D', event(1)); c.frame('P', event(0)); c.frame('D', event(2));
    c.frame('D', event(3, 1)); c.frame('D', finish(4))
    s = c.summary()
    assert s['completed'] and s['out_of_order_buffered'] and s['client_token_ID_TTFT_s'] == 2
    assert abs(s['client_token_ID_decode_tok_s'] - 2 / 3) < 1e-12
    checks.append('Two-source reordered frames buffered untilactualinorderdelivery; no remoteclocks')
    c = client([11, 12], mode='co', budget=1)
    c.frame('D', event(0)); c.frame('D', finish(1, 'length'))
    assert c.summary()['completed'] and c.summary()['finish_reason'] == 'length'
    checks.append('SinglecoDrole controlledcap explicitly retained')
    c = client([11, 12])
    c.frame('P', event(0, 1)); c.frame('P', finish(1))
    assert c.summary()['completed'] and c.summary()['visible_tokens'] == 0
    assert c.summary()['client_token_ID_decode_tok_s'] is None
    checks.append('Prefill immediateEOS no fictitiousDtoken or speed')
    c = client([11, 12]); rejected(lambda: c.frame('D', event(0)))
    rejected(lambda: c.frame('P', event(0)))
    assert c.summary()['error'] and not c.summary()['completed']
    checks.append('Wrongstage poisonsrequest and cannotreplay')
    c = client([11, 12]); c.frame('P', event(0)); rejected(lambda: c.frame('P', event(0)))
    checks.append('Duplicatetoken rejected ratherthaninflatedthroughput')
    c = client([11]); foreign = event(0); foreign['epoch'] = 4
    rejected(lambda: c.frame('P', foreign)); assert c.summary()['error']
    checks.append('Staleepoch refused')
    c = client([101]); rejected(lambda: c.frame('P', event(0)))
    checks.append('Originalabsolutedeadline notextendedbytokenprogress')
    c = client([11, 12]); c.frame('P', event(0)); c.frame('D', finish(3))
    assert not c.summary()['completed'] and c.summary()['next_expected_sequence'] == 1
    checks.append('Earlyfinishwithmissingtokens remainsincompleteandindenominator')
    c = client([11, 12, 13]); c.frame('P', event(0)); c.frame('D', event(1))
    rejected(lambda: c.frame('D', finish(2)))
    checks.append('NaturalEOSassertion checkedagainstactualIDs')
    print(json.dumps(dict(passed=True, checks=checks, CPU_clock_contract_only=True,
                          Native=False, GPU=0, no_PD_performance_measurement=True)))


if __name__ == '__main__':
    main()
