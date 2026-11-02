"""Token-ID benchmark client clock; not buffered HTTP/SSE text timing.

Feed actual decoded control frames from authenticated role connections. Socket
readers may race across P/D: retain receive ticks, deliver in sequence, and
charge any reorder wait to visible delivery. Remote timestamps are never used
for client latency. This module does not synthesize source device receipts.
"""
import math
import threading
import time

from state_wire import require


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (position - lo)


class TokenClient:
    def __init__(self, nonce, epoch, mode, co_role, output_budget=128,
                 *, admitted=None, deadline=None, clock=time.monotonic):
        require(mode in ['co', 'PD'] and co_role in ['P', 'D'], 'Unknown role/mode')
        require(type(epoch) is int and epoch > 0 and type(output_budget) is int
                and 1 <= output_budget <= 8192, 'Invalid request budget')
        self.nonce, self.epoch, self.mode, self.co_role = nonce, epoch, mode, co_role
        self.budget, self.clock = output_budget, clock
        self.admitted = clock() if admitted is None else admitted
        self.deadline = self.admitted + 3600 if deadline is None else deadline
        require(math.isfinite(self.admitted) and math.isfinite(self.deadline)
                and self.deadline > self.admitted, 'Invalid original clock interval')
        self.pending, self.delivered = {}, []
        self.received, self.finish = {}, None
        self.error, self.complete_tick = None, None
        self.lock = threading.Lock()

    def frame(self, role, event):
        """role comes from the caller's authenticated connection, not the frame."""
        with self.lock:
            try:
                tick = self.clock()
                require(math.isfinite(tick) and self.admitted <= tick <= self.deadline,
                        'Actual client clock outside original deadline')
                require(not self.error and self.complete_tick is None, 'Request already terminal')
                require(event['request_nonce'] == self.nonce and type(event['epoch']) is int
                        and event['epoch'] == self.epoch, 'Foreign request/epoch')
                if event['kind'] == 'token':
                    require(set(event) == {'request_nonce', 'epoch', 'kind', 'seq', 'token_id'},
                            'Unexpected token fields')
                    seq, token = event['seq'], event['token_id']
                    require(type(seq) is int and 0 <= seq < self.budget
                            and seq not in self.received and type(token) is int
                            and 0 <= token < 129280, 'Repeated or invalid token event')
                    expected = ('P' if seq == 0 else 'D') if self.mode == 'PD' else self.co_role
                    require(role == expected, 'Token came from wrong stage')
                    if self.finish:
                        require(seq < self.finish['generated_tokens'], 'Token exceeds terminal count')
                    self.received[seq] = dict(token_id=token, received_monotonic=tick, role=role)
                    self.pending[seq] = self.received[seq]
                    while len(self.delivered) in self.pending:
                        index = len(self.delivered)
                        row = self.pending.pop(index)
                        require(not self.delivered or self.delivered[-1]['token_id'] != 1,
                                'Token emitted after EOS')
                        self.delivered.append(dict(seq=index, **row, delivered_monotonic=tick))
                elif event['kind'] == 'finish':
                    require(set(event) == {'request_nonce', 'epoch', 'kind', 'generated_tokens', 'reason'},
                            'Unexpected finish fields')
                    count = event['generated_tokens']
                    require(self.finish is None and type(count) is int and 0 < count <= self.budget
                            and event['reason'] in ['eos', 'length'], 'Invalid terminal receipt')
                    terminal_role = ('P' if count == 1 else 'D') if self.mode == 'PD' else self.co_role
                    require(role == terminal_role and all(i < count for i in self.received),
                            'Wrong terminal stage/count')
                    self.finish = dict(event)
                else:
                    raise ValueError('Unexpected client event kind')
                if self.finish and len(self.delivered) == self.finish['generated_tokens']:
                    eos = self.delivered[-1]['token_id'] == 1
                    require((self.finish['reason'] == 'eos') == eos, 'EOS/finish differs')
                    require(eos or len(self.delivered) == self.budget, 'Unexpected short length stop')
                    self.complete_tick = tick
            except BaseException as error:
                self.error = repr(error)
                raise

    def summary(self):
        with self.lock:
            visible = [r for r in self.delivered if r['token_id'] != 1]
            ticks = [r['delivered_monotonic'] for r in visible]
            gaps = [b - a for a, b in zip(ticks, ticks[1:])]
            return dict(request_nonce=self.nonce, epoch=self.epoch, mode=self.mode,
                        completed=self.complete_tick is not None and not self.error,
                        error=self.error, generated_received=len(self.received),
                        visible_tokens=len(visible), next_expected_sequence=len(self.delivered),
                        original_output_budget=self.budget, admitted_monotonic=self.admitted,
                        client_token_ID_TTFT_s=ticks[0] - self.admitted if ticks else None,
                        client_token_ID_decode_tok_s=(len(ticks) - 1) / (ticks[-1] - ticks[0])
                        if len(ticks) > 1 and ticks[-1] > ticks[0] else None,
                        client_token_ID_ITL_p50_s=percentile(gaps, .50),
                        client_token_ID_ITL_p95_s=percentile(gaps, .95),
                        client_token_ID_ITL_p99_s=percentile(gaps, .99),
                        client_token_ID_ITL_max_s=max(gaps) if gaps else None,
                        token_ID_completion_wall_s=self.complete_tick - self.admitted
                        if self.complete_tick is not None else None,
                        finish_reason=self.finish['reason'] if self.finish else None,
                        out_of_order_buffered=any(r['received_monotonic'] < r['delivered_monotonic']
                                                 for r in self.delivered),
                        no_cross_host_clock_subtraction=True,
                        HTTP_SSE_text_SLO_qualified=False, Native_state_qualified=False)
