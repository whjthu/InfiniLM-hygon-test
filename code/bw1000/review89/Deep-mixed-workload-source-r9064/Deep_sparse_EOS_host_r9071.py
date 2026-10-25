"""Host-only stable-row metadata/EOS helper. Native sparse-row support is unqualified."""
import numpy as np

class StableRows:
    def __init__(self, prompts_by_row, eos_id, vocab_size, max_new_tokens=128):
        assert 1 <= max_new_tokens <= 128 and prompts_by_row
        self.eos_id, self.vocab_size, self.cap = eos_id, vocab_size, max_new_tokens
        self.pending = None
        self.rows = {}
        for row, ids in prompts_by_row.items():
            assert isinstance(row, int) and 0 <= row < 128
            assert ids and all(isinstance(x, int) and 0 <= x < vocab_size and x != eos_id for x in ids)
            self.rows[row] = dict(prompt=list(ids), past=0, generated=[], events=[], prefilled=False, done=False, stop_reason=None)

    def prepare(self, rows, kind):
        assert self.pending is None and kind in ('prefill', 'decode')
        assert rows and len(rows) == len(set(rows))
        inputs, pasts, totals, offsets, positions = [], [], [], [0], []
        for row in rows:
            s = self.rows[row]
            assert not s['done']
            if kind == 'prefill':
                assert not s['prefilled'] and s['past'] == 0
                tokens = s['prompt']
            else:
                assert s['prefilled'] and s['generated'] and s['generated'][-1] != self.eos_id
                tokens = [s['generated'][-1]]
            past = s['past']; total = past + len(tokens)
            assert total <= 1024
            inputs.extend(tokens); pasts.append(past); totals.append(total)
            offsets.append(len(inputs)); positions.extend(range(past, total))
        self.pending = (list(rows), kind, totals)
        values = [rows, pasts, totals, offsets, inputs, positions]
        return tuple(np.ascontiguousarray(x, dtype=np.int32 if i < 4 else np.int64) for i, x in enumerate(values))

    def next_decode(self):
        assert self.pending is None
        active = [row for row, s in sorted(self.rows.items()) if s['prefilled'] and not s['done']]
        return self.prepare(active, 'decode') if active else None

    def observe(self, emitted_ids, api_ready_time, emitter_time):
        assert self.pending is not None and emitter_time >= api_ready_time
        rows, kind, totals = self.pending
        assert len(emitted_ids) == len(rows)
        assert all(isinstance(x, int) and 0 <= x < self.vocab_size for x in emitted_ids)
        for row, total, token in zip(rows, totals, emitted_ids):
            s = self.rows[row]; s['past'] = total; s['prefilled'] = True
            index = len(s['generated']); s['generated'].append(token)
            s['events'].append(dict(index=index, kind=kind, token=token, visible=token != self.eos_id, api_ready=api_ready_time, emitter=emitter_time))
            if token == self.eos_id:
                s['done'] = True; s['stop_reason'] = 'EOS'; s['first_eos_index'] = index
            elif len(s['generated']) == self.cap:
                s['done'] = True; s['stop_reason'] = 'max_new_tokens'
        self.pending = None

    def accounting(self):
        events = [e for s in self.rows.values() for e in s['events']]
        return dict(physical_generated_tokens=len(events), useful_visible_output_tokens=sum(e['visible'] for e in events), useful_decode_tokens=sum(e['visible'] and e['kind'] == 'decode' for e in events), EOS_tokens=sum(not e['visible'] for e in events), finished=sum(s['done'] for s in self.rows.values()), capped=sum(s['stop_reason'] == 'max_new_tokens' for s in self.rows.values()))

def selftest():
    x = StableRows({0:[0,20], 7:[0,21], 31:[0,22]}, 1, 163840)
    a = x.prepare([0,7,31], 'prefill'); assert a[0].tolist() == [0,7,31]
    x.observe([1,90,91], 1., 1.1)
    a = x.next_decode(); assert a[0].tolist() == [7,31] and a[1].tolist() == [2,2] and a[4].tolist() == [90,91]
    x.observe([1,92], 2., 2.1)
    a = x.next_decode(); assert a[0].tolist() == [31] and a[1].tolist() == [3] and a[4].tolist() == [92]
    x.observe([1], 3., 3.1); assert x.next_decode() is None
    assert x.accounting() == dict(physical_generated_tokens=6, useful_visible_output_tokens=3, useful_decode_tokens=1, EOS_tokens=3, finished=3, capped=0)
    y = StableRows({4:[0,30]}, 1, 163840, 2)
    y.prepare([4], 'prefill'); y.observe([7], 1., 1.)
    y.next_decode(); y.observe([8], 2., 2.); assert y.next_decode() is None and y.rows[4]['stop_reason'] == 'max_new_tokens'
    assert y.accounting()['useful_decode_tokens'] == 1
    return dict(passed=True, CPU_only=True, native_CAPI_called=False, sparse_native_cache_correctness='NOT_TESTED', stable_rows=[0,7,31], EOS_after_prefill_and_decode=True, cap_without_EOS=True, no_empty_CAPI=True, no_finished_row_or_EOS_refeed=True)

if __name__ == '__main__':
    import json
    print(json.dumps(selftest()))
