"""Private C2/M1 decode consumer for unchanged038d ABI; no imports of GPU/runtime.

Prefill two independent rows first using the existing bounded prefill path. Pass
the resulting past/last emitted token here. No row is reused until BOTH finish;
only then may the owner call the original whole-context reset. No HTTP adapter,
constructor, parameter loader, or GPU launch is implemented by this CPU card.
"""
from dataclasses import dataclass
from decode_wave_r9225 import decode_wave_r9225


@dataclass
class Row:
    row: int
    past: int
    token: int
    budget: int
    produced: int = 0
    done: bool = False


def decode_round(rows, forward, feedback, *, enabled=False, capacity=32768, check=lambda: None):
    """forward gets exact CAPI metadata; feedback owns actual global token vote.

    forward(tokens=1,count=1,descriptor=(row,past,1),input_ids=(token,),
            positions=(past,)) returns an owned logits handle and timing tuple.
    feedback(row, logits) performs the ORIGINAL owner-thread control/argmax/
    transport boundary and returns (token, eos). Keep that collective order on
    all32 owners. At most two independent forwards precede either feedback.
    """
    if type(enabled) is not bool or len(rows) != 2 or {r.row for r in rows} != {0, 1}:
        raise ValueError('Exactly two independent stable rows required')
    for r in rows:
        if (any(type(v) is not int for v in [r.row,r.past,r.token,r.budget,r.produced])
                or not 0 <= r.past <= capacity or r.token < 0 or r.budget < 1
                or not 0 <= r.produced <= r.budget or type(r.done) is not bool):
            raise ValueError('Invalid row state')
        if not r.done and (r.past >= capacity or r.produced >= r.budget):
            raise ValueError('Live row has no remaining context/output capacity')
    by_row = {r.row:r for r in rows}
    active = [[r.row] for r in rows if not r.done]

    def invoke(group):
        check()
        r = by_row[group[0]]
        return forward(tokens=1, count=1, descriptor=(r.row,r.past,1),
                       input_ids=(r.token,), positions=(r.past,))

    for group, logits, wall, tick, ended in decode_wave_r9225(active, invoke, enabled):
        check()
        r = by_row[group[0]]
        token, eos = feedback(r.row, logits)
        if type(token) is not int or token < 0 or type(eos) is not bool:
            raise ValueError('Invalid actual feedback')
        r.past += 1
        r.token = token
        r.produced += 1
        r.done = eos or r.produced == r.budget
        del logits
    return all(r.done for r in rows)


def run(rows, forward, feedback, reset, close, *, enabled=False, check=lambda: None, max_rounds=8192):
    """One finite pair; no refill/retry. Existing owner callbacks own all cleanup.

    A failed partial round must not resume with its frontend ledger: Native may
    already have advanced one or both rows. Reset only after owned operations
    have drained, and close the context. The supplied reset/close callbacks must
    retain current32-owner coordination. A process/CCL failure remains fail-stop.
    check must not introduce a WORLD feedback between the two forwards; global
    cancellation is agreed at round boundary by the existing owner/controller.
    """
    if type(max_rounds) is not int or not 1<=max_rounds<=8192:
        raise ValueError('Bounded output schedule required')
    completed=False
    try:
        for _ in range(max_rounds):
            check()
            if decode_round(rows,forward,feedback,enabled=enabled):
                completed=True
                break
        if not completed:
            raise TimeoutError('Finite cohort budget exhausted; no replay')
        return True
    finally:
        try:reset()
        finally:close()
