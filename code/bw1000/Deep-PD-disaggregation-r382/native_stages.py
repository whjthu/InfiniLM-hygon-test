"""Split the existing C1 greedy continuation at the final prefill sample.

Same forward arguments and logits/sample checks as native_decode_r206b. State
export/import is external: the caller must prove the complete Native snapshot
has been restored before calling decode_from_import. Python carry is not KV.
The D function never accepts prompt IDs and never recomputes their prefill.
"""


def _sample(sampled, vocab):
    token = int(sampled[0])
    if not 0 <= token < vocab:
        raise RuntimeError('Native sampled token outside actual vocabulary')
    return token


def _abort(request, reasons):
    return (request.abort_reason or reasons.CANCELED) if request.cancel_event.is_set() else None


def prefill_until_first(request, forward, check_rc, finite_check, reasons, expired,
                        *, context=32768, chunk_size=1024, eos=1, vocab=129280):
    """Consume every original input once; emit y0 immediately, then return.

    Only carry['handoff_required'] permits a state export. In particular an EOS,
    budget1 or canceled prefill must not be passed off as active D work.
    """
    expired()
    ids, budget = list(request.prompt_token_ids), request.max_new_tokens
    if type(budget) is not int or budget < 1:
        raise ValueError('Native budget must be the original positive JSON integer')
    if not ids or len(ids) + budget > context:
        raise ValueError('Full prompt plus budget exceeds actual Native context; no crop')
    if not 1 <= chunk_size <= 4096:
        raise ValueError('Native per-forward token cap4096 is preserved')
    trace, past, token = [], 0, None
    for offset in range(0, len(ids), chunk_size):
        expired()
        aborted = _abort(request, reasons)
        if aborted is not None:
            return dict(handoff_required=False, finish_reason=aborted, trace=trace,
                        past=past, first_token=None)
        part = ids[offset:offset + chunk_size]
        code, logits, sampled = forward(part, list(range(past, past + len(part))),
                                       [0], [past], [past + len(part)], [0, len(part)])
        check_rc(code); finite_check(logits); expired()
        trace.append(dict(kind='prefill', past=past, tokens=len(part), total=past + len(part), emitted=False))
        past += len(part); token = _sample(sampled, vocab)
    assert past == len(ids) and token is not None
    expired()
    aborted = _abort(request, reasons)
    if aborted is not None:
        return dict(handoff_required=False, finish_reason=aborted, trace=trace,
                    past=past, first_token=None)
    # The caller's emit callback must send the actual first event before any
    # snapshot packing, D2H or transport; do not hide the migration ITL gap.
    request.emit_token(token); trace[-1]['emitted'] = True
    terminal = reasons.EOS_TOKEN if token == eos else reasons.LENGTH if budget == 1 else None
    return dict(handoff_required=terminal is None, finish_reason=terminal,
                past=past, first_token=token, generated_so_far=1, trace=trace)


def decode_from_import(request, past, first_token, forward, check_rc, finite_check,
                       reasons, expired, *, context=32768, eos=1, vocab=129280):
    """Continue after imported cache with past prompt tokens and carried y0.

    The caller already emitted y0 on P; never emit it a second time on D. The
    first call is ([y0], [past], [0], [past], [past+1], [0,1]). The same function
    also serves co continuation, following the same prefill function locally.
    """
    expired()
    budget = request.max_new_tokens
    if type(budget) is not int or budget < 1 or type(past) is not int or not 0 < past < context:
        raise ValueError('Imported continuation geometry differs')
    if type(first_token) is not int or not 0 <= first_token < vocab or past + budget > context:
        raise ValueError('Imported token or full context/output budget differs')
    token, generated, trace, summary = first_token, 1, [], None
    if token == eos:
        return reasons.EOS_TOKEN, trace
    while generated < budget:
        expired()
        aborted = _abort(request, reasons)
        if aborted is not None:
            return aborted, trace
        if past + 1 > context:
            raise RuntimeError('Unexpected Native context overflow')
        code, logits, sampled = forward([token], [past], [0], [past], [past + 1], [0, 1])
        check_rc(code); finite_check(logits); expired()
        if summary is None:
            summary = dict(kind='decode_summary', calls=0, first_past=past, last_total=past)
            trace.append(summary)
        summary['calls'] += 1; summary['last_total'] = past + 1
        past += 1; token = _sample(sampled, vocab)
        request.emit_token(token); generated += 1
        if token == eos:
            return reasons.EOS_TOKEN, trace
    return reasons.LENGTH, trace
