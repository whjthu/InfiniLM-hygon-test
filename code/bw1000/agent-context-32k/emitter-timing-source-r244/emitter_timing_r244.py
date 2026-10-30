"""Source-only server emitter timing snapshot; no runtime/GPU imports.
Accepted means after frontend encoding/admission, not HTTP ingress.
Model non-EOS tokens include reasoning and tool syntax. Buffered transport
provides no incremental client TTFT/ITL evidence.
"""
import math


def snapshot_emitter_timing(*, accepted, started, token_times, generated_count, ends_in_eos):
    base = {'scope': 'server admitted to emitter; frontend encoding and HTTP ingress excluded',
            'client_incremental_latency_qualified': False,
            'percentile_method': 'linear interpolation at (N-1)*q, per request',
            'timestamps_valid': False}
    def number(v):
        try:
            return type(v) in (int, float) and math.isfinite(v)
        except (OverflowError, TypeError):
            return False
    if (type(generated_count) is not int or not 0 <= generated_count <= 8192 or
            type(ends_in_eos) is not bool or (ends_in_eos and not generated_count) or
            not isinstance(token_times, (list, tuple)) or len(token_times) != generated_count or
            not number(accepted) or not number(started) or accepted > started or
            any(not number(t) for t in token_times) or
            any(b < a for a, b in zip(token_times, token_times[1:])) or
            (token_times and token_times[0] < started)):
        return dict(base, error='invalid_or_missing_timing_metadata')
    times = list(token_times[:-1] if ends_in_eos else token_times)
    deltas = [b-a for a, b in zip(times, times[1:])]
    if (any(not number(v) for v in deltas) or
            (times and (not number(times[-1]-times[0]) or not number(times[0]-accepted)))):
        return dict(base, error='invalid_or_missing_timing_metadata')
    def percentile(q):
        if not deltas:
            return None
        ordered = sorted(deltas); x = (len(ordered)-1)*q; left = int(x)
        return ordered[left] + (ordered[min(left+1, len(ordered)-1)]-ordered[left])*(x-left)
    span = times[-1]-times[0] if len(times) >= 2 else None
    return dict(base, timestamps_valid=True, generated_tokens_including_terminal_eos=generated_count,
                terminal_eos_excluded=ends_in_eos, model_non_eos_tokens=len(times),
                accepted_monotonic=accepted, backend_started_after_initial_reset_monotonic=started,
                non_eos_token_monotonic=times,
                server_admitted_to_first_non_eos_s=times[0]-accepted if times else None,
                backend_started_to_first_non_eos_s=times[0]-started if times else None,
                emitter_first_to_last_non_eos_s=span,
                user_decode_tok_s=(len(times)-1)/span if span is not None and span > 0 else None,
                ITL_P95_s=percentile(.95), ITL_P99_s=percentile(.99),
                rate_formula='(non_eos_tokens-1)/(last_non_eos-first_non_eos)',
                rate_available=span is not None and span > 0)
