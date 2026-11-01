"""One immutable remaining-quality and conditional prefill-pair schedule."""
MULTICALL_REQUESTS = 4
SEMANTIC_REQUESTS = 68
PAIR_WINDOWS = (2, 4, 2, 4, 4, 2)
PREFILL_WINDOWS = (2,) * SEMANTIC_REQUESTS + PAIR_WINDOWS
TOTAL_REQUESTS = len(PREFILL_WINDOWS)
PHASES = [('multicall', 'client_multicall_r290.py', MULTICALL_REQUESTS),
          ('quality', 'client_quality_r290.py', SEMANTIC_REQUESTS),
          ('paired', 'client_paired_r290.py', TOTAL_REQUESTS)]


def request_plan(index):
    if type(index) is not int or not 0 <= index < TOTAL_REQUESTS:
        raise ValueError('Immutable 74-request Native schedule exhausted or invalid')
    pair = index - SEMANTIC_REQUESTS if index >= SEMANTIC_REQUESTS else None
    return {'admitted_index': index, 'pair_index': pair,
            'prefill_control_window': PREFILL_WINDOWS[index],
            'paired_warmup': pair is not None and pair < 2}
