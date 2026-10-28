"""Default-OFF request-local verification widths; no current-cycle oracle.

The trained DFlash2 block and existing Native per-row maximum remain eight.
This policy changes only requested proposal prefix lengths, never acceptance.
"""

def request_width(history, active_count, default_width, *, enabled=False):
    if type(enabled) is not bool:
        raise ValueError('enabled must be a boolean')
    if any(type(x) is not int for x in (active_count, default_width)):
        raise ValueError('integer counts required')
    if not 1 <= active_count <= 32 or default_width not in (1, 2, 4, 8):
        raise ValueError('existing width/capacity required')
    if not enabled or active_count < 4 or default_width != 8:
        return default_width
    # Four full-width observations are required before the first decision.
    # A shorter verify cannot reveal whether its omitted suffix would accept.
    full=[]
    for index,row in enumerate(history):
        if row.get('proposed') == 7 and row.get('terminal') is False:
            a=row.get('accepted_proposals')
            if type(a) is not int or not 0 <= a <= 7:
                raise ValueError('invalid committed acceptance history')
            full.append((index,a))
    if len(full) < 4:
        return 8
    recent=full[-4:]
    # Probe every fourth cycle while narrowed. All inputs are prior commits.
    if len(history)-1-recent[-1][0] >= 3:
        return 8
    # Shorten only repeatedly low-acceptance rows, leaving high rows at W8.
    return 4 if sum(a <= 2 for _,a in recent) >= 3 else 8


def effective_widths(requests, default_width, max_tokens=64, enabled=False):
    active=list(requests)
    if not active:
        return {}
    result={}
    for q in active:
        remaining=q.cap-len(q.tokens)
        context_remaining=1024-q.past
        if q.done or remaining < 1 or context_remaining < 1:
            raise ValueError('only live admissible requests')
        if q.logical in result:
            raise ValueError('duplicate logical row')
        result[q.logical]=min(request_width(q.cycles,len(active),default_width,enabled=enabled),remaining,context_remaining)
    if sum(result.values())>max_tokens:
        raise ValueError('unchanged Native flattened-token bound exceeded')
    return result
