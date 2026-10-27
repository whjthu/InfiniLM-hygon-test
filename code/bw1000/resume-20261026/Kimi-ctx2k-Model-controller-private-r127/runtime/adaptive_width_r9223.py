"""Pure CPU verification-width budget. Width includes the pending anchor.
This function does not authorize a model capacity or change native T limits.
"""
def width_cap(active_count, max_tokens=64, max_width=8):
    values=(active_count,max_tokens,max_width)
    if any(type(v) is not int for v in values):
        raise ValueError('counts must be integers, not booleans')
    if active_count<0 or max_tokens<1:
        raise ValueError('invalid active count or token budget')
    if max_width<1 or max_width>8 or max_width&(max_width-1):
        raise ValueError('width must be a power of two no greater than trained width 8')
    if active_count==0:
        return 0
    if active_count>max_tokens:
        raise ValueError('even one anchor per active request exceeds token budget')
    bound=min(max_width,max_tokens//active_count)
    return 1<<(bound.bit_length()-1)
