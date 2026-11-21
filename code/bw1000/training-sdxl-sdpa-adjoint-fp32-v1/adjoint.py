"""Blockwise SDPA adjoint for already validated, finite FP32 operands."""


QUERY_BLOCK = 64


def vjp(query, key, value, upstream, scale, *, softmax, row_sum, concatenate):
    key_t = key.swapaxes(-2, -1)
    value_t = value.swapaxes(-2, -1)
    d_query = []
    d_key = key * 0
    d_value = value * 0
    for start in range(0, query.shape[-2], QUERY_BLOCK):
        q = query[..., start:start+QUERY_BLOCK, :]
        dy = upstream[..., start:start+QUERY_BLOCK, :]
        probabilities = softmax((q @ key_t) * scale)
        d_probability = dy @ value_t
        d_logits = probabilities * (d_probability - row_sum(d_probability * probabilities))
        d_query.append((d_logits @ key) * scale)
        d_key += (d_logits.swapaxes(-2, -1) @ q) * scale
        d_value += probabilities.swapaxes(-2, -1) @ dy
    return concatenate(d_query), d_key, d_value
