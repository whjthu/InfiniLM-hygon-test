"""Mask-equivalent prefix SDPA for the frozen B=1 noncausal attention layout."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixLayout:
    heads: int
    length: int
    width: int
    valid: int

    def __post_init__(self):
        if (any(type(v) is not int or v <= 0 for v in (self.heads, self.length, self.width, self.valid))
                or self.valid >= self.length):
            raise ValueError("The fixed input must have a nonempty proper valid prefix")

    @property
    def shape(self):
        return (1, self.heads, self.length, self.width)

    @property
    def stride(self):
        return (self.heads * self.length * self.width, self.width, self.heads * self.width, 1)


def torch_attention(functional, query, key, value, mask, layout, *, use_prefix):
    if use_prefix:
        inputs = (t[..., :layout.valid, :] for t in (query, key, value))
        compact = functional.scaled_dot_product_attention(
            *inputs, attn_mask=None, dropout_p=0.0, is_causal=False)
        return functional.pad(compact, (0, 0, 0, layout.length - layout.valid))
    scores = functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return scores.masked_fill(~mask.transpose(-1, -2), 0)
