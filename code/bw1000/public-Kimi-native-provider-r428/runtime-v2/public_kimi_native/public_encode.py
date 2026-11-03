"""Consume the current public route's already-derived Kimi intent exactly once."""
from collections.abc import Mapping


def encode(codec, messages, data, add_generation_prompt, kwargs):
    if not isinstance(data.get('_agent_schemas'), Mapping) or data.get('_agent_choice') not in {'none', 'auto', 'required'}:
        raise ValueError('Expected current upstream-prepared Kimi request')
    # Current c2b714... prepare_agent_request already inserted the exact named
    # and parallel rules previously owned by r223. Do not insert them again.
    return codec.encode_chat(messages, kwargs, add_generation_prompt)
