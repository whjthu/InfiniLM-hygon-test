"""Encode requests already prepared by the current upstream agent routes.

The route owns tool filtering and per-turn constraints. The Native bridge owns
admission, the full context/output budget, queueing and lifecycle as before.
"""
import copy
from collections.abc import Mapping

_RESERVED = {"tokenize", "return_dict", "return_tensors", "add_generation_prompt"}


def make_prepared_encode_chat(processor):
    """Build a NativeEngineBridge encode callback for the prepared route only.

    Kimi's segment-aware processor path must survive into Native input IDs;
    plain rendered text must not be flattened and tokenized again on that path.
    This function performs no Native/weights import or model construction.
    """
    def encode_chat(messages, data, add_generation_prompt, kwargs):
        if not isinstance(data.get("_agent_schemas"), Mapping) or data.get("_agent_choice") not in {"auto", "none", "required"}:
            raise ValueError("Expected an upstream-prepared agent request")
        if _RESERVED.intersection(kwargs):
            raise ValueError("Processor control fields must not be supplied as chat template options")
        before_messages, before_kwargs = copy.deepcopy(messages), copy.deepcopy(kwargs)
        prompt = processor.apply_chat_template(
            copy.deepcopy(messages), tokenize=False,
            add_generation_prompt=add_generation_prompt, **copy.deepcopy(kwargs))
        segment_encoder = getattr(processor, "encode_chat_inputs", None)
        if callable(segment_encoder):
            encoded = segment_encoder(copy.deepcopy(messages), return_tensors=None,
                                      add_generation_prompt=add_generation_prompt,
                                      **copy.deepcopy(kwargs))
        else:
            encoded = processor(prompt, return_tensors=None)
        if not isinstance(prompt, str) or not isinstance(encoded, Mapping):
            raise ValueError("Invalid processor prompt or encoded input mapping")
        ids = encoded.get("input_ids")
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, (list, tuple)) and len(ids) == 1 and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        if not isinstance(ids, (list, tuple)) or not ids or any(type(value) is not int or value < 0 for value in ids):
            raise ValueError("Expected one complete, nonempty sequence of integer input IDs")
        if messages != before_messages or kwargs != before_kwargs:
            raise ValueError("Processor changed caller history or template options")
        return prompt, list(ids)
    return encode_chat
