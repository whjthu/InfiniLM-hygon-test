"""Bind historical reasoning to the already-prepared Native queue request.

Use the same encode callback for admission and audit. The route owns all request
constraints; this module never prepares the request or adds constraints again.
"""
import copy
import hashlib
import json


def ids_sha256(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def make_prepared_history_prequeue(encode_chat, expected_generation_prefix):
    """Audit existing text roles before queue admission; no Native calls.

    expected_generation_prefix(data, kwargs) must use the model's actual template.
    Re-encoding markers is a CPU proof only; marker IDs never enter the request.
    """
    if not callable(encode_chat) or not callable(expected_generation_prefix):
        raise TypeError("An encode callback and model prefix resolver are required")

    def prequeue(req, messages, data, add_generation_prompt, kwargs):
        before = copy.deepcopy(messages)
        before_data, before_kwargs = copy.deepcopy(data), copy.deepcopy(kwargs)
        prompt, ids = encode_chat(messages, data, add_generation_prompt, kwargs)
        if prompt != req.prompt or tuple(ids) != req.prompt_token_ids:
            raise ValueError("Prepared prequeue prompt or token binding mismatch")
        marked = copy.deepcopy(messages)
        fields = []
        for index, message in enumerate(marked):
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError("History audit supports existing text roles only")
            if message.get("content") is not None and not isinstance(message.get("content"), str):
                raise ValueError("History audit supports text content only")
            if message.get("role") != "assistant":
                continue
            key = "reasoning_content" if message.get("reasoning_content") else "reasoning"
            value = message.get(key)
            if value is None or value == "":
                continue
            if not isinstance(value, str) or "<|kimi_image_placeholder|>" in value:
                raise ValueError("History audit supports text reasoning only")
            fingerprint = hashlib.sha256(value.encode()).hexdigest()
            marker = "PREPARED_HISTORY_RC_%d_%s_END" % (index, fingerprint)
            if marker in prompt:
                raise ValueError("History marker collision")
            fields.append((marker, value, fingerprint))
            message[key] = marker
        marked_prompt, _ = encode_chat(marked, data, add_generation_prompt, kwargs)
        restored = marked_prompt
        offsets = []
        for marker, value, _ in fields:
            if marked_prompt.count(marker) != 1:
                raise ValueError("Missing or repeated historical reasoning contribution")
            offsets.append(marked_prompt.index(marker))
            restored = restored.replace(marker, value, 1)
        if restored != prompt or offsets != sorted(offsets) or messages != before or data != before_data or kwargs != before_kwargs:
            raise ValueError("History reconstruction or order mismatch")
        prefix_checked = bool(add_generation_prompt)
        if prefix_checked:
            prefix = expected_generation_prefix(copy.deepcopy(data), copy.deepcopy(kwargs))
            if not isinstance(prefix, str) or not prefix:
                raise ValueError("A concrete generation prefix is required")
            if not prompt.endswith(prefix):
                raise ValueError("Current generation prefix mismatch")
        return {"history_render_audit": {
            "qualified": True, "scope_existing_text_roles_only": True,
            "input_RC_fields": len(fields), "rendered_RC_contributions": len(fields),
            "source_RC_SHA256_sequence": [item[2] for item in fields],
            "RC_one_contribution_per_source_field": True, "RC_order_preserved": True,
            "marked_render_reconstruction_equals_actual_prompt": True,
            "same_prepared_encoder_reproduces_prompt_and_IDs": True,
            "actual_prompt_SHA256": hashlib.sha256(prompt.encode()).hexdigest(),
            "Native_prompt_token_ids_SHA256": ids_sha256(ids),
            "prompt_tokens": len(ids), "output_budget": req.max_new_tokens,
            "generation_prefix_checked": prefix_checked,
            "generation_prefix_correct": True if prefix_checked else None,
            "RC_field_semantics": "Effective assistant reasoning_content or reasoning by truthy-priority; not both aliases independently",
            "prequeue_bound": True,
            "markers_sent_to_Native": False,
            "request_constraints_owned_by_upstream_prepared_route": True}}
    return prequeue
