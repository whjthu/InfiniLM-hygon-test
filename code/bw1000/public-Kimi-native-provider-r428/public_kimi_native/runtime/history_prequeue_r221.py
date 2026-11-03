"""Finite text-history contribution proof, bound before the owner can dequeue."""
import copy
import hashlib
import json
from .chat_prepare_r223 import encode as encode_prepared


def ids_SHA256(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def make_prequeue_factory(codec):
    def factory(req, messages, data, add_generation_prompt, kwargs):
        # This is the exact deep copy already used by the bridge encoder.
        before = copy.deepcopy(messages)
        prompt, ids = encode_prepared(codec, messages, data, add_generation_prompt, kwargs)
        if prompt != req.prompt or tuple(ids) != req.prompt_token_ids:
            raise ValueError("prequeue prompt binding mismatch")
        marked = copy.deepcopy(messages)
        fields = []
        for i, message in enumerate(marked):
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError("history audit supports existing text roles only")
            if message.get("content") is not None and not isinstance(message.get("content"), str):
                raise ValueError("history audit supports text content only")
            if message.get("role") != "assistant":
                continue
            key = "reasoning_content" if message.get("reasoning_content") else "reasoning"
            value = message.get(key)
            if value is None or str(value) == "":
                continue
            if not isinstance(value, str) or "<|kimi_image_placeholder|>" in value:
                raise ValueError("history audit supports text reasoning only")
            fingerprint = hashlib.sha256(value.encode()).hexdigest()
            marker = "R221_HISTORY_RC_%d_%s_END" % (i, fingerprint)
            if marker in prompt or any(marker == item[0] for item in fields):
                raise ValueError("history audit marker collision")
            fields.append((marker, value, fingerprint))
            message[key] = marker
        marked_prompt, _ = encode_prepared(codec, marked, data, add_generation_prompt, kwargs)
        restored = marked_prompt
        offsets = []
        for marker, value, _ in fields:
            if marked_prompt.count(marker) != 1:
                raise ValueError("missing or repeated history contribution")
            offsets.append(marked_prompt.index(marker))
            restored = restored.replace(marker, value, 1)
        if restored != prompt or offsets != sorted(offsets) or messages != before:
            raise ValueError("history contribution reconstruction mismatch")
        thinking = kwargs.get("thinking", kwargs.get("enable_thinking", True))
        if type(thinking) is not bool:
            raise ValueError("thinking must be boolean")
        prefix = "<|open|>" + ("think" if thinking else "response") + "<|sep|>"
        if add_generation_prompt and not prompt.endswith(prefix):
            raise ValueError("current generation prefix mismatch")
        return {"history_render_audit": {
            "qualified": True, "scope_existing_text_roles_only": True,
            "input_RC_fields": len(fields), "rendered_RC_contributions": len(fields),
            "source_RC_SHA256_sequence": [item[2] for item in fields],
            "RC_one_contribution_per_source_field": True, "RC_order_preserved": True,
            "marked_render_reconstruction_equals_actual_prompt": True,
            "official_reencode_equals_actual_prompt_and_IDs": True,
            "actual_prompt_SHA256": hashlib.sha256(prompt.encode()).hexdigest(),
            "Native_prompt_token_ids_SHA256": ids_SHA256(ids),
            "prompt_tokens": len(ids), "output_budget": req.max_new_tokens,
            "generation_thinking": thinking, "generation_prefix_correct": True,
            "prequeue_bound": True, "markers_sent_to_Native": False}}
    return factory
