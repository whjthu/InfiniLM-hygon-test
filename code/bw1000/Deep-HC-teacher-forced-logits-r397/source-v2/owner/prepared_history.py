"""Bind complete prepared text contributions before Native queue publication.

The actual public processor renders the original and marked histories. Markers
are CPU audit inputs only and never become Native request IDs or prompt tokens.
"""
import copy
import hashlib
import json


def ids_sha256(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def make_prepared_history_prequeue(encode_chat, expected_generation_prefix, *, reasoning_boundaries=None, normalize_content=None):
    if not callable(encode_chat) or not callable(expected_generation_prefix):
        raise TypeError("An actual encode callback and template prefix resolver are required")

    def prequeue(req, messages, data, generation, kwargs):
        original = copy.deepcopy((messages, data, kwargs))
        prompt, ids = encode_chat(messages, data, generation, kwargs)
        if prompt != req.prompt or tuple(ids) != req.prompt_token_ids:
            raise ValueError("Prepared prompt or token binding mismatch")
        marked = copy.deepcopy(messages)
        fields = []
        content_groups = []
        for index, message in enumerate(marked):
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError("History audit supports existing standard text roles only")
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("History audit supports text content only")
            # Form full tool-message blocks even when one result is empty.
            if role == "tool" and content_groups and content_groups[-1]["tool"] and content_groups[-1]["last_message"] == index - 1:
                group = content_groups[-1]
                group["all_indices"].append(index)
                group["last_message"] = index
            else:
                group = dict(tool=role == "tool", indices=[], all_indices=[index], last_message=index)
                content_groups.append(group)
            keys = []
            if role == "assistant":
                key = "reasoning_content" if message.get("reasoning_content") else "reasoning"
                if message.get(key):
                    keys.append((key, "RC"))
            normalized_content = normalize_content(messages[index]) if normalize_content is not None else content
            if normalized_content is not None and not isinstance(normalized_content, str):
                raise ValueError("Invalid official content normalization")
            if normalized_content:
                keys.append(("content", "content"))
                group["indices"].append(index)
            for key, kind in keys:
                value = normalized_content if kind == "content" else message[key]
                if not isinstance(value, str) or "<|kimi_image_placeholder|>" in value:
                    raise ValueError("History audit supports text fields only")
                fingerprint = hashlib.sha256(value.encode()).hexdigest()
                marker = "PREPARED_HISTORY_%s_%d_%s_END" % (kind, index, fingerprint)
                if marker in prompt:
                    raise ValueError("History marker collision")
                fields.append(dict(index=index, kind=kind, value=value,
                                   SHA256=fingerprint, marker=marker))
                message[key] = marker
        marked_prompt, _ = encode_chat(marked, data, generation, kwargs)
        for field in fields:
            if marked_prompt.count(field["marker"]) != 1:
                raise ValueError("Missing or repeated source text contribution")
            field["marker_offset"] = marked_prompt.index(field["marker"])
        ordered = sorted(fields, key=lambda field: field["marker_offset"])
        pieces, cursor, length = [], 0, 0
        for field in ordered:
            offset = field["marker_offset"]
            segment = marked_prompt[cursor:offset]
            pieces.extend((segment, field["value"]))
            field["start"] = length + len(segment)
            field["end"] = field["start"] + len(field["value"])
            length = field["end"]
            cursor = offset + len(field["marker"])
        pieces.append(marked_prompt[cursor:])
        if "".join(pieces) != prompt or (messages, data, kwargs) != original:
            raise ValueError("Prepared source-text reconstruction or caller mutation mismatch")
        rc = [field for field in fields if field["kind"] == "RC"]
        rendered_rc = [field for field in ordered if field["kind"] == "RC"]
        content = [field for field in fields if field["kind"] == "content"]
        rendered_content = [field for field in ordered if field["kind"] == "content"]
        if [f["index"] for f in rc] != [f["index"] for f in rendered_rc]:
            raise ValueError("Historical reasoning order changed")
        group_for = {index: group for group, item in enumerate(content_groups) for index in item["indices"]}
        group_sequence = [group_for[f["index"]] for f in rendered_content]
        if group_sequence != sorted(group_sequence):
            raise ValueError("Outer history content order changed")
        for group, item in enumerate(content_groups):
            expected = item["indices"]
            actual = [f["index"] for f in rendered_content if group_for[f["index"]] == group]
            if item["tool"] and len(item["all_indices"]) > 1:
                # Concurrent tool results may retain wire order or normalize to
                # the preceding assistant's call order by opaque original ID.
                block = item["all_indices"]
                prior = messages[block[0] - 1] if block[0] else {}
                calls = prior.get("tool_calls", []) if prior.get("role") == "assistant" else []
                call_ids = [call.get("id") for call in calls]
                result_ids = [messages[index].get("tool_call_id") for index in block]
                if not all(isinstance(value, str) and value for value in call_ids + result_ids) or len(set(call_ids)) != len(call_ids) or len(set(result_ids)) != len(result_ids) or not set(result_ids) <= set(call_ids):
                    raise ValueError("Parallel tool-result IDs are not bound to original calls")
                canonical = sorted(expected, key=lambda index: call_ids.index(messages[index]["tool_call_id"]))
                if actual not in (expected, canonical):
                    raise ValueError("Parallel tool results changed beyond allowed ID normalization")
            elif actual != expected:
                raise ValueError("Source content order changed")
        rc_spans = []
        if rc and reasoning_boundaries is not None:
            opening, closing = reasoning_boundaries
            if not isinstance(opening, str) or not opening or not isinstance(closing, str) or not closing:
                raise ValueError("Concrete reasoning boundaries are required")
            for field in rendered_rc:
                start, end = field["start"], field["end"]
                opened = prompt[:start].endswith(opening)
                closed = prompt[end:].startswith(closing)
                equal = prompt[start:end] == field["value"]
                if not (opened and closed and equal):
                    raise ValueError("Historical reasoning moved outside its actual thinking segment")
                rc_spans.append(dict(message_index=field["index"], start_char=start,
                                     end_char=end, RC_SHA256=field["SHA256"],
                                     opening_correct=opened, closing_correct=closed,
                                     actual_prompt_slice_equal=equal))
        prefix_checked = bool(generation)
        if prefix_checked:
            prefix = expected_generation_prefix(copy.deepcopy(data), copy.deepcopy(kwargs))
            if not isinstance(prefix, str) or not prefix or not prompt.endswith(prefix):
                raise ValueError("Actual generation prefix mismatch")
        audit = dict(qualified=True, scope_existing_text_roles_only=True,
                     input_RC_fields=len(rc), rendered_RC_contributions=len(rc),
                     source_RC_SHA256_sequence=[field["SHA256"] for field in rc],
                     RC_one_contribution_per_source_field=True, RC_order_preserved=True,
                     reasoning_structural_boundaries_checked=bool(reasoning_boundaries is not None),
                     RC_spans=rc_spans,
                     content_normalization="Explicit actual-template normalizer" if normalize_content is not None else "Exact source string",
                     RC_contributions_nonoverlapping=all(a["end"] <= b["start"] for a, b in zip(rendered_rc, rendered_rc[1:])),
                     input_content_parts=len(content), rendered_content_contributions=len(content),
                     content_one_contribution_each_and_ordered=True,
                     all_content_contribution_slices_equal=all(prompt[f["start"]:f["end"]] == f["value"] for f in fields),
                     content_order_policy="Outer source history ordered; consecutive parallel tool results may retain wire order or normalize by original opaque call ID",
                     actual_tool_ID_continuation_correctness="Requires real Model tool-result continuation; not established by this render audit alone",
                     content_sequence_SHA256=hashlib.sha256(json.dumps([f["SHA256"] for f in content], separators=(",", ":")).encode()).hexdigest(),
                     marked_render_reconstruction_equals_actual_prompt=True,
                     official_normalized_render_equals_actual_prompt=True,
                     same_prepared_encoder_reproduces_prompt_and_IDs=True,
                     actual_prompt_SHA256=hashlib.sha256(prompt.encode()).hexdigest(),
                     Native_prompt_token_ids_SHA256=ids_sha256(ids),
                     prompt_tokens=len(req.prompt_token_ids), output_budget=req.max_new_tokens,
                     generation_prefix_checked=prefix_checked,
                     generation_prefix_correct=True if prefix_checked else None,
                     RC_field_semantics="Effective reasoning_content or reasoning by truthy priority",
                     prequeue_bound=True, markers_sent_to_Native=False,
                     request_constraints_owned_by_upstream_prepared_route=True)
        return {"history_render_audit": audit}
    return prequeue
