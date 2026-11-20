"""Capture the first historical transport disagreement without changing sampling."""
import hashlib
import heapq
import json
import math
from pathlib import Path
import struct


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def attach(request, pin, *, rank, arm, past, vocab, read_logits):
    """Called after the native greedy choice is broadcast and recorded."""
    if (rank != 0 or request["ordinal"] != pin["ordinal"]
            or len(request["generated_token_ids"]) != pin["sample"]):
        return
    if "transport_rank0_sample43" in request:
        raise ValueError("duplicate sample 43 evidence")
    if arm not in pin["historic_argmax_SHA256"]:
        raise ValueError("unknown requested transport")
    for name in ("prompt_tokens", "prompt_IDs_SHA256", "body_SHA256"):
        if request[name] != pin[name]:
            raise ValueError("different frozen ordinal 150 input")
    ids = request["generated_token_ids"]
    if any(type(token) is not int or not 0 <= token < vocab for token in ids):
        raise ValueError("bad greedy output inventory")
    expected_past = pin["prompt_tokens"] + pin["prefix_len"]
    if past != expected_past:
        raise ValueError("sample 43 did not follow 42 decode forwards")
    prefix_sha = digest(ids[:-1])
    chosen_hash = digest(ids[-1:])
    info = dict(state="PREFIX_MISMATCH", prefix_SHA256=prefix_sha, chosen_SHA256=chosen_hash,
                historical_argmax_repeated=chosen_hash == pin["historic_argmax_SHA256"][arm],
                sample=pin["sample"], past=past)
    if prefix_sha == pin["common_prefix_SHA256"]:
        raw = read_logits()
        if not isinstance(raw, bytes) or len(raw) != vocab * 2:
            raise ValueError("expected one complete BF16 logit row")
        scores = (struct.unpack("<f", b"\0\0" + raw[i:i+2])[0] for i in range(0, len(raw), 2))

        def values():
            for token, score in enumerate(scores):
                if not math.isfinite(score):
                    raise ValueError("non-finite BF16 logit")
                yield token, score

        top = heapq.nsmallest(min(8, vocab), values(), key=lambda item: (-item[1], item[0]))
        if len(top) < 2 or top[0][0] != ids[-1]:
            raise ValueError("native greedy choice does not match inspected logit row")
        info.update(state="MATCHED_PREFIX", logits_SHA256=hashlib.sha256(raw).hexdigest(),
                    dtype="BF16", vocabulary=vocab, top_tokens_private=top,
                    top1_minus_top2=top[0][1] - top[1][1])
    request["transport_rank0_sample43"] = info


def preserve_missing(request, pin, rank):
    if rank != 0 or request["ordinal"] != pin["ordinal"]:
        return
    if "transport_rank0_sample43" not in request:
        if len(request["generated_token_ids"]) >= pin["sample"]:
            raise ValueError("sampling hook did not run")
        request["transport_rank0_sample43"] = dict(
            state="NO_SAMPLE", observed_tokens=len(request["generated_token_ids"]))


def load_pin(source):
    return json.loads(Path(source).with_name("OBSERVATION_BINDING.json").read_bytes())
