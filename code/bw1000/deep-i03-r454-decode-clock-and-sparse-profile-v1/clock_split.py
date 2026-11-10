"""Partition the original owner clock without replaying either HTTP request."""

import hashlib
import json
import math


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def pinned(folder, name, expected):
    with (folder / name).open("rb") as f:
        raw = f.read(8 * 1024**2 + 1)
    require(len(raw) <= 8 * 1024**2 and hashlib.sha256(raw).hexdigest() == expected,
            "original artifact hash differs")
    return raw


def split_owner_clock(owner, reads, verified):
    ticks = owner["backend_token_monotonic"]
    start, end = owner["started_monotonic"], owner["ended_monotonic"]
    require(all(type(x) in (float, int) and math.isfinite(x) for x in [start, *ticks, end]),
            "invalid owner clock")
    require(2 <= len(ticks) == owner["actual_generated_tokens"] == verified["generated_including_EOS"]
            and owner["native_finish_reason"] == "eos_token"
            and verified["score"]["natural_EOS"] is True
            and owner["normal_reset_before"] is True and owner["normal_reset_after"] is True
            and start <= ticks[0] and ticks[-1] <= end
            and all(b > a for a, b in zip(ticks, ticks[1:])), "owner lifecycle differs")
    counters = owner["Native_forward_host_summary"]
    prefill = counters["prefill_calls"]
    decode = counters["decode_calls"]
    require(type(prefill) is int and type(decode) is int
            and prefill == (owner["prompt_tokens"] + 1023) // 1024 and decode == len(ticks)-1
            and prefill+decode == owner["actual_forward_attempts"] == owner["actual_forward_completed"]
            == verified["native_forwards"], "Native forward counts differ")
    prefill_s, decode_s = counters["prefill_s"], counters["decode_s"]
    require(all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in (prefill_s,decode_s)),
            "invalid cumulative wrapper duration")
    ttft = ticks[0]-start
    decode_span = ticks[-1]-ticks[0]
    require(prefill_s <= ttft and decode_s <= decode_span
            and math.isclose(ttft, verified["backend_TTFT_s"], abs_tol=1e-7),
            "nested clocks exceed original owner span")
    non_eos_count = len(ticks)-1
    require(non_eos_count >= 2 and non_eos_count == verified["backend_non_eos_token_count"],
            "non-EOS denominator differs")
    rate = (non_eos_count-1)/(ticks[-2]-ticks[0])
    require(math.isclose(rate, verified["backend_non_eos_decode_tok_s"], rel_tol=1e-9),
            "non-EOS rate differs")
    require(reads[0]["stage"] == "send_enter" and reads[-1]["stage"] == "read" and reads[-1]["size"] == 0,
            "wrong socket clock boundaries")
    send = reads[0]["ns"]/1e9
    eof = reads[-1]["ns"]/1e9
    first_text = send + verified["socket_first_text_s"]
    require(send <= start <= ticks[0] <= first_text <= eof and end <= eof,
            "socket and backend clocks not ordered")
    return {
        "case_id": verified["case_id"], "prompt_tokens": owner["prompt_tokens"],
        "prefill_forward_calls": prefill, "decode_forward_calls_including_eos": decode,
        "client_send_to_owner_start_s": start-send,
        "owner_start_to_first_emit_s": ttft,
        "prefill_policy_forward_host_s": prefill_s,
        "prefill_outside_policy_forward_s": ttft-prefill_s,
        "first_backend_emit_to_first_client_text_s": first_text-ticks[0],
        "first_emit_to_eos_emit_s": decode_span,
        "decode_policy_forward_host_s": decode_s,
        "decode_outside_policy_forward_s": decode_span-decode_s,
        "decode_policy_forward_host_fraction": decode_s/decode_span,
        "mean_decode_policy_forward_ms_including_eos": 1000*decode_s/decode,
        "mean_decode_outside_policy_forward_ms_including_eos": 1000*(decode_span-decode_s)/decode,
        "eos_emit_to_owner_end_s": end-ticks[-1],
        "client_send_to_eof_s": eof-send,
        "current_mean_non_eos_interval_ms": 1000/rate,
        "mean_interval_reduction_to_15tok_s_boundary_ms": 1000/rate-1000/15,
        "boundary_speedup_ratio": 15/rate,
    }


def analyze(folder, receipt):
    require(receipt["step"] == "1087656.58" and receipt["normal_service_completion"] is True
            and receipt["independent_post_clean"] is True
            and receipt["terminal"]["state"] == "COMPLETED", "wrong terminal evidence")
    pins = receipt["artifacts_SHA256"]
    name = "i03-deep-r454-http-first2-v1/owner.json"
    owner = json.loads(pinned(folder,name,pins[name]))
    require(owner["identity"]["actual_step"] == receipt["step"] and owner["completed"] is True
            and owner["normal_Native_shutdown"] is True
            and len(owner["requests"]) == len(receipt["requests"]) == 2, "wrong owner record")
    result = []
    for i, (request, verified) in enumerate(zip(owner["requests"],receipt["requests"])):
        require(request["r454_index"] == i and request["r454_mode"] == 0
                and hashlib.sha256(json.dumps(request["generated_token_ids"],separators=(",",":")).encode()).hexdigest()
                == verified["generated_IDs_SHA256"], "request order or generation identity differs")
        name = f"request-{i}.clock.jsonl"
        reads = [json.loads(line) for line in pinned(folder,name,pins[name]).splitlines()]
        result.append(split_owner_clock(request,reads,verified))
    return {"step": receipt["step"], "requests": result, "owner_artifact_SHA256": pins[
        "i03-deep-r454-http-first2-v1/owner.json"],
        "scope": "Same-process host policy.forward wrapper and complementary owner span. Not exclusive device kernels, CCL or busy; includes final EOS for wrapper partition.",
        "new_model_loads": 0, "new_HTTP": 0, "new_Slurm_steps": 0}
