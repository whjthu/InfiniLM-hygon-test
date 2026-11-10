"""Capture a real localhost P1 stream and reconcile it with its fixed R454 owner."""

import http.client
import json
import math
import socket
import time

from pilot_inputs import compact, require, sha


def capture(port, raw, deadline, wire, clocks):
    """Write response bytes and socket read clocks incrementally to private files."""
    require(type(port) is int and 1024 <= port <= 65535, "invalid local port")

    def remaining():
        seconds = deadline - time.monotonic()
        if not 0 < seconds:
            raise TimeoutError("original pilot deadline reached")
        return seconds

    with wire.open("xb") as output, clocks.open("xb") as trace:
        def tick(stage, **extra):
            now = time.monotonic_ns()
            trace.write(compact(dict(stage=stage, ns=now, **extra)) + b"\n")
            trace.flush()
        tick("send_enter")
        with socket.create_connection(("127.0.0.1", port), timeout=remaining()) as sock:
            request = (
                "POST /v1/chat/completions HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                "Content-Type: application/json\r\n"
                "Connection: close\r\n"
                f"Content-Length: {len(raw)}\r\n\r\n"
            ).encode("ascii")
            sock.settimeout(remaining())
            sock.sendall(request + raw)
            sock.settimeout(remaining())
            with http.client.HTTPResponse(sock) as response:
                response.begin()
                tick("headers", status=response.status, content_type=response.getheader("Content-Type"))
                offset = 0
                while True:
                    sock.settimeout(remaining())
                    data = response.read1(65536)
                    if offset + len(data) > 16 * 1024**2:
                        raise ValueError("wire exceeded finite capture cap")
                    output.write(data)
                    output.flush()
                    tick("read", offset=offset, size=len(data))
                    offset += len(data)
                    if not data:
                        break
                return response.status


def wire_identity_and_first_text(wire, reads):
    require(len(reads) >= 4 and type(reads[0]["ns"]) is int, "missing socket receipt records")
    identifiers, first_text = set(), None
    cursor = 0
    last_tick = reads[0]["ns"]
    require(reads[0]["stage"] == "send_enter" and reads[1]["stage"] == "headers"
            and reads[1]["status"] == 200
            and "text/event-stream" in (reads[1]["content_type"] or "").lower(),
            "not a successful SSE response")
    for index, read in enumerate(reads[1:], start=1):
        require(type(read["ns"]) is int and read["ns"] >= last_tick, "receipt clock went backwards")
        last_tick = read["ns"]
        if index == 1:
            continue
        require(read["stage"] == "read" and read["offset"] == cursor
                and type(read["size"]) is int and read["size"] >= 0,
                "missing or noncontiguous socket receipt")
        require(bool(read["size"]) == (index < len(reads) - 1), "incorrect EOF position")
        cursor += read["size"]
    require(cursor == len(wire), "incomplete wire capture")
    offset = 0
    iterator = iter(reads[2:])
    read = next(iterator)
    for line in wire.splitlines(keepends=True):
        offset += len(line)
        while read["offset"] + read["size"] < offset:
            read = next(iterator)
        if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
            continue
        frame = json.loads(line[6:])
        if frame.get("id"):
            identifiers.add(frame["id"])
        if first_text is None and any(any(choice.get("delta", {}).get(field)
                for field in ("content", "reasoning_content")) for choice in frame.get("choices", [])):
            first_text = (read["ns"] - reads[0]["ns"]) / 1e9
    require(len(identifiers) == 1, "foreign or missing SSE request identity")
    return identifiers.pop(), first_text


def evaluate(entry, case, wire, reads, owner, *, decode_sse, grade):
    """Keep endpoint correctness, socket text latency, and backend token clocks separate."""
    request_id, socket_text = wire_identity_and_first_text(wire, reads)
    message, finish, usage = decode_sse(wire)
    history = owner["admission"]["history_render_audit"]
    require(owner["request_id"] == request_id and owner["r454_mode"] == 0
            and owner["r454_r366_fixed"] == 0 and owner["r454_quiescent"] is True
            and owner["r454_poisoned"] is False
            and history["qualified"] is True and history["prequeue_bound"] is True
            and history["pending_thinking_prefix_open"] is False
            and history["output_budget"] == entry["output_budget"]
            and owner["admission"]["text_delivery"] == "P1"
            and owner["final_metadata"]["buffered_P0_text"] is False
            and history["Native_prompt_token_ids_SHA256"] == entry["prompt_IDs_SHA256"]
            and owner["prompt_tokens"] == entry["prompt_tokens"]
            and owner["normal_reset_before"] is True and owner["normal_reset_after"] is True
            and not owner.get("error") and not owner.get("reset_error"), "owner request proof differs")
    ids = owner["generated_token_ids"]
    ticks = owner["backend_token_monotonic"]
    start, end = owner["started_monotonic"], owner["ended_monotonic"]
    require(0 < len(ids) == len(ticks) == owner["actual_generated_tokens"] <= entry["output_budget"]
            and all(type(token) is int and 0 <= token < 129280 for token in ids)
            and all(type(tick) in (float, int) and math.isfinite(tick) for tick in [start, *ticks, end])
            and reads[0]["ns"] / 1e9 <= start <= ticks[0] <= ticks[-1] <= end <= reads[-1]["ns"] / 1e9
            and all(b > a for a, b in zip(ticks, ticks[1:])), "invalid generated IDs or backend clock")
    eos = owner["native_finish_reason"] == "eos_token"
    require((eos and ids[-1] == 1 and ids.count(1) == 1 and finish in {"stop", "tool_calls"})
            or (owner["native_finish_reason"] == "length" and 1 not in ids
                and len(ids) == entry["output_budget"] and finish in {"length", "tool_calls"}),
            "invalid EOS or cap accounting")
    require(usage == {"prompt_tokens": entry["prompt_tokens"], "completion_tokens": len(ids),
                      "total_tokens": entry["prompt_tokens"] + len(ids)}, "HTTP usage differs")
    forwards = (entry["prompt_tokens"] + 1023) // 1024 + len(ids) - 1
    coverage = owner["r454_actual_T1_coverage"]
    require(owner["actual_forward_attempts"] == owner["actual_forward_completed"] == forwards
            and coverage["passed"] is True and coverage["completed_forwards"] == forwards
            and coverage["decode_forwards"] == len(ids) - 1,
            "incomplete native continuation")
    memory = owner["GPU_free_after_request"]
    require(len(memory) == 8 and all(type(free) is int and type(total) is int
            and 16 * 1024**3 <= free <= total for free, total in memory), "post-request headroom differs")
    visible = ticks[:-1] if eos else ticks
    rate = (len(visible) - 1) / (visible[-1] - visible[0]) if len(visible) >= 2 else None
    score = grade(case, message["content"], natural_EOS=eos, cap_hit=not eos,
                  unexpected_tool=bool(message.get("tool_calls")))
    return dict(case_id=entry["case_id"], suite=entry["suite"], request_id=request_id,
                wire_SHA256=sha(wire), generated_IDs_SHA256=sha(compact(ids)),
                generated_including_EOS=len(ids), backend_non_eos_token_count=len(visible),
                backend_TTFT_s=visible[0] - start if visible else None,
                backend_ITL_s=[b - a for a, b in zip(visible, visible[1:])],
                backend_non_eos_decode_tok_s=rate, socket_first_text_s=socket_text,
                native_forwards=forwards, minimum_post_request_free_bytes=min(free for free, _ in memory),
                score={key: value for key, value in score.items() if key != "parsed_answer"})
