"""Compare the compact C4 bank in one loaded, quiescent Native context."""

import ctypes as C
import hashlib
import json
import statistics
import time

from profile_control import PROMPT_SHA, Request

MODES = (False, True, False, True, True, False)
LAYERS = 21
TOPK = 512


def eligible(past):
    return past // 4 > TOPK and past % 4 != 3


class CompactPolicy:
    def __init__(self, lib):
        self.setter = lib.deep_c4_compact_bank_set_v1
        self.setter.argtypes = [C.c_int]
        self.setter.restype = C.c_int
        self.reader = lib.deep_c4_compact_bank_stats_v1
        self.reader.argtypes = []
        self.reader.restype = C.c_char_p

    def set(self, enabled):
        if self.setter(int(enabled)):
            raise RuntimeError("Native compact bank mode transition failed")

    def snapshot(self, enabled):
        raw = self.reader()
        if not raw:
            raise RuntimeError("Native compact bank counters unavailable")
        state = json.loads(raw)
        owners = state["owners"]
        assert state["enabled"] is enabled and [x["device"] for x in owners] == list(range(8))
        return [(x["calls"], x["historical_rows_avoided"]) for x in owners]


def paired_decode(off, on):
    """Only compare positions whose full sampled prefix still matches."""
    prefix = 0
    for a, b in zip(off["generated_token_ids"], on["generated_token_ids"]):
        if a != b:
            break
        prefix += 1
    off_calls, on_calls = off["decode_calls"], on["decode_calls"]
    matched = min(prefix, len(off_calls), len(on_calls))
    partitions = {"eligible": [], "boundary": [], "other": []}
    for a, b in zip(off_calls[:matched], on_calls[:matched], strict=True):
        past = a["past"]
        assert past == b["past"]
        group = "eligible" if eligible(past) else "boundary" if past % 4 == 3 else "other"
        partitions[group].append((a["CAPI_s"], b["CAPI_s"]))
    return dict(off_index=off["index"], on_index=on["index"], matched_decode_calls=matched,
                same_all_generated_ids=off["generated_token_ids"] == on["generated_token_ids"],
                by_geometry={k: dict(count=len(values),
                    off_median_ms=1000*statistics.median(a for a, _ in values) if values else None,
                    on_median_ms=1000*statistics.median(b for _, b in values) if values else None,
                    paired_on_minus_off_median_ms=1000*statistics.median(b-a for a,b in values) if values else None)
                    for k, values in partitions.items()})


def run_loaded(context, prompt, report, save, *, generate, reasons, control, finite,
               deadline, policy_factory=CompactPolicy):
    """Measure two warm-ups followed by OFF/ON/ON/OFF, with observers disabled."""
    assert len(prompt) == 8489
    assert hashlib.sha256(json.dumps(list(prompt), separators=(",", ":")).encode()).hexdigest() == PROMPT_SHA
    assert context.cfg.get("native_c4_index_topk", TOPK) == TOPK
    n = context.cfg["num_hidden_layers"]
    assert n == 43 and context.cfg["compress_ratios"][:n].count(4) == LAYERS
    lib = context.binding.lib
    for name in ("deep_owner_arm_r8900", "deep_attention_inner_arm_r9193"):
        context.binding.check(getattr(lib, name)(0))
    policy = policy_factory(lib)
    policy.snapshot(False)
    rows = report["actual_single_CAPI_calls"]
    records = report["requests"] = []
    try:
        for index, enabled in enumerate(MODES):
            if time.time() >= deadline:
                raise TimeoutError("original compact bank residency deadline")
            context.check_thread()
            request = Request(prompt)
            row = dict(index=index, warm=index < 2, compact=enabled, decode_calls=[])
            records.append(row)
            row["GPU_free_total_before"] = context.floor()
            before = policy.snapshot(False if index == 0 else MODES[index-1])
            start = len(rows)
            attempt = []

            def forward(*args):
                tokens, past = len(args[0]), args[3][0]
                assert args[2] == [0]
                attempt.append((tokens, past))
                return control.forward(*args)

            with control.request(0):
                policy.set(enabled)
                reason, _ = generate(request, forward, context.binding.check, finite, reasons,
                                     context=32768, chunk_size=1024, eos=1, vocab=context.cfg["vocab_size"],
                                     absolute_deadline=deadline)
            assert control.normal_reset_after and not control.poisoned
            hc = control.last_counts
            assert len(hc) == 8
            for device, value in enumerate(hc):
                assert value["owner"] == device and value["created"] == value["ready"] == 86
                assert not any(value["fullpre"].values()) and value["legacy"]["f32_t1"] == 0
            calls = rows[start:]
            assert len(calls) == len(attempt) == 9 + len(request.generated_token_ids) - 1
            assert [t for t, _ in attempt[:9]] == [1024] * 8 + [297]
            assert all(v["count"] == 1 and v["rc"] == 0 and v["tokens"] == t
                       for (t, _), v in zip(attempt, calls, strict=True))
            decode = row["decode_calls"] = [
                dict(past=p, CAPI_s=v["seconds"]) for (t, p), v in zip(attempt[9:], calls[9:], strict=True)
                if t == 1]
            assert [x["past"] for x in decode] == list(range(8489, 8489 + len(calls) - 9))
            active = [x["past"] for x in decode if enabled and eligible(x["past"])]
            expected = (LAYERS * len(active), LAYERS * sum(p // 4 - TOPK for p in active))
            after = policy.snapshot(enabled)
            deltas = [(end[0]-begin[0], end[1]-begin[1]) for begin,end in zip(before, after, strict=True)]
            assert deltas == [expected] * 8
            ids = request.generated_token_ids
            eos = ids[-1] == 1
            assert ids.count(1) == int(eos)
            assert (reason is reasons.EOS_TOKEN) if eos else (reason is reasons.LENGTH and len(ids) == 128)
            row.update(generated_token_ids=ids, token_monotonic=request.token_monotonic,
                       natural_EOS=eos, cap=not eos, normal_reset_after=True,
                       compact_delta_per_owner=deltas, total_prefill_CAPI_s=sum(v["seconds"] for v in calls[:9]),
                       GPU_free_total_after=context.floor())
            save()
    finally:
        if not control.poisoned:
            policy.set(False)
            policy.snapshot(False)
    report["paired_decode"] = [paired_decode(records[2], records[3]), paired_decode(records[5], records[4])]
    report["compact_returned_default_off"] = True
    report["quality_qualification"] = False
    report["HTTP"] = 0
    save()
    return report
