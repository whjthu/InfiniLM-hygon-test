"""Verify the paired C4 entry's original terminal, mode counters and clocks."""

import json
import math
from pathlib import Path
import sys

from support import PACKAGE, sha

sys.path.insert(0, str(PACKAGE))
from pair_control import LAYERS, MODES, TOPK, eligible, paired_decode


class TerminalNotVisible(RuntimeError):
    pass


def collect(output, step, epoch):
    path = Path(output) / "entry-report.json"
    q = json.loads(path.read_bytes())
    if not q.get("completed"):
        raise TerminalNotVisible("Original terminal JSON still pending")
    assert q["passed"] and q["identity"]["actual_step"] == step and q["identity"]["gpu_start_epoch"] == epoch
    assert q["HTTP"] == 0 and not q["Model_READY"] and q["loaded_key_count"] == 67569
    assert q["normal_Native_shutdown"] and q["engine_worker_shutdown_completed"] and q["observed_all8_threads_exited"]
    assert q["precision_policy"]["policy"] == 0 and not q["precision_policy"]["F64_allowed"]
    assert len(q["owner_thread_observation"]["owners"]) == 8
    assert q["compact_returned_default_off"] and not q["quality_qualification"]
    assert len(q["GPU_memory_after_load"]) == 8 and all(f >= 16*1024**3 for f, _ in q["GPU_memory_after_load"])
    requests = q["requests"]
    assert len(requests) == len(MODES) and [v["compact"] for v in requests] == list(MODES)
    capi = q["actual_single_CAPI_calls"]
    offset = 0
    summaries = []
    for i, r in enumerate(requests):
        assert r["index"] == i and r["warm"] is (i < 2) and r["normal_reset_after"]
        ids = r["generated_token_ids"]
        clock = r["token_monotonic"]
        assert 1 <= len(ids) <= 128 and len(clock) == len(ids)
        assert all(type(t) is int and 0 <= t < 129280 for t in ids)
        assert all(math.isfinite(t) for t in clock)
        assert all(a < b for a, b in zip(clock, clock[1:]))
        eos = ids[-1] == 1
        assert ids.count(1) == int(eos) and r["natural_EOS"] is eos and r["cap"] is not eos
        assert eos or len(ids) == 128
        for key in ("GPU_free_total_before", "GPU_free_total_after"):
            assert len(r[key]) == 8 and all(f >= 16*1024**3 for f, _ in r[key])
        count = len(ids) - 1
        events = capi[offset:offset+9+count]
        assert len(events) == 9+count
        assert [v["tokens"] for v in events] == [1024]*8+[297]+[1]*count
        assert all(v["rc"] == 0 and v["count"] == 1 and math.isfinite(v["seconds"]) and v["seconds"] >= 0
                   for v in events)
        assert r["total_prefill_CAPI_s"] == sum(v["seconds"] for v in events[:9])
        decode = r["decode_calls"]
        assert decode == [dict(past=8489+n, CAPI_s=v["seconds"]) for n,v in enumerate(events[9:])]
        offset += 9+count
        active = [v["past"] for v in decode if r["compact"] and eligible(v["past"])]
        expected = [len(active)*LAYERS, LAYERS*sum(p//4-TOPK for p in active)]
        assert r["compact_delta_per_owner"] == [expected]*8
        visible = len(ids) - int(eos)
        emitter_tok_s = (visible-1)/(clock[visible-1]-clock[0]) if visible > 1 else None
        summaries.append(dict(index=i, compact=r["compact"], warm=r["warm"], visible_tokens=visible,
                              natural_EOS=eos, decode_tok_s=emitter_tok_s,
                              decoder_capi_calls=count, compact_layers_executed_per_owner=expected[0],
                              historical_rows_avoided_per_owner=expected[1]))
    assert offset == len(capi) == q["HC_dispatch_checked_forwards"] == q["layout_dispatch_checked_forwards"]
    matched = [paired_decode(requests[2], requests[3]), paired_decode(requests[5], requests[4])]
    assert q["paired_decode"] == matched
    return dict(entry_report_SHA256=sha(path), normal_Native_shutdown=True, normal8owner_exit=True,
                logical_requests=len(requests), Native_CAPI_calls=offset, natural_EOS=sum(r["natural_EOS"] for r in requests),
                caps=sum(r["cap"] for r in requests), request_summary=summaries, matched_prefix_decode=matched,
                HTTP=0, exclusive_GPUbusy=False, quality_qualification=False,
                clock_scope="CAPI wall per matched state and native emitter token span, no HTTP client clock.")
