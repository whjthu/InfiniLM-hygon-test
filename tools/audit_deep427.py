"""Read sealed original Deep427 evidence; never execute producer or Native code."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def audit(path):
    raw = path.read_bytes()
    wrapper = json.loads(raw)
    seal = wrapper['sealed_archive']
    original = seal['text'].encode()
    assert len(original) == seal['bytes']
    assert hashlib.sha256(original).hexdigest() == seal['SHA256']
    archive = json.loads(original)
    reader = Path(__file__).resolve().parents[1] / 'code/bw1000/public-Deep-native-provider-r427/analysis-terminal-v1/analyze_archive.py'
    assert hashlib.sha256(reader.read_bytes()).hexdigest() == '46aa8f6286f1c623955dfd3b0dd23e5dc263821518d24bef72cb6b49a55859a3'
    spec = importlib.util.spec_from_file_location('deep427_archive_reader', reader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.analyze(archive)
    owner = json.loads(next(v['text'] for n, v in archive['files'].items() if n.endswith('/owner.json')))
    details = []
    for request in owner['requests']:
        ids, times = request['generated_token_ids'], request['backend_token_monotonic']
        assert len(ids) == len(times) and all(type(v) is int and v >= 0 for v in ids)
        assert ids[-1] == 1 and 1 not in ids[:-1]
        assert all(type(v) in (int, float) and math.isfinite(v) for v in times)
        assert request['started_monotonic'] <= times[0] <= times[-1] <= request['ended_monotonic']
        assert all(a <= b for a, b in zip(times, times[1:]))
        # worker records immediately before actual_emit; NativeRequest.emit
        # records a second clock inside the same synchronous call.
        emit_times = request['final_metadata']['token_monotonic']
        assert len(emit_times) == len(times)
        assert all(type(v) in (int, float) and math.isfinite(v) for v in emit_times)
        assert all(a <= b <= c for a, b, c in zip(
            times, emit_times, times[1:] + [request['ended_monotonic']]))
        past = 0
        stages = request['Native_continuation']
        assert len(stages) == 10
        for stage in stages[:-1]:
            assert stage['kind'] == 'prefill' and stage['past'] == past
            assert all(type(stage[k]) is int for k in ('past', 'tokens', 'total'))
            assert 0 < stage['tokens'] <= 1024
            past += stage['tokens']
            assert stage['total'] == past <= 32768
            assert stage['emitted'] == (past == request['prompt_tokens'])
        decode = stages[-1]
        assert decode == dict(kind='decode_summary', calls=len(ids) - 1,
                              first_past=past, last_total=past + len(ids) - 1)
        assert past == request['prompt_tokens']
        assert past + request['max_new_tokens'] <= 32768
        assert request['actual_forward_completed'] == len(stages) - 1 + decode['calls']
        assert request['Native_forward_host_summary']['decode_calls'] == decode['calls']
        details.append(dict(request_id=request['request_id'], token_timeline_entries=len(times),
                            EOS_only_last=True, monotone_finite_clock=True,
                            synchronous_emit_clock_enclosure=True,
                            contiguous_prefill_calls=len(stages) - 1, decode_calls=decode['calls'],
                            final_cache_total=decode['last_total']))
    memory = owner['GPU_memory_after_load']
    assert len(memory) == 8 and all(0 < free <= total for free, total in memory)
    result.update(archive_SHA256=seal['SHA256'], local_archive_wrapper_SHA256=hashlib.sha256(raw).hexdigest(),
                  audit_reader_SHA256=hashlib.sha256(reader.read_bytes()).hexdigest(),
                  additional_root_checks=details, minimum_loaded_free_bytes=min(v[0] for v in memory),
                  qualification='Two ordinary public requests/quality/normal eight-owner lifecycle only; buffered P0, no new speedup or incremental-client SLO')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.archive)
    with args.output.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(passed=value['passed'], HTTP=value['actual_HTTP'],
                          Native_forwards=value['Native_CAPI_forwards'], normal_owners=8)))
