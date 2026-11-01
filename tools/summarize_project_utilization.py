"""Summarize one sysfs/CPU snapshot; allocation is not a GPU utilization metric."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(path):
    raw = Path(path).read_bytes(); snapshot = json.loads(raw)
    assert snapshot['CPU_only_read_snapshot']
    assert all(p['rc'] == 0 for p in snapshot['samples'])
    rows = [r for p in snapshot['samples'] for r in p['rows']]
    assert len(rows) == len({r['host'] for r in rows}) == 17
    output = []
    for r in rows:
        # Each host also exposes one DRM record without accelerator counters/VRAM.
        valid = [c for c in r['cards'] if c['gpu_busy_percent'] is not None and c['vram_total_bytes'] is not None]
        omitted = [c for c in r['cards'] if c not in valid]
        assert len(valid) == 8 and all(0 <= c['gpu_busy_percent'] <= 100 for c in valid)
        assert all(c['vram_total_bytes'] >= 60*2**30 for c in valid)
        assert all(c['gpu_busy_percent'] is None and c['vram_total_bytes'] is None for c in omitted)
        assert r['GPU_runtime_initialized'] is False
        output.append(dict(host=r['host'], probe_step=r['actual_step'], accelerators=8,
                           instantaneous_gpu_busy_percent=[c['gpu_busy_percent'] for c in valid],
                           accelerator_VRAM_used_bytes=sum(c['vram_used_bytes'] for c in valid),
                           accelerator_VRAM_total_bytes=sum(c['vram_total_bytes'] for c in valid),
                           KFD_process_count=len(r['KFD_handles']), CPU_count=r['CPU_count'],
                           host_CPU_busy_percent_1s=r['host_CPU_busy_percent_1s'],
                           excluded_DRM_records_without_accelerator_counters=len(omitted)))
    gpu_values = [v for r in output for v in r['instantaneous_gpu_busy_percent']]
    return dict(UTC=snapshot['UTC'], source_local_path=str(Path(path).resolve()),
                source_SHA256=hashlib.sha256(raw).hexdigest(), CPU_only=True, GPU_initialized=False,
                nodes=17, accelerators=136, instantaneous_gpu_busy_arithmetic_mean_percent=statistics.mean(gpu_values),
                accelerators_with_nonzero_busy_counter=sum(v > 0 for v in gpu_values),
                KFD_processes=sum(r['KFD_process_count'] for r in output),
                mean_host_CPU_busy_percent_1s=statistics.mean(r['host_CPU_busy_percent_1s'] for r in output),
                per_node=output,
                scope='One point sample of sysfs GPU gauges; CPU has a one-second interval. Not time-weighted GPU utilization, kernel occupancy, HBM bandwidth or inference throughput.',
                memory_gauge_warning='Raw mem_busy_percent is not used as HBM utilization; it reads100 even on idle cards in this snapshot.',
                interpretation='Kimi539 was loading weights with four KFD processes/32allocatedcards; snapshot precedes GLM828 start20:31:35UTC. Dedicated48cardoperatorpool was idle. Source preparation/allocation/load must not be called busy GPU compute.',
                next_action='Continue originalKimi539; GLM828 later started independently; prepare separateDeep353 current8K attribution and GLM354 smallestmemory-feasibleC2 systemconsumer. No equivalent parked kernel replay to fill cards.')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=summarize(a.input)
    with a.output.open('x',encoding='utf8',newline='\n') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps({k:result[k] for k in ['UTC','nodes','accelerators','instantaneous_gpu_busy_arithmetic_mean_percent','KFD_processes']}))
