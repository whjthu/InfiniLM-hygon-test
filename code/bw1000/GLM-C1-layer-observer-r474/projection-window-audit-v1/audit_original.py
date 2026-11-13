"""Aggregate exact projection gaps on the original, independently verified GLM906 prefix."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median

R = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
PATH = R/'results/GLM474-original906-failed-prefix-consumption-v2/private-result.json'
PIN = '9e60449833c85a04fe4d1cdfbb87439745d3f23a5b4e861ad77becbb543e754a'
DETAIL = re.compile(r'detail\.layer\.(\d+)\.prepare\.projection\.N(\d+)\.K(\d+)\.(.+)')
FENCE = 'projection.end.before_original_fence'

def group_name(before, after):
    left = DETAIL.fullmatch(before)
    if not left or left[4] != FENCE:
        return None
    right = DETAIL.fullmatch(after)
    if right and right[1] == left[1] and right[4]=='quant.begin':
        return f'N{left[2]}.K{left[3]} -> N{right[2]}.K{right[3]}.quant.begin'
    ending = re.fullmatch(r'detail\.layer\.(\d+)\.(attention|FFN)\.end',after)
    if ending and ending[1]==left[1]:
        return f'N{left[2]}.K{left[3]} -> {ending[2]}.end'
    return None

def aggregate(result):
    assert result['independently_validated_prefix']==31 and not result['cross_errors']
    reports=result['rank_reports']
    assert {r['rank'] for r in reports}==set(range(16)) and all(not r['errors'] for r in reports)
    groups=defaultdict(list)
    unmatched=0
    frames=0
    for report in reports:
        capture=report['frames']
        assert capture['coverage_complete'] and len(capture['frames'])==3
        for frame in capture['frames']:
            frames+=1
            assert frame['global_tp_rank']==report['rank']
            assert len(frame['adjacent_mark_windows'])==frame['events']-1
            for row in frame['adjacent_mark_windows']:
                key=group_name(row['before'],row['after'])
                if key is None:
                    unmatched+=row['before'].endswith('.'+FENCE)
                    continue
                h=row['host_delta_s']*1e6
                s=row['stream_delta_s']*1e6
                assert all(math.isfinite(v) and v>=0 for v in (h,s))
                groups[key].append((report['rank'],h,s))
    assert frames==48 and groups
    def summarize(rows):
        return dict(n=len(rows),ranks=len({x[0] for x in rows}),
             host_median_us=median(x[1] for x in rows),stream_median_us=median(x[2] for x in rows),
             paired_host_minus_stream_median_us=median(x[1]-x[2] for x in rows),
             stream_gt_host_count=sum(x[2]>x[1] for x in rows))
    values={key:summarize(rows) for key,rows in groups.items()}
    return dict(frames=frames,unmatched_projection_end_marks=unmatched,
                groups=dict(sorted(values.items(),key=lambda x:-x[1]['stream_median_us'])),
                original_private_result_SHA256=PIN,GPU_steps=0,model_loads=0)

if __name__=='__main__':
    raw=PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==PIN
    print(json.dumps(aggregate(json.loads(raw)),sort_keys=False))
