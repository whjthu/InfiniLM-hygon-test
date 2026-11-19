"""Invoke the frozen comparator with its full original decision and denominator."""
import hashlib
import json
from pathlib import Path
import sys

import torch
import hygon_sdxl_compare_first_step as frozen


ROOT=Path(__file__).resolve().parent


def main():
    assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
    src=ROOT/'candidate/replay'
    report=ROOT/'comparison.json'
    sys.argv=['frozen','--reference','/reference/reference.pt','--policy','/reference/policy.json',
        '--observed',str(src/'observed.pt'),'--receipt',str(src/'summary.json'),'--output',str(report),'--accelerator','hip']
    decision=frozen.main()
    raw=report.read_bytes()
    full=json.loads(raw)
    assert decision in (0,1) and full['passed']==(decision==0) and 'structural_failure' not in full
    fields=['prediction','gradients_before_clip','gradients_after_clip','parameter_update','optimizer_exp_avg','optimizer_exp_avg_sq']
    metrics={key:{k:v for k,v in full['metrics'][key].items() if k not in ('per_tensor','worst_relative_l2')} for key in fields}
    total={k:len(full['metrics'][k]['per_tensor']) for k in fields[1:]}
    assert set(total.values())=={1120}
    summary=dict(numeric_passed=full['passed'],checks=full['checks'],metrics=metrics,parameter_counts=total,
                 full_private_comparison_SHA256=hashlib.sha256(raw).hexdigest(),GPU_initialized=torch.cuda.is_initialized())
    with (ROOT/'comparison-summary.json').open('x',encoding='utf-8') as f:
        json.dump(summary,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    main()
