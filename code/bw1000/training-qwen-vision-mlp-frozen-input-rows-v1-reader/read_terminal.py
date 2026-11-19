"""Audit the original isolated vision result after Slurm accounting is terminal."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import statistics
import math


ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-vision-mlp-frozen-input-rows-v1-824845f0cab243ba9a8b0bd39485356f')
NAME='training-qwen-vision-mlp-frozen-input-rows-v1'
REQUEST='824845f0cab243ba9a8b0bd39485356f'
COLUMNS='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,AllocTRES%160'
ARMS=(False,True,True,False,True,False,False,True)


def read(name,cap=524288):
    with (ROOT/name).open('rb') as f:
        data=f.read(cap+1)
    assert 0<len(data)<=cap
    return json.loads(data),hashlib.sha256(data).hexdigest()


def signature(value):
    assert type(value['requires_grad']) is bool
    assert re.fullmatch('[a-f0-9]{64}',value['SHA256'])
    assert len(value['shape'])==len(value['stride'])
    return value['SHA256'],value['shape'],value['dtype']


def checked():
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    stdout=subprocess.check_output(['sacct','-n','-P','-j','1103606','--format='+COLUMNS],text=True,timeout=12)
    rows=[line.split('|') for line in stdout.splitlines()
          if line.split('|')[1] in {NAME,NAME+'-pre',NAME+'-post',NAME+'-failure-post'}]
    out=dict(utc=stamp,request=REQUEST,accounting_columns=COLUMNS.split(','),accounting=rows,
             remote_intent=(ROOT/'intent.json').is_file())
    if not (ROOT/'terminal.json').is_file():
        return out|dict(state='AWAITING_ORIGINAL_TERMINAL')
    t,digest=read('terminal.json')
    assert t['request']==REQUEST
    out.update(terminal_SHA256=digest,original_GPU_passed=t.get('gpu_passed'),original_post_clean=t.get('clean_post'))
    if not(t['gpu_passed'] and t['clean_post']):
        log=ROOT/(NAME+'.log')
        if log.is_file():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-8192))
                raw=f.read()
            matches=re.findall(rb'(?m)^(ValueError|AssertionError|RuntimeError|KeyError|ImportError|TypeError|OSError|NotImplementedError): ([A-Z][A-Z0-9_]{3,100})',raw)
            out['bounded_error_codes']=[[k.decode(),v.decode()] for k,v in matches[-5:]]
            out['traceback_type_only']=[v.decode() for v in re.findall(rb'(?m)^([A-Za-z]+Error):',raw)][-5:]
        return out|dict(state='ORIGINAL_GPU_OR_CLEANUP_FAILED',controller_error=t.get('error_class'))
    gpu,=[row for row in rows if row[1]==NAME]
    assert len(rows)==3 and all(r[2:4]==['COMPLETED','0:0'] for r in rows)
    assert gpu[0]==t['step'] and 'gres/dcu=1' in gpu[5].split(',')
    assert t['before']['pci']==t['after']['pci'] and all(row['passed'] for row in t['runs'])
    assert t['before']['boot_id']==t['after']['boot_id'] and t['after']['kfd_holder_count']==0
    result,pin=read('result.json')
    assert result==t['worker_result'] and result['identity']['step']==gpu[0]
    assert result['identity']['PCI'] in t['before']['pci'] and result['identity']['node']=='f13r1n10'
    assert result['actual_backward']==result['actual_optimizer']==0
    assert result['original_visual_image_rank']==1 and result['original_full_vision_forwards']==1
    assert result['actual_mlp_input']['shape']==[640,1152] and result['actual_mlp_input']['dtype']=='torch.bfloat16'
    assert result['row_chunk']==128
    records=result['records']
    assert len(records)==8 and [row['split'] for row in records]==list(ARMS)
    stages=('fc1','activation','fc2')
    geometry={'fc1':[640,4304],'activation':[640,4304],'fc2':[640,1152]}
    groups={split:{key:[] for key in stages} for split in (False,True)}
    host_comparisons={split:{key:[] for key in stages} for split in (False,True)}
    for row in records:
        assert row['stage_calls']==(5 if row['split'] else 1) and set(row['stages'])==set(stages)
        for key, tensor in row['stages'].items():
            assert tensor['shape']==geometry[key] and tensor['dtype']=='torch.bfloat16'
            groups[row['split']][key].append(signature(tensor))
            values=tensor['vs_float32_cpu_then_BF16']
            assert math.isfinite(values['max_abs']) and values['max_abs']>=0
            assert values['relative_L2'] is None or math.isfinite(values['relative_L2']) and values['relative_L2']>=0
            host_comparisons[row['split']][key].append(values)
    unique={}
    for split,rows_for_arm in groups.items():
        key='row128' if split else 'native'
        unique[key]={name:len({sig[0] for sig in rows_for_arm[name]}) for name in stages}
    timing=result['timing']
    assert [r['split'] for r in timing]==[False,True,True,False]
    for row in timing:
        assert row['warm']==5 and row['measured']==len(row['event_ms'])==12
        assert all(math.isfinite(x) and x>0 for x in row['event_ms'])
        assert row['median_event_ms']==statistics.median(row['event_ms'])
    ratio=statistics.mean(r['median_event_ms'] for r in timing if r['split'])/statistics.mean(
          r['median_event_ms'] for r in timing if not r['split'])
    return out|dict(state='ACTUAL_QWEN_NATIVE_MLP_FROZEN_OPERAND_PAIR_VERIFIED',step=gpu[0],result_SHA256=pin,
        identity=result['identity'],observed_input=result['actual_mlp_input'],two_fc_layouts=[geometry['fc1'],geometry['fc2']],
        scope='same genuine BF16 Qwen vision block-zero MLP input, four unchunked and four row128 observations',
        actual_full_vision_forwards=1,standalone_observations=8,distinct_outputs=unique,
        first_native_repeated_divergence=next((k for k in stages if unique['native'][k]>1),None),
        first_row128_repeated_divergence=next((k for k in stages if unique['row128'][k]>1),None),
        host_reference_errors=host_comparisons,ratio_row128_over_native_event_median=ratio,
        timing_block_medians_ms=[r['median_event_ms'] for r in timing],formal_training=False)


if __name__=='__main__':
    print(json.dumps(checked(),allow_nan=False))
