"""Audit the original isolated vision result after Slurm accounting is terminal."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import math
import statistics
from collections import Counter


ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-vision-block-stream-sync-v1-aead72ae722c454fb4c3c7c707bfca93')
NAME='training-qwen-vision-block-stream-sync-v1'
REQUEST='aead72ae722c454fb4c3c7c707bfca93'
COLUMNS='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,AllocTRES%160'
RANKS=(1,)
ORDER=(False,True,True,False,True,False,False,True)*3


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
    t,digest=read('terminal.json',cap=2<<20)
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
    assert result['actual_backward']==result['actual_updates']==0 and result['actual_forwards']==24
    assert result['frozen_visual_weights']==351 and result['vision_weight_bytes']==1152776672
    assert [rank['rank'] for rank in result['ranks']]==list(RANKS)
    item,=result['ranks']
    assert item['inputs']['pixel']['shape']==[640,1536]
    rows=item['runs']
    assert len(rows)==24 and [r['sync_blocks'] for r in rows]==list(ORDER)
    assert all(r['block_barrier_count']==(27 if r['sync_blocks'] else 0) for r in rows)
    canonical=list(rows[0]['trace'])
    assert len(canonical)==60 and all(list(row['trace'])==canonical for row in rows)
    values=[{name:signature(v) for name,v in row['trace'].items()} for row in rows]
    final=('final_pooler','final_deepstack_0','final_deepstack_1','final_deepstack_2')
    assert all(row['trace'][name]['requires_grad'] for name in final for row in rows)
    assert all(row['forward_seconds']>0 and math.isfinite(row['forward_seconds']) for row in rows)
    arms={}
    for synced in (False,True):
        indices=[i for i,row in enumerate(rows) if row['sync_blocks']==synced]
        initial=values[indices[0]]
        count=Counter()
        extremes={key:dict(different=0,max_abs=0.0,relative_L2=0.0) for key in final}
        for i in indices:
            first=next((name for name in canonical if initial[name]!=values[i][name]),None)
            count[first or 'same']+=1
            for key in final:
                for part,anchor in (('delta_vs_first_same_arm',initial),('delta_vs_first_default',values[0])):
                    d=rows[i][part][key]
                    assert type(d['different']) is int and d['different']>=0
                    assert (d['different']==0)==(values[i][key]==anchor[key])
                    for k,v in d.items():
                        assert v is not None and v>=0 and math.isfinite(v)
                        if part=='delta_vs_first_same_arm':
                            extremes[key][k]=max(v,extremes[key][k])
        arms['per_block_sync' if synced else 'native_enqueue']=dict(samples=len(indices),
            first_trace_difference_count=dict(count),
            output_drift_samples=sum(any(values[i][key]!=initial[key] for key in final) for i in indices),
            unique_final_signatures={key:len({values[i][key][0] for i in indices}) for key in final},
            max_output_delta_vs_same_arm_baseline=extremes,
            forward_median_ms=1000*statistics.median(rows[i]['forward_seconds'] for i in indices))
    return out|dict(state='ACTUAL_QWEN_VISION_BLOCK_SYNC_PAIRED_DATA',step=gpu[0],result_SHA256=pin,
        identity=result['identity'],weight_load_seconds=result['vision_load_seconds'],
        actual_total_forwards=24,measurements=arms,formal_training=False,
        ratio_sync_over_native_forward_median=arms['per_block_sync']['forward_median_ms']/arms['native_enqueue']['forward_median_ms'])


if __name__=='__main__':
    print(json.dumps(checked(),allow_nan=False))
