"""Audit the original isolated vision result after Slurm accounting is terminal."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from collections import Counter
import math


ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-vision-all-native-boundaries-v1-c55289e3fbf3409e84d15d321f6eaadb')
NAME='training-qwen-vision-all-native-boundaries-v1'
REQUEST='c55289e3fbf3409e84d15d321f6eaadb'
COLUMNS='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,AllocTRES%160'
RANKS=(1,)


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
    t,digest=read('terminal.json',cap=4<<20)
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
    result,pin=read('result.json',cap=2<<20)
    assert result==t['worker_result'] and result['identity']['step']==gpu[0]
    assert result['identity']['PCI'] in t['before']['pci'] and result['identity']['node']=='f13r1n10'
    assert result['actual_backward']==result['actual_updates']==0 and result['actual_forwards']==8
    assert result['frozen_visual_weights']==351 and result['vision_weight_bytes']==1152776672
    assert [rank['rank'] for rank in result['ranks']]==list(RANKS)
    item,=result['ranks']
    assert item['inputs']['pixel']['shape']==[640,1536]
    runs=item['runs']
    assert len(runs)==8 and all(row['connected'] is True for row in runs)
    final=('final_pooler','final_deepstack_0','final_deepstack_1','final_deepstack_2')
    boundaries=('entry','attention_proj_entry','after_attention_add','norm1','attention_qkv',
                'attention_proj','norm2','mlp_fc1','mlp_activation','mlp_fc2','output')
    expected={'patch_conv3d','patch_output','position_embedding','rotary_output','cu_seqlens',
              'position_cosine','position_sine','vision_merger',*final}
    expected.update(f'deepstack_merger_{i}' for i in range(3))
    expected.update(f'block_{i:02d}_{name}' for i in range(27) for name in boundaries)
    canonical=list(runs[0]['trace'])
    assert len(canonical)==312 and set(canonical)==expected
    anchor={key:signature(v) for key,v in runs[0]['trace'].items()}
    count=Counter()
    details=[]
    outputs=[]
    for run_no,row in enumerate(runs):
        assert list(row['trace'])==canonical and all(row['trace'][key]['requires_grad'] for key in final)
        values={key:signature(v) for key,v in row['trace'].items()}
        assert all(vectors_equal(values, i) for i in range(1,27))
        first=next((key for key in canonical[:-4] if values[key]!=anchor[key]),None)
        item=row['first_difference']
        assert (item is None)==(first is None)
        count[first or 'same']+=1
        if item is not None:
            assert item['boundary']==first and item['shape']==values[first][1] and item['dtype']==values[first][2]
            assert item['changed_elements']>=0 and item['max_abs']>=0 and math.isfinite(item['max_abs'])
            assert item['relative_L2'] is None or item['relative_L2']>=0 and math.isfinite(item['relative_L2'])
            details.append(dict(repeat=run_no,**item))
        outputs.append({key:values[key][0] for key in final})
    return out|dict(state='ACTUAL_QWEN_VISION_ALL27_NATIVE_BOUNDARIES',step=gpu[0],result_SHA256=pin,
        identity=result['identity'],vision_provider=result['vision_provider'],vision_weight_load_seconds=result['vision_load_seconds'],
        peak_allocated_bytes=result['peak_allocated_bytes'],observed_blocks=27,native_boundaries_before_consumed_outputs=308,actual_forwards=8,
        first_difference_counts=dict(count),first_differences=details,
        unique_consumed_outputs={key:len({row[key] for row in outputs}) for key in final},
        formal_training=False)


def vectors_equal(values, index):
    return values[f'block_{index:02d}_entry']==values[f'block_{index-1:02d}_output']


if __name__=='__main__':
    print(json.dumps(checked(),allow_nan=False))
