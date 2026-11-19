"""Audit the original isolated vision result after Slurm accounting is terminal."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-isolated-vision-native-v1-3ec7a502299a43f3b2e3f7fa160cd418')
NAME='training-qwen-isolated-vision-native-v1'
REQUEST='3ec7a502299a43f3b2e3f7fa160cd418'
COLUMNS='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,AllocTRES%160'
RANKS=(1,0,5)


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
    assert result['actual_backward']==result['actual_updates']==0 and result['actual_forwards']==12
    assert result['frozen_visual_weights']==351 and result['vision_weight_bytes']==1152776672
    assert [rank['rank'] for rank in result['ranks']]==list(RANKS)
    required={'patch_conv3d','patch_output','position_embedding','rotary_output','post_position_add',
        'first_block_cu_seqlens','first_block_cosine','first_block_sine','block_00_norm1','block_00_qkv',
        'block_00_attention','block_00_norm2','block_00_mlp','vision_merger','final_pooler'}
    required.update(f'block_{i:02d}_output' for i in range(27))
    required.update(f'deepstack_merger_{i}' for i in range(3))
    required.update(f'final_deepstack_{i}' for i in range(3))
    reports=[]
    for item in result['ranks']:
        runs=item['runs']
        assert [r['connected'] for r in runs]==[False,True,True,False]
        traces=[r['trace'] for r in runs]
        canonical=list(traces[0])
        assert set(canonical)==required and len(canonical)==48 and all(list(x)==canonical for x in traces)
        values=[{k:signature(trace[k]) for k in canonical} for trace in traces]
        def diffs(i,j):
            return [k for k in canonical if values[i][k]!=values[j][k]]
        repeat_a,repeat_b,cross=diffs(0,3),diffs(1,2),diffs(0,1)
        for key,computed in [('native_repeated_first_difference',repeat_a),('connected_repeated_first_difference',repeat_b),
                             ('native_vs_connected_first_difference',cross)]:
            assert item[key]==(computed[0] if computed else None)
        for run in runs:
            assert all(run['trace'][field]['requires_grad']==run['connected']
                       for field in ('final_pooler','final_deepstack_0','final_deepstack_1','final_deepstack_2'))
        reports.append(dict(rank=item['rank'], original_input_shapes={k:item['inputs'][k]['shape'] for k in ('pixel','grid')},
            original_pixel_dtype=item['inputs']['pixel']['dtype'],position_helper_extra_names=item['inputs']['vision_extra_names'],
            native_repeat_differences=repeat_a,connected_repeat_differences=repeat_b,cross_mode_differences=cross,
            patch_notes=[{k:v for k,v in row['trace']['patch_conv3d'].items() if k!='SHA256'} for row in runs]))
    return out|dict(state='ACTUAL_NATIVE_VISION_ISOLATED_FORWARDS_VERIFIED',step=gpu[0], result_SHA256=pin,
        native_vision_provider=result['vision_provider'],identity=result['identity'],weight_load_seconds=result['vision_load_seconds'],
        weight_tensors=351,original_weights_bytes=1152776672,accelerator_peak_allocated_bytes=result['peak_allocated_bytes'],
        vision_forwards=12,native_training_updates=0,full_model=False,reports=reports)


if __name__=='__main__':
    print(json.dumps(checked(),allow_nan=False))
