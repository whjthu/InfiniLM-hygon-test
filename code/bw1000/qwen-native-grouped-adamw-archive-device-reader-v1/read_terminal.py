"""Check the original GPU archive replay, its device identities and paired results."""
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/qwen-native-grouped-adamw-archive-device-v1-a41acf97c26242298a7094dcbff79ca2')
LABEL='qwen-native-grouped-adamw-archive-device-v1'
REQUEST='a41acf97c26242298a7094dcbff79ca2'
PARENT='1103606'
COLUMNS='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,AllocTRES%160'
SOURCE='grouped_fused_adamw.py'


def read(path,limit):
    with path.open('rb') as stream:
        raw=stream.read(limit+1)
    assert 0<len(raw)<=limit
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def main(runtime_sha):
    executed=datetime.datetime.now(datetime.timezone.utc).isoformat()
    accounting=subprocess.check_output(['sacct','-n','-P','-j',PARENT,'--format='+COLUMNS],text=True,timeout=12)
    rows=[line.split('|') for line in accounting.splitlines()
          if line.split('|')[1] in {LABEL,LABEL+'-pre',LABEL+'-post',LABEL+'-failure-post'}]
    result=dict(utc=executed,request=REQUEST,accounting_columns=COLUMNS.split(','),accounting=rows)
    if not (ROOT/'terminal.json').is_file():
        return result|dict(state='ACTIVE_OR_TERMINAL_RECEIPT_PENDING',intent=(ROOT/'intent.json').is_file())
    terminal,digest=read(ROOT/'terminal.json',131072)
    assert terminal['request']==REQUEST
    result.update(terminal_SHA256=digest,original_GPU_passed=terminal.get('gpu_passed'),
                  clean_post=terminal.get('clean_post'),actual_GPU_step=terminal.get('step'))
    if not (terminal.get('gpu_passed') and terminal.get('clean_post')):
        return result|dict(state='ORIGINAL_GPU_OR_POST_FAILED',receipt_error_class=terminal.get('error_class'),
          card_errors={str(i): read(ROOT/f'device-{i}'/'result.json',4096)[0].get('error')
                       for i in range(8) if (ROOT/f'device-{i}'/'result.json').is_file()})
    sha=hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()
    assert runtime_sha==sha
    assert len(rows)==3 and all(r[2:4]==['COMPLETED','0:0'] for r in rows)
    gpu,=[r for r in rows if r[1]==LABEL]
    assert gpu[0]==terminal['step'] and 'gres/dcu=8' in gpu[5].split(',')
    assert terminal['before']['pci']==terminal['after']['pci'] and all(r['passed'] for r in terminal['runs'])
    assert terminal['after']['kfd_holder_count']==0 and terminal['before']['boot_id']==terminal['after']['boot_id']
    per_card=[]
    identities=set()
    for card in range(8):
        path=ROOT/f'device-{card}'
        identity,_=read(path/'device.json',4096)
        source_rank=(0,6)[card%2]
        metrics,pin=read(path/f'rank-{source_rank:02d}-result.json',32768)
        assert identity['PCI'] in terminal['before']['pci'] and identity['card']==card and identity['step']==gpu[0]
        assert identity['PCI'] not in identities
        identities.add(identity['PCI'])
        assert metrics['rank']==source_rank and metrics['tensors']==metrics['active_gradients']==504
        assert metrics['elements']==43646976 and metrics['numeric_pass'] and set(metrics['numeric'])=={'fused','fused64'}
        for sample in metrics['numeric'].values():
            assert 0<sample['changed_tensors']<=504
            for key in ('parameter_update','exp_avg','exp_avg_sq'):
                assert 0<=sample[key]['absolute_max']<=5e-5
                assert sample[key]['error_L2']==0 or 0<=sample[key]['relative_L2']<=5e-4
        blocks=metrics['blocks']
        assert [b['mode'] for b in blocks]==['fused','fused64','fused64','fused']
        assert [b['groups'] for b in blocks]==[1,1,1,1]
        assert all(b['timed']==20 and b['warmup']==5 and len(b['event_ms'])==len(b['wall_seconds'])==20 for b in blocks)
        def ratio(field):
            return statistics.mean(blocks[i][field] for i in (1,2))/statistics.mean(blocks[i][field] for i in (0,3))
        per_card.append(dict(card=card,archived_source_rank=source_rank,measurement_SHA256=pin,
            median_pair_event_ratio=ratio('event_median_ms'),median_pair_wall_ratio=ratio('wall_median_seconds'),
            memory_incremental_peaks=[b['incremental_peak_allocated_bytes'] for b in blocks],
            numeric=metrics['numeric']))
    assert identities==set(terminal['before']['pci'])
    return result|dict(state='ACTUAL_NATIVE_GROUPED_OPTIMIZER_ARCHIVE_AND_TIMING_PASS', verified_cards=8,
        optimizer_implementation_SHA256=sha,logical_groups_per_arm=[1,1,1,1],
        median_event_ratio=statistics.median(r['median_pair_event_ratio'] for r in per_card),
        median_wall_ratio=statistics.median(r['median_pair_wall_ratio'] for r in per_card),cards=per_card,
        full_model_qualification=False)


if __name__=='__main__':
    import sys
    print(json.dumps(main(sys.argv[1]),allow_nan=False))
