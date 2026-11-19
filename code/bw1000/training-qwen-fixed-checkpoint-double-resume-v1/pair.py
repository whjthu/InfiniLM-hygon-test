"""Run two new native process groups from one immutable checkpoint 20."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

from scope import ROOT, NATIVE_SECONDS, EpochClock


def read(path):
    raw = path.read_bytes()
    if not (0 < len(raw) <= 262144):
        raise ValueError(f'Unexpected receipt size: {path.name}')
    return json.loads(raw)


def save(path, data):
    with path.open('x', encoding='utf-8') as out:
        json.dump(data, out, indent=2, allow_nan=False)
        out.write('\n')


def pin(path):
    first = path.lstat()
    if not stat.S_ISREG(first.st_mode) or first.st_size <= 0:
        raise ValueError(f'Unexpected checkpoint file: {path.name}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    dig = digest.hexdigest()
    second = path.stat()
    if (first.st_ino, first.st_size, first.st_mtime_ns) != (second.st_ino, second.st_size, second.st_mtime_ns):
        raise ValueError(f'Checkpoint mutated while pinning: {path.name}')
    return dict(bytes=first.st_size, SHA256=dig)


CHECKPOINT = {'README.md', 'adapter_config.json', 'adapter_model.safetensors', 'additional_config.json',
              'args.json', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'training_args.bin'}
CHECKPOINT |= {f'rng_state_{rank}.pth' for rank in range(8)}
MATCH_TENSORS = ('adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt') + tuple(
    f'rng_state_{rank}.pth' for rank in range(8))


def checkpoints(path, step):
    if set(p.name for p in path.iterdir()) != CHECKPOINT or read(path/'trainer_state.json')['global_step'] != step:
        raise ValueError(f'Native checkpoint-{step} is incomplete')
    return {name: pin(path/name) for name in sorted(CHECKPOINT)}


def rank_series(branch, phase, rank, origin=None):
    record = branch/f'train/{phase}/rank-{rank:02d}/result.json'
    if origin:
        assert pin(record) == origin
    d=read(record)
    assert d['rank']==rank and d['world_size']==8 and d['completed_global_step']==25
    assert d['continuous_tail_complete' if phase=='continuous-tail' else 'native_resume_window_complete']
    series={}
    for group,field in (('inputs','full_input'),('outputs','loss')):
        rows=d[group]
        keys=[(x['ordinal'],x.get('global_step')) for x in rows]
        assert len(keys)==len(set(keys))==20
        assert all(x[field]['finite'] and len(x[field]['SHA256'])==64 for x in rows)
        series[group]=dict(zip(keys,(x[field]['SHA256'] for x in rows)))
    return series


def compare(left,right):
    assert all(list(left[k])==list(right[k]) for k in ('inputs','outputs'))
    input_errors=[k for k in left['inputs'] if left['inputs'][k]!=right['inputs'][k]]
    loss_errors=[k for k in left['outputs'] if left['outputs'][k]!=right['outputs'][k]]
    return dict(compared=20,matching_input=20-len(input_errors),
                matching_loss=20-len(loss_errors),first_input_difference=input_errors[0] if input_errors else None,
                first_loss_difference=loss_errors[0] if loss_errors else None)


def main():
    intent=read(ROOT/'intent.json')
    clock=EpochClock(intent['epoch'])
    anchor=read(ROOT/'src/FIXED_ANCHOR.json')
    origin=Path(anchor['origin'])
    checkpoint=origin/'continuous/train/checkpoint-20'
    phases={}
    owners={}
    result=None
    error=None
    branches=('resume_a','resume_b')
    save(ROOT/'native-intent.json',dict(epoch=intent['epoch'],branches=list(branches),
          updates_per_branch=5,frozen_source_step=anchor['original_GPU_step'],frozen_checkpoint_file_count=len(anchor['files'])))
    try:
        for name in branches:
            clock.left(NATIVE_SECONDS,50)
            branch=ROOT/name
            branch.mkdir(mode=0o700)
            save(branch/'intent.json',intent)
            save(branch/'before.json',read(ROOT/'before.json'))
            save(branch/'checkpoint20-pins.json',anchor['files'])
            env=dict(os.environ,QWEN_PAIR_PHASE='resumed',QWEN_RUN_BRANCH=name)
            p=subprocess.run([sys.executable,'-B','-u',str(ROOT/'src/native_resumed.py')],
                             env=env,timeout=clock.left(NATIVE_SECONDS,50))
            d=read(branch/'native-outcome.json')
            phases[name]=dict(rc=p.returncode,normal=d['normal_native_exit'],
                native=pin(branch/'native-outcome.json'),assigned=pin(branch/'assigned.json'),
                child_owner=read(branch/'child-owner.json'))
            if p.returncode or not d['normal_native_exit']:
                raise RuntimeError('Native fresh recovery did not complete')
            assert read(branch/'resume-config.json')['checkpoint']==str(checkpoint)
        first,second=(phases[k]['child_owner']['identity'] for k in branches)
        assert (first['PID'],first['startticks'])!=(second['PID'],second['startticks'])
        rows=[]
        for rank in range(8):
            witness={}
            for name in branches:
                branch=ROOT/name
                policy=read(branch/'train'/f'math-sdpa-rank-{rank:02d}.json')
                assert policy['rank']==rank and policy['phase']=='resumed' and policy['deterministic'] is True and policy['warn_only'] is False
                assert policy['mode']=='MATH' and policy['completed_step']==25 and policy['text_layers']==36 and sum(policy['mask_calls'].values())>0
                witness[name]=rank_series(branch,'resume-witness',rank)
            original_a=rank_series(origin/'continuous','continuous-tail',rank,anchor['original_window_witnesses']['continuous'][str(rank)])
            original_b=rank_series(origin/'resumed','resume-witness',rank,anchor['original_window_witnesses']['resumed'][str(rank)])
            rows.append(dict(rank=rank,fresh_pair=compare(witness['resume_a'],witness['resume_b']),
                original_pair=compare(original_a,original_b),
                fresh_vs_original_continuous={n:compare(witness[n],original_a) for n in branches},
                fresh_vs_original_resumed={n:compare(witness[n],original_b) for n in branches}))
        check_a,check_b=(checkpoints(ROOT/n/'train/checkpoint-25',25) for n in branches)
        result=dict(ranks=rows,checkpoint20=anchor['files'],checkpoint25_equal={name:check_a[name]==check_b[name] for name in MATCH_TENSORS},
                    fresh_input_matches=sum(r['fresh_pair']['matching_input'] for r in rows),
                    fresh_loss_matches=sum(r['fresh_pair']['matching_loss'] for r in rows),comparison_denominator=160)
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc)[:240])
    finally:
        for name in branches:
            for candidate in ('owned-registry.json','native-outcome.json'):
                path=ROOT/name/candidate
                if path.is_file():
                    d=read(path)
                    owners.update(d['owners'] if candidate=='native-outcome.json' else d)
        save(ROOT/'native-outcome.json',dict(normal_native_exit=error is None,phases=phases,result=result,error=error,owners=owners))
    return int(error is not None)


if __name__=='__main__':
    raise SystemExit(main())
