"""Run a 25-step native branch and a fresh 20-to-25 resume on the same lease."""
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


def compare_witness(continuous, resumed):
    comparison = []
    for rank in range(8):
        sub = f'rank-{rank:02d}'
        a = read(continuous/'train/continuous-tail'/sub/'result.json')
        b = read(resumed/'train/resume-witness'/sub/'result.json')
        evidence = {
            phase: read(path/'train'/f'math-sdpa-rank-{rank:02d}.json')
            for phase, path in (('continuous', continuous), ('resumed', resumed))
        }
        for phase, record in evidence.items():
            if not (record['rank'] == rank and record['phase'] == phase and record['completed_step'] == 25
                    and record['mode'] == 'MATH' and record.get('deterministic') is True
                    and record.get('warn_only') is False and record['text_layers'] == 36 and sum(record['mask_calls'].values()) > 0):
                raise ValueError('Incomplete real Qwen math route witness')
        if not (a['rank'] == b['rank'] == rank and a['world_size'] == b['world_size'] == 8
                and a['completed_global_step'] == b['completed_global_step'] == 25
                and a['continuous_tail_complete'] and b['native_resume_window_complete']):
            raise ValueError('Incomplete continuous or recovery record')
        def series(data, group, key):
            rows = data[group]
            if len(rows) != 20 or len({r['ordinal'] for r in rows}) != 20 or not all(r[key]['finite'] for r in rows):
                raise ValueError('Native recovery microbatch count or tensor invalid')
            return [(r['ordinal'], r.get('global_step'), r[key]['SHA256']) for r in rows]
        inputs_a, inputs_b = (series(data, 'inputs', 'full_input') for data in (a, b))
        loss_a, loss_b = (series(data, 'outputs', 'loss') for data in (a, b))
        if [x[:2] for x in inputs_a] != [x[:2] for x in inputs_b]:
            raise ValueError('Continuous and resumed microbatch order differs')
        comparison.append(dict(rank=rank, compared_microbatches=20,
                               matched_inputs=sum(x == y for x, y in zip(inputs_a, inputs_b)),
                               matched_losses=sum(x == y for x, y in zip(loss_a, loss_b)),
                               route=evidence))
    return comparison


def main():
    intent = read(ROOT/'intent.json')
    clock = EpochClock(intent['epoch'])
    original_preflight = ROOT/'before.json'
    owners = {}
    phases = {}
    result = None
    error = None
    save(ROOT/'native-intent.json', dict(epoch=intent['epoch'], phases=['continuous', 'resumed'],
                                        updates=[25, 5], phase_budget_uses_original_epoch=True))
    try:
        for phase in ('continuous', 'resumed'):
            branch = ROOT/phase
            branch.mkdir(mode=0o700)
            save(branch/'intent.json', intent)
            save(branch/'before.json', read(original_preflight))
            if phase == 'resumed':
                save(branch/'checkpoint20-pins.json', checkpoints(ROOT/'continuous/train/checkpoint-20', 20))
            env = dict(os.environ, QWEN_PAIR_PHASE=phase)
            script = ROOT/'src'/f'native_{phase}.py'
            completed = subprocess.run([sys.executable, '-B', '-u', str(script)],
                                       env=env, timeout=clock.left(NATIVE_SECONDS, 50))
            phase_result = read(branch/'native-outcome.json')
            owners.update(phase_result['owners'])
            phases[phase] = dict(returncode=completed.returncode, normal=phase_result['normal_native_exit'],
                                 artifact=pin(branch/'native-outcome.json'), assigned=pin(branch/'assigned.json'))
            if completed.returncode or not phase_result['normal_native_exit']:
                raise RuntimeError('Original native rank processes did not complete')
        a, b = ROOT/'continuous', ROOT/'resumed'
        keys = compare_witness(a, b)
        cont25, res25 = (checkpoints(branch/'train/checkpoint-25', 25) for branch in (a, b))
        equal = {name: cont25[name] == res25[name] for name in MATCH_TENSORS}
        result = dict(ranks=keys, checkpoint20=read(b/'checkpoint20-pins.json'),
                      continuous25=cont25, resumed25=res25, checkpoint25_equal=equal,
                      all_input_fingerprints_equal=sum(r['matched_inputs'] for r in keys)==160,
                      all_loss_fingerprints_equal=sum(r['matched_losses'] for r in keys)==160)
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc)[:240])
    finally:
        for phase in ('continuous', 'resumed'):
            branch = ROOT/phase
            for name in ('owned-registry.json', 'native-outcome.json'):
                path = branch/name
                if path.is_file():
                    record = read(path)
                    owners.update(record['owners'] if name == 'native-outcome.json' else record)
        save(ROOT/'native-outcome.json', dict(normal_native_exit=error is None, phases=phases,
                                             result=result, error=error, owners=owners))
    return int(error is not None)


if __name__ == '__main__':
    raise SystemExit(main())
