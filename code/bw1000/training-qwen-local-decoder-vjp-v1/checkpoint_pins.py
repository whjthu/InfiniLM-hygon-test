"""Pin the original checkpoint payloads under Python 3.10 and 3.12."""
import hashlib
import json
import stat

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
