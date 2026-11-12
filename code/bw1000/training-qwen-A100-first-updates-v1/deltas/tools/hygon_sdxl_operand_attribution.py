"""CPU-only attribution of the two original saved SDXL selected-LoRA operands."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ARCHIVES = {
    'a100': (176936, '5c45a6b3c16fd74ff0092d84556e3529d1a8ad04a689327dcb164ff065062941'),
    'hip': (176936, 'e4876b3c6ab6129900a68460d2aba93e40cb85e5e220ac0b7adf00760fa12c5a'),
}


def check_deadline(epoch, deadline):
    now = time.time()
    if not all(math.isfinite(x) for x in (epoch, deadline)) or not 0 < deadline - epoch <= 220:
        raise ValueError('invalid original CPU epoch')
    if not epoch <= now < deadline:
        raise TimeoutError('original CPU epoch exhausted')


def run(args):
    started = time.monotonic()
    check_deadline(args.epoch, args.deadline)
    if args.output.exists():
        raise FileExistsError('original paired result exists; reconcile without replay')
    if any(os.environ.get(n) not in ('', '-1') for n in
           ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')):
        raise ValueError('explicit empty device visibility is required')
    identities = {}
    for name in ('a100', 'hip'):
        path = getattr(args, name)
        size, expected = ARCHIVES[name]
        if path.is_symlink() or path.stat().st_size != size:
            raise ValueError('original operand archive size or ownership differs')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError('original operand archive identity differs')
        identities[name] = dict(bytes=size, SHA256=digest)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('fresh CPU-only process required')
    torch.set_num_threads(2)

    def forbid(*_a, **_kw):
        raise RuntimeError('GPU context forbidden in saved-operand attribution')

    torch.cuda._lazy_init = forbid
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'generation/src'))
    from gencuda.trellis_operand_attribution import attribute_site
    rows, hashes = {}, {}
    for name in ('a100', 'hip'):
        value = torch.load(getattr(args, name), map_location='cpu', weights_only=True)
        if not isinstance(value, dict) or set(value) != {'input', 'output_gradient', 'weight_gradient'}:
            raise ValueError('original selected capture fields differ')
        for key, shape in [('input', (1, 64, 4)), ('output_gradient', (1, 64, 1280)),
                           ('weight_gradient', (1280, 4))]:
            tensor = value[key]
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                    or tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape
                    or not bool(torch.isfinite(tensor).all())):
                raise ValueError('original selected tensor schema differs')
        rows[name] = dict(input=value['input'], upstream=value['output_gradient'], gradient=value['weight_gradient'])
        hashes[name] = {n: hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                        for n, t in value.items()}
    check_deadline(args.epoch, args.deadline)
    result = attribute_site(rows['a100'], rows['hip'], rank=4, gradient_dtype=torch.bfloat16)
    check_deadline(args.epoch, args.deadline)
    if torch.cuda.is_initialized():
        raise RuntimeError('unexpected GPU initialization')
    output = dict(passed=True, qualification='selected-saved-operand-CPU-algebra-only',
                  archives=identities, operand_SHA256=hashes, rank=4, operand_dtype='torch.bfloat16',
                  gradient_dtype='torch.bfloat16', result=result, epoch=args.epoch,
                  deadline=args.deadline, wall_seconds=time.monotonic()-started,
                  CUDA_initialized=False, GPU_calls=0, model_replayed=False,
                  numerical_repair_applied=False, Model_GO=False)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('a100', 'hip', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('epoch', 'deadline'):
        parser.add_argument('--' + name, type=float, required=True)
    args = parser.parse_args()
    try:
        value = run(args)
    except Exception as error:
        value = dict(passed=False, error_type=type(error).__name__, Model_GO=False)
        if not args.output.exists():
            with args.output.open('x', encoding='utf8') as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
                stream.write('\n')
        print(json.dumps(value))
        return 1
    print(json.dumps({k: value[k] for k in ('passed', 'qualification', 'Model_GO')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
