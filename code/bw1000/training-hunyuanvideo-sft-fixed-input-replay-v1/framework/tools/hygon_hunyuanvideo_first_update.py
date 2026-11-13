"""Run canonical HunyuanVideo SFT with private first-update capture enabled."""
import argparse
from contextlib import contextmanager
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'generation'/'src'))
from gencuda.hunyuanvideo_first_update import instrument_backend
from gencuda.hy_resume_checkpoint import cpu_tree
from gencuda.vggt_first_update import LimitedWriter
import hygon_training_recipe

LAUNCH_SHA256 = '3fdd29e7ea1a09fc54872c044699199cd5ef16bd73af6e7d522fe2ff1b32662f'
REPLAY_PINS = {
    'before': (7149635, '2856fae7918e67fcda1abf4b9ec0106f8fe11900132397ab71c65d9ed5f1ac46'),
    'inputs': (1853149, '5ee197e410ca7a5595b98375be64d392028f466f249a222aea532f2841a1851c'),
}


def read_replay(root):
    import io
    import torch

    result = {}
    for phase, (size, digest) in REPLAY_PINS.items():
        with (root / (phase + '.pt')).open('rb') as f:
            raw = f.read(size + 1)
        if len(raw) != size or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), digest):
            raise ValueError(f'HunyuanVideo replay {phase} identity differs')
        result[phase] = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    return result


def require_tensor_match(reference, actual, label):
    import torch

    if (not isinstance(reference, torch.Tensor) or reference.device.type != 'cpu'
            or reference.dtype != actual.dtype or reference.shape != actual.shape
            or reference.layout != actual.layout):
        raise ValueError(f'HunyuanVideo replay tensor metadata differs for {label}')


class Capture:
    def __init__(self, root, deadline, reference=None):
        self.reference = reference
        self.root = root
        self.deadline = deadline
        self.records = []
        self.bytes = 0

    def save(self, phase, **values):
        import torch
        expected = ('before','inputs','forward','preclip','postclip','updated')
        if phase != expected[len(self.records)] or time.monotonic() >= self.deadline:
            raise RuntimeError('capture phase or deadline differs')
        partial = self.root/(phase+'.partial')
        final = self.root/(phase+'.pt')
        with partial.open('xb') as f:
            writer = LimitedWriter(f,min(128*1024**2,256*1024**2-self.bytes))
            torch.save(cpu_tree(values),writer)
            f.flush()
            os.fsync(f.fileno())
        digest = hashlib.sha256()
        with partial.open('rb') as f:
            for chunk in iter(lambda:f.read(1024**2),b''):
                digest.update(chunk)
        os.link(partial,final)
        partial.unlink()
        self.bytes += writer.count
        self.records.append(dict(phase=phase,file=final.name,bytes=writer.count,SHA256=digest.hexdigest()))
        with (self.root/'artifacts.ndjson').open('a',encoding='utf-8') as f:
            f.write(json.dumps(self.records[-1])+'\n')
            f.flush()
            os.fsync(f.fileno())

    def trainable(self, model, gradients=False):
        named = {n: p.grad if gradients else p for n,p in model.named_parameters() if p.requires_grad}
        if (len(named)!=216 or any(v is None for v in named.values())
                or sum(v.numel() for v in named.values())!=1769472):
            raise ValueError('native LoRA tensor coverage differs')
        return named

    def before(self, model, optimizer, latents, cond, text_emb, text_mask, byt5, byt5_mask):
        values = self.trainable(model)
        owned = [p for group in optimizer.param_groups for p in group['params']]
        if (type(optimizer).__name__!='Muon' or {id(p) for p in owned}!={id(p) for p in values.values()}
                or any(optimizer.state[p]!={'use_muon':True} for p in owned)):
            raise ValueError('native fresh Muon state differs')
        condition = dict(latents=latents, condition=cond, text=text_emb, text_mask=text_mask,
                         byt5=byt5, byt5_mask=byt5_mask)
        if self.reference is not None:
            import torch

            original = self.reference['before']
            if original['params'].keys() != values.keys() or original['optimizer'] != optimizer.state_dict():
                raise ValueError('HunyuanVideo replay initial LoRA or Muon structure differs')
            for name, value in condition.items():
                require_tensor_match(original[name], value, name)
                if not torch.equal(original[name], value.detach().cpu()):
                    raise ValueError(f'HunyuanVideo replay cache values differ for {name}')
            for name, value in values.items():
                require_tensor_match(original['params'][name], value, name)
            with torch.no_grad():
                for name, value in values.items():
                    value.copy_(original['params'][name])
        self.save('before', params=values, optimizer=optimizer.state_dict(), **condition)

    def replay(self, noise, timestep):
        inputs = self.reference['inputs']
        require_tensor_match(inputs['noise'], noise, 'noise')
        require_tensor_match(inputs['timestep'], timestep, 'timestep')
        return inputs['noise'].to(device=noise.device), inputs['timestep'].to(device=timestep.device)

    def inputs(self, noise, timestep, hidden_states, target):
        values = dict(noise=noise, timestep=timestep, hidden_states=hidden_states, target=target)
        if self.reference is not None:
            import torch

            for name, value in values.items():
                expected = self.reference['inputs'][name]
                require_tensor_match(expected, value, name)
                if not torch.equal(expected, value.detach().cpu()):
                    raise ValueError(f'HunyuanVideo replay constructed input differs for {name}')
        self.save('inputs', **values)

    def forward(self, prediction, loss):
        self.save('forward',prediction=prediction,loss=loss)

    def preclip(self, model):
        self.save('preclip',gradients=self.trainable(model,gradients=True))

    def postclip(self, model, norm):
        self.save('postclip',gradients=self.trainable(model,gradients=True),norm=norm)

    def updated(self, model, optimizer):
        self.save('updated',params=self.trainable(model),optimizer=optimizer.state_dict())


@contextmanager
def observe(capture):
    from gencuda.backends import hunyuanvideo_training_walltime as backend
    original = backend.run_hunyuanvideo_training
    backend.run_hunyuanvideo_training = instrument_backend(original, capture, replay=capture.reference is not None)
    try:
        yield
    finally:
        backend.run_hunyuanvideo_training = original


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-root',type=Path,required=True)
    parser.add_argument('--capture-timeout',type=float,required=True)
    parser.add_argument('--a100-launch',type=Path,required=True)
    parser.add_argument('--replay-root', type=Path, help='Local private copy of the verified A100 first-step inputs')
    parser.add_argument('--capture-accelerator',choices=('cuda','hip'),required=True)
    options, rest = parser.parse_known_args(argv)
    launch = options.a100_launch.read_bytes()
    if hashlib.sha256(launch).hexdigest()!=LAUNCH_SHA256:
        raise ValueError('capture requires the exact fresh A100 SFT launch')
    if '--run' not in rest or '--dry-run' in rest:
        raise ValueError('capture requires the real canonical run path')
    if not math.isfinite(options.capture_timeout) or not 0<options.capture_timeout<=3600:
        raise ValueError('invalid capture deadline')
    if options.replay_root is not None and options.capture_accelerator != 'hip':
        raise ValueError('fixed-input replay requires the Hygon diagnostic path')
    deadline = time.monotonic() + options.capture_timeout
    reference = read_replay(options.replay_root) if options.replay_root is not None else None
    options.capture_root.mkdir(mode=0o700)
    capture=Capture(options.capture_root, deadline, reference)
    with observe(capture):
        rc=hygon_training_recipe.main(['T09.sft','--a100-launch',str(options.a100_launch),*rest],
                                      generation_accelerator=options.capture_accelerator)
    if rc!=0 or len(capture.records)!=6:
        raise RuntimeError('native run or capture incomplete')
    with (options.capture_root/'capture-result.json').open('x',encoding='utf-8') as f:
        json.dump(dict(case='T09.sft',actual_global_step=1,artifacts=capture.records,
                       artifact_bytes=capture.bytes,fixed_a100_input_replay=reference is not None,
                       performance_eligible=False),f,indent=2)
    return rc


if __name__=='__main__':
    raise SystemExit(main())
