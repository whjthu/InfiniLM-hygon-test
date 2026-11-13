"""Run canonical HunyuanVideo SFT with private first-update capture enabled."""
import argparse
from contextlib import contextmanager
import hashlib
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


class Capture:
    def __init__(self, root, deadline):
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
        self.save('before',params=values,optimizer=optimizer.state_dict(),
                  latents=latents,condition=cond,text=text_emb,text_mask=text_mask,byt5=byt5,byt5_mask=byt5_mask)

    def inputs(self, noise, timestep, hidden_states, target):
        self.save('inputs',noise=noise,timestep=timestep,hidden_states=hidden_states,target=target)

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
    backend.run_hunyuanvideo_training = instrument_backend(original,capture)
    try:
        yield
    finally:
        backend.run_hunyuanvideo_training = original


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-root',type=Path,required=True)
    parser.add_argument('--capture-timeout',type=float,required=True)
    parser.add_argument('--a100-launch',type=Path,required=True)
    options, rest = parser.parse_known_args(argv)
    launch = options.a100_launch.read_bytes()
    if hashlib.sha256(launch).hexdigest()!=LAUNCH_SHA256:
        raise ValueError('capture requires the exact fresh A100 SFT launch')
    if '--run' not in rest or '--dry-run' in rest:
        raise ValueError('capture requires the real canonical run path')
    if not math.isfinite(options.capture_timeout) or not 0<options.capture_timeout<=3600:
        raise ValueError('invalid capture deadline')
    options.capture_root.mkdir(mode=0o700)
    capture=Capture(options.capture_root,time.monotonic()+options.capture_timeout)
    with observe(capture):
        rc=hygon_training_recipe.main(['T09.sft','--a100-launch',str(options.a100_launch),*rest])
    if rc!=0 or len(capture.records)!=6:
        raise RuntimeError('native run or capture incomplete')
    with (options.capture_root/'capture-result.json').open('x',encoding='utf-8') as f:
        json.dump(dict(case='T09.sft',actual_global_step=1,artifacts=capture.records,
                       artifact_bytes=capture.bytes,performance_eligible=False),f,indent=2)
    return rc


if __name__=='__main__':
    raise SystemExit(main())
