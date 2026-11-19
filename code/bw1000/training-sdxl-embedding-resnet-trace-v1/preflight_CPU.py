"""CPU preflight for the forward-only observer and actual model metadata."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

from micro_observer import EmbeddingResnetProbe, FIRST, TARGETS


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.act=nn.SiLU()

    def forward(self,t):
        return self.act(self.act(t))


def main(config_path, result):
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
    example=Toy()
    tensor=torch.tensor([1.,2.],dtype=torch.bfloat16,requires_grad=True)
    expected=example(tensor)
    expected_grad,=torch.autograd.grad(expected.sum(),tensor)
    with EmbeddingResnetProbe(Toy,{'act':2}) as probe:
        output=example(tensor)
        output.sum().backward()
        trace=probe.report()
        assert torch.equal(output,expected) and torch.equal(tensor.grad,expected_grad)
        assert trace['events']==['act#1.in','act#1.out','act#2.in','act#2.out']
        assert trace['edges']['act#1.out']['sha256']==trace['edges']['act#2.in']['sha256']
    assert not probe.data and not probe.handles
    from diffusers import UNet2DConditionModel
    config=Path(config_path).read_bytes()
    assert hashlib.sha256(config).hexdigest()=='30ebc70750223e59006f7f2b4e1e6c102570aa19a9c4ae3e1fbe7591332dbae6'
    with torch.device('meta'):
        unet=UNet2DConditionModel.from_config(json.loads(config))
    assert all(x.is_meta for x in unet.parameters())
    targets={p:type(unet.get_submodule(p)).__name__ for p in TARGETS}
    unit=unet.get_submodule(FIRST)
    assert unit.time_embedding_norm=='default' and not unit.skip_time_act
    assert unit.output_scale_factor==1.0 and unit.conv_shortcut is None
    with Path(result).open('x',encoding='utf-8') as f:
        json.dump(dict(passed=True,model_nodes=targets,tests='ordered duplicate SiLU calls, same CPU output/gradient, real meta modules',
                      GPU_initialized=False,raw_weight_reads=0),f,indent=2)


if __name__=='__main__':
    main(*sys.argv[1:])
