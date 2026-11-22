"""An opt-in Swift callback for the native T01 v_proj layout observation."""

from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import torch
from peft.tuners.lora.layer import Linear as PeftLinear
from swift.callbacks import TrainerCallback, callbacks_map


KEY = 'qwen_native_vproj_upstream_layout'
HERE = Path(__file__).resolve().parent
SOURCE_NAME = 'upstream_layout.py'
SOURCE_BYTES = 11373
SOURCE_SHA = '1e5d47104725c3107e91a9ed95de10a9f89391616847dcef362bb0dbd3fbc3f4'
MF_SHA = '0fc3eb3f03dd0022d615350a9ae9fcb1bb5396e249cc3f7654dc33af316253d6'
REPORT = 'vproj-upstream-layout-rank-01.json'


spec = importlib.util.spec_from_file_location('qwen_t01_layout_observer', HERE / 'observer.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def read_exact(path, size, digest):
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError('expected a canonical exact source member')
    with path.open('rb') as stream:
        raw = stream.read(size+1)
    if len(raw) != size or sha256(raw).hexdigest() != digest:
        raise ValueError('frozen upstream-layout dependency differs')
    return raw


def load_layout(directory):
    folder = directory.resolve(strict=True)
    if folder != directory:
        raise ValueError('expected original canonical upstream-layout directory')
    manifest = json.loads(read_exact(folder / 'SOURCE_FILES.json', 1590, MF_SHA))
    if manifest['files'][SOURCE_NAME] != dict(bytes=SOURCE_BYTES, SHA256=SOURCE_SHA):
        raise ValueError('frozen upstream-layout manifest member differs')
    path = folder / SOURCE_NAME
    raw = read_exact(path, SOURCE_BYTES, SOURCE_SHA)
    module = ModuleType('qwen_t01_original_upstream_layout')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def find_edge(model):
    ddp = torch.nn.parallel.DistributedDataParallel
    root = model.module if isinstance(model, ddp) else model
    language = root.get_base_model().get_submodule('model.language_model')
    if len(language.layers) != 36:
        raise ValueError('unexpected native Qwen language layer count')
    projection = language.layers[6].get_submodule('self_attn.v_proj')
    if (type(projection) is not PeftLinear or list(projection.active_adapters) != ['default']
            or projection.disable_adapters or projection.merged or projection.use_dora.get('default', False)):
        raise ValueError('native v_proj PEFT wrapper unsupported')
    base = projection.get_base_layer()
    if (type(base) is not torch.nn.Linear or base.bias is not None or base.weight.requires_grad
            or tuple(base.weight.shape) != (1024, 4096) or base.weight.dtype != torch.bfloat16):
        raise ValueError('native frozen v_proj base unsupported')
    return root, base


def publish(output, result):
    target = output / REPORT
    temporary = target.with_suffix('.tmp')
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + '\n').encode()
    stream = temporary.open('xb')
    try:
        with stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink()


class QwenLayoutCallback(TrainerCallback):
    observation = None
    unavailable = None

    def on_train_begin(self, args, state, control, **kwargs):
        if os.environ.get('RANK') != '1':
            return control
        output = Path(args.output_dir)
        try:
            if (os.environ.get('LOCAL_RANK') != '1' or os.environ.get('WORLD_SIZE') != '8'
                    or not torch.version.hip or torch.cuda.current_device() != 1 or not args.bf16
                    or args.per_device_train_batch_size != 1 or args.gradient_accumulation_steps != 4):
                raise ValueError('unsupported target recipe or physical rank')
            deadline = core.validate_deadline(float(os.environ['QWEN_VPROJ_LAYOUT_DEADLINE']))
            layout = load_layout(Path(os.environ['QWEN_VPROJ_LAYOUT_SOURCE']))
            root, base = find_edge(kwargs.get('model', self.trainer.model))
            self.observation = core.Observer(torch, layout, self.trainer, root, base, deadline,
                                             lambda result: publish(output, result))
            self.observation.install()
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            self.observation = None
            self.unavailable = dict(rank=1, site=core.SITE, state='OBSERVER_SETUP_UNSUPPORTED', coverage=[],
                                    diagnostics_invoked=0, failure=core.error_info(error))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.observation is not None:
            self.observation.finish('TRAINING_ENDED_BEFORE_ELIGIBLE_EDGE')
        elif self.unavailable is not None:
            try:
                publish(Path(args.output_dir), self.unavailable)
            except (OSError, ValueError):
                pass
        return control


if KEY in callbacks_map:
    raise ValueError('Qwen v_proj layout callback already registered')
callbacks_map[KEY] = QwenLayoutCallback
