"""New pure-stdlib boundaries only; never import Torch or installed providers."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
from types import SimpleNamespace

D = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def main():
    recipe = module('new_recipe', D / 'deltas/tools/hygon_vlm_recipe.py')
    contract = module('new_contract', D / 'deltas/vlm/runtime/cuda_resume_contract.py')
    valid = []; rejected = []
    def reject(name, call):
        try:
            call()
        except (ValueError, TypeError):
            rejected.append(name); return
        raise AssertionError('bad boundary accepted: ' + name)
    for case in ['T03.sft', 'T04.sft']:
        raw = (D / 'cases' / (case + '-original-launch.json')).read_bytes()
        original = json.loads(raw)
        kwargs = dict(case=case, original=raw, swift=PurePosixPath('/env/bin/swift'),
                      models=PurePosixPath('/models'), dataset=PurePosixPath('/data/coco64.jsonl'),
                      plugins=PurePosixPath('/plugins'), output=PurePosixPath('/new/train'), master_port=29911,
                      diagnostic_plugin=PurePosixPath('/framework/vlm/plugins/cuda_resume_training.py'),
                      diagnostic_callback='native_cuda_resume_witness',
                      resume_from_checkpoint=PurePosixPath('/original/train/checkpoint-20'))
        launch = recipe.build_launch(**kwargs); argv = launch['command']
        assert launch['resume_device'] == 'cuda' and 'nvidia_baseline' not in argv
        assert not any(Path(x).name == 'swift_callback_nvidia_baseline_v2.py' for x in argv)
        for flag in ['--max_steps', '--max_length', '--per_device_train_batch_size', '--gradient_accumulation_steps',
                     '--gradient_checkpointing', '--lora_rank', '--lora_alpha', '--lora_dropout', '--attn_impl']:
            assert argv[argv.index(flag) + 1] == original['command'][original['command'].index(flag) + 1]
        assert argv[argv.index('--max_steps') + 1] == '25'
        if case == 'T04.sft':
            assert '/framework/vlm/plugins/llava_onevision.py' in argv
        valid.append(case + ':single CUDA callback/recipe preserved')
        reject(case + ':no checkpoint', lambda: recipe.build_launch(**dict(kwargs, resume_from_checkpoint=None)))
        reject(case + ':old HIP callback', lambda: recipe.build_launch(**dict(kwargs, diagnostic_callback='native_resume_witness')))
        reject(case + ':foreign plugin', lambda: recipe.build_launch(**dict(kwargs, diagnostic_plugin=PurePosixPath('/wrong/cuda_resume_training.py'))))
        reject(case + ':double observer', lambda: recipe.build_launch(**dict(kwargs,
            diagnostic_plugin=[kwargs['diagnostic_plugin'], PurePosixPath('/extra.py')],
            diagnostic_callback=['native_cuda_resume_witness', 'old_formal25'])))
    config = json.loads((D / 'cases/T03.sft-config-template.json').read_bytes())
    config.update(root_GO=True, GPU_GO=True, epoch=time.time() - 1, deadline=time.time() + 100,
                  original_root=str(D / 'synthetic-reference'), checkpoint=str(D / 'synthetic-reference/train/checkpoint-20'),
                  comparison_checkpoint=str(D / 'synthetic-reference/train/checkpoint-25'), output=str(D / 'synthetic-output'))
    contract.check_config(config); valid.append('synthetic bounded config')
    for name, delta in [('GO false', {'GPU_GO': False}), ('wrong device', {'device': 'hip'}),
                        ('expired epoch', {'epoch': time.time() - 200, 'deadline': time.time() - 1}),
                        ('missing rank RNG', {'checkpoint_files': {k: v for k, v in config['checkpoint_files'].items() if k != 'rng_state_7.pth'}})]:
        reject(name, lambda delta=delta: contract.check_config(dict(config, **delta)))
    tree = ast.parse((D / 'deltas/vlm/runtime/cuda_resume_witness.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CudaResumeWitness')
    write = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'write')
    namespace = dict(json=json, os=os, require=contract.require)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[write], type_ignores=[])), '<actual-write-method>', 'exec'), namespace)
    with tempfile.TemporaryDirectory(prefix='cuda-resume-JSON-') as temporary:
        target = Path(temporary)
        namespace['write'](SimpleNamespace(root=target), 'roundtrip.json', {'planned': 160, 'actual': None})
        raw = (target / 'roundtrip.json').read_bytes()
        assert raw[-1:] == b'\n' and json.loads(raw) == {'planned': 160, 'actual': None}
        valid.append('actual write method JSON roundtrip/newline0A')
    # Execute only the real getter selector against stdlib stand-ins, not CUDA.
    current_rng = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'current_rng')
    class FakeTensor:
        dtype = 'uint8'; ndim = 1
    cuda = SimpleNamespace(get_rng_state=lambda: 'current', get_rng_state_all=lambda: 'all')
    namespace = dict(torch=SimpleNamespace(Tensor=FakeTensor, uint8='uint8', cuda=cuda),
                     require=contract.require, rng_values=lambda getter: getter())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[current_rng], type_ignores=[])), '<actual-RNG-selector>', 'exec'), namespace)
    call = lambda value: namespace['current_rng'](SimpleNamespace(expected_rng20={'cuda': value}))
    assert call(FakeTensor()) == 'current' and call([FakeTensor() for _ in range(8)]) == 'all'
    valid += ['saved Tensor selects current-device getter', 'saved list8 selects all-device getter']
    reject('saved CUDA list7', lambda: call([FakeTensor() for _ in range(7)]))
    reject('saved mixed CUDA list', lambda: call([FakeTensor()] * 7 + [None]))
    for path in D.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    return dict(passed=True, valid=len(valid), rejected=len(rejected), valid_cases=valid, rejected_cases=rejected,
                actual_Torch_import=False, actual_provider_import=False, actual_GPU=False,
                actual_checkpoint_read=False, actual_model_run=False,
                witness_source_SHA256=hashlib.sha256((D / 'deltas/vlm/runtime/cuda_resume_witness.py').read_bytes()).hexdigest())


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
