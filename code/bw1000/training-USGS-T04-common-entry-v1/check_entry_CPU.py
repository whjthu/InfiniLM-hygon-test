"""Exercise only the new opt-in dispatcher and pure launch translation."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

D = Path(__file__).resolve().parent
G = D.parents[2]
TOOLS = D / 'deltas/tools'
OLD = G / 'code/bw1000/training-vlm-cuda-resume20-v1'


def main():
    sys.path.insert(0, str(OLD / 'deltas/tools'))
    spec = importlib.util.spec_from_file_location('usgs_entry_under_test', TOOLS / 'usgs_domain_launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = (OLD / 'cases/T04.sft-original-launch.json').read_bytes()
    config = dict(case='T04.sft', mode='fresh25', arm='domain', device='cuda', formal17=False,
                  output='/private/domain/witness', training800={'path': '/private/domain/train-800.jsonl'})
    def plan(c):
        return module.make_launch(c, original, swift='/native/bin/swift', models='/native/models',
            plugins='/native/plugins', framework='/native/framework', master_port=29917)
    fresh = plan(config)
    raw = json.loads(original)['command']
    def value(command, flag):
        return command[command.index(flag) + 1]
    for flag in ['--max_steps', '--torch_dtype', '--lora_rank', '--lora_alpha', '--lora_dropout',
                 '--per_device_train_batch_size', '--gradient_accumulation_steps', '--gradient_checkpointing',
                 '--learning_rate', '--warmup_ratio', '--attn_impl', '--optim', '--dataset_shuffle',
                 '--dataloader_drop_last', '--packing', '--padding_free', '--save_steps']:
        assert value(fresh['command'], flag) == value(raw, flag)
    assert value(fresh['command'], '--callbacks') == 'native_usgs_domain_witness'
    assert 'nvidia_baseline' not in fresh['command']
    assert not any(Path(p).name == 'swift_callback_nvidia_baseline_v2.py' for p in fresh['required_assets']['plugins'])
    assert value(fresh['command'], '--dataset') == config['training800']['path']
    assert fresh['environment_overrides']['NPROC_PER_NODE'] == '8'
    assert value(fresh['command'], '--model') == '/native/models/LLaVA-OneVision-1.5-8B-Instruct'
    generic = plan(dict(config, arm='generic', output='/private/generic/witness'))
    resume = plan(dict(config, mode='resume20to25', checkpoint='/private/fresh/train/checkpoint-20'))
    assert value(resume['command'], '--resume_only_model') == 'false'
    assert value(resume['command'], '--ignore_data_skip') == 'false'
    assert value(resume['command'], '--load_args') == 'false'
    assert generic['arm'] == 'generic'
    rejected = []
    cases = [dict(config, case='T03.sft'), dict(config, mode='resume'), dict(config, arm='other'),
             dict(config, device='hip'), dict(config, formal17=True), dict(config, output='/private/domain/other'),
             dict(config, output='relative/witness'), dict(config, mode='resume20to25', checkpoint='/private/domain/train'),
             dict(config, mode='resume20to25', checkpoint='/private/domain/train/nested')]
    for index, bad in enumerate(cases):
        try:
            plan(bad)
        except (ValueError, argparse.ArgumentTypeError):
            rejected.append(index)
        else:
            raise AssertionError('Bad launch admitted: ' + str(index))
    try:
        module.make_launch(config, original + b' ', swift='/native/bin/swift', models='/native/models',
            plugins='/native/plugins', framework='/native/framework', master_port=29917)
    except ValueError:
        rejected.append('changed-original')
    else:
        raise AssertionError('Changed frozen launch admitted')
    entry = types.ModuleType('common_entry_under_test')
    entry.__dict__['__file__'] = str(TOOLS / 'hygon_training_recipe.py')
    exec(compile((TOOLS / 'hygon_training_recipe.py').read_bytes(), entry.__file__, 'exec'), entry.__dict__)
    old_main = entry.vlm.main
    old_domain = sys.modules.get('usgs_domain_launch')
    called = []
    domain = types.ModuleType('usgs_domain_launch')
    domain.main = lambda argv: called.append(('domain', argv)) or 19
    entry.vlm.main = lambda argv: called.append(('original', argv)) or 17
    sys.modules['usgs_domain_launch'] = domain
    try:
        for case in ['T01.sft', 'T02.sft', 'T03.sft', 'T04.sft', 'T05.sft']:
            assert entry.main([case]) == 17 and called[-1] == ('original', [case])
        args = ['T04.sft', '--usgs-domain-config', '/private/config.json']
        assert entry.main(args) == 19 and called[-1] == ('domain', args)
    finally:
        entry.vlm.main = old_main
        if old_domain is None:
            del sys.modules['usgs_domain_launch']
        else:
            sys.modules['usgs_domain_launch'] = old_domain
    for path in [*TOOLS.glob('*.py'), Path(__file__)]:
        ast.parse(path.read_bytes(), filename=str(path))
    # A new scope above900s must call the explicit new deadline function, never old load_config.
    devices = [dict(index=i, UUID='GPU-test-' + str(i), PCI='0000:%02x:00.0' % (i + 1)) for i in range(8)]
    clock = dict(expected_GPU_inventory=devices, remaining_seconds=1500)
    queries = []
    def query(argv, **kw):
        queries.append(kw['timeout'])
        rows = '' if 'compute-apps' in argv[1] else '\n'.join(
            '%d,%s,%s,20000,40000' % (d['index'], d['UUID'], d['PCI']) for d in devices)
        return types.SimpleNamespace(returncode=0, stdout=rows)
    class PortProbe:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def bind(self, address): assert address == ('127.0.0.1', 29917)
    with patch.object(module.subprocess, 'run', query), patch.object(module.socket, 'socket', PortProbe):
        actual = module.fresh_gpu_admission(clock, 29917, lambda c: c['remaining_seconds'])
    assert len(actual['devices']) == 8 and actual['active_compute_processes'] == 0 and queries == [10, 10]
    return dict(passed=True, valid_plans=3, rejected_plans=len(rejected), dispatcher_routes=6,
                new_deadline_binding_above900_CPU_fixture=True,
                native_model_calls=0, Torch_import=False, GPU_calls=0, formal17_recipe_changes=False,
                actual_framework_installed=False)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
