"""Read-only check in the original target image on an independently verified zero-GRES step."""
import importlib.metadata as md
import json
import os
from pathlib import Path
import socket
import sys

from bind_ready import bind_value
from blind_inputs import INPUT_MANIFEST_SHA
from effect_contract import DOMAIN, identity, need, pin, read
from infer_blind36 import ENGINE_PINS


def run():
    need(socket.gethostname().split('.')[0] == 'f13r1n12'
         and os.environ.get('SLURM_JOB_ID') == '1103606'
         and os.environ.get('SLURM_STEP_ID', '').isdecimal()
         and not os.environ.get('SLURM_STEP_GPUS')
         and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
         and os.environ.get('HIP_VISIBLE_DEVICES') == ''
         and sys.dont_write_bytecode, 'exact approved zero-GRES target step required')
    value = bind_value()
    need(value['input_manifest']['SHA256'] == INPUT_MANIFEST_SHA, 'wrong input bundle')
    need(value['same_compute_host'] == socket.gethostname() and value['same_compute_boot_id'] ==
         Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'original n12 compute host/boot differs')
    members = value['model']
    need(len(members) == 16, 'original full-model16 manifest differs')
    for path, expected in members.items():
        need(identity(Path(path).lstat()) == expected['file_identity'], 'original full-model16 lifetime differs')
    adapter_count = 0
    for checkpoint in value['checkpoints'].values():
        need(len(checkpoint['files']) == 14, 'original checkpoint manifest differs')
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            expected = checkpoint['files'][name]
            path = Path(checkpoint['root']) / name
            need(identity(path.lstat()) == expected['file_identity'], 'original adapter lifetime differs')
            if name.endswith('.json'):
                read(path, expected)
            adapter_count += 1
    initial = value['initial']
    need(identity(Path(initial['path']).lstat()) == initial['file_identity'], 'original .851 initial lifetime differs')
    framework = json.loads(read(DOMAIN/'fresh-framework-files.json', value['framework']))
    need(len(framework['files']) == 396, 'original framework member count differs')
    for name, expected in framework['files'].items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'bad framework member')
        read(Path(value['framework_root']) / name, expected)
    runtime = json.loads(read(DOMAIN/'runtime-binding.json', value['runtime']))
    need({name: md.version(name) for name in runtime['versions']} == runtime['versions'],
         'original runtime packages differ')
    for name, expected in ENGINE_PINS.items():
        need(pin(read(Path('/swift-runtime/site/swift') / name))['SHA256'] == expected,
             'native Swift engine bytes differ')
    return dict(state='EXACT_ASSETS_OBSERVED', model16_stat=16, adapter_members_stat=adapter_count,
                adapter_config_read=2, initial_stat=1, framework_files_hash=396, engine_source_hash=4,
                input_manifest=value['input_manifest'], planned=108, generation_calls=0, gold_reads=0,
                large_model_or_adapter_payload_hashed=0, actual_native_provider_import=False,
                actual_step=f"1103606.{os.environ['SLURM_STEP_ID']}")


if __name__ == '__main__':
    try:
        print(json.dumps(run(), sort_keys=True))
    except Exception as error:
        print(json.dumps(dict(state='BLOCKED', error_type=type(error).__name__,
                              planned=108, generation_calls=0, gold_reads=0), sort_keys=True))
        raise SystemExit(2)
