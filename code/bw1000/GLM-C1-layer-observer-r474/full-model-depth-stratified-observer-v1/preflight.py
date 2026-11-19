"""Actual future two-node container admission: source/input/library hashes, no Native load."""
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, '/paired-worker')
import shared
import native_arm
from budget import PAIR_SECONDS, remaining
from depth_binding import BINDING, INDEXER_TYPES, MLP_TYPES


def main(out):
    started = time.monotonic()
    epoch = float(os.environ['GLM470_EPOCH'])
    shared.require(remaining(epoch, PAIR_SECONDS) > 0, 'original preflight epoch expired')
    shared.require(all(os.environ.get(key) == '-1' for key in
                       ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')),
                   'CPU preflight device mask differs')
    node = socket.gethostname().split('.')[0]
    shared.require(node == shared.NODES[int(os.environ['SLURM_PROCID'])], 'preflight node identity differs')
    shared.require([socket.gethostbyname(node) for node in shared.NODES] == shared.IPS, 'frozen peer addresses differ')
    for name in ('SOURCE_MANIFEST.json', 'HOST_MANIFEST.json'):
        manifest = json.loads((shared.D/name).read_bytes())
        shared.require(all(shared.H(shared.D/path) == digest for path, digest in manifest['files_SHA256'].items()),
                       'installed exact source bytes differ')
    inp = json.loads((shared.D/'INPUT_BINDING.json').read_bytes())
    private = inp['private_plan']
    private_path = shared.R/Path(private['path']).relative_to(shared.HOST)
    shared.require(private_path.stat().st_size == private['bytes'], 'original plan bytes differ')
    plan = shared.load_pinned(private_path, private['SHA256'])
    shared.verify_inputs(inp, json.loads((shared.D/'TRIAL.json').read_bytes()), plan)
    model = Path(inp['model_path'])
    shared.require(shared.H(model/'config.json') == inp['model_config_SHA256']
                   and shared.H(model/'model.safetensors.index.json') == inp['model_index_SHA256']
                   and shared.H(inp['tokenizer']['path']) == inp['tokenizer']['SHA256'], 'frozen model metadata differs')
    config = json.loads((model/'config.json').read_bytes())
    shared.require(BINDING['original_config_SHA256'] == inp['model_config_SHA256']
                   and tuple(config['indexer_types']) == INDEXER_TYPES
                   and tuple(config['mlp_layer_types']) == MLP_TYPES, 'depth schedule differs from deployed config')
    library, digest = native_arm.verified_library('socket')
    providers = json.loads((shared.D/'PROVIDER_BINDING.json').read_bytes())['providers']
    shared.require(len(providers) == 71, 'provider denominator differs')
    # Original179 records all71 provider hashes; refresh only the same8 project
    # providers used by actual Model runtime admission, never rehash giant SDKs.
    project=json.loads((shared.D/'CPU_BINDING.json').read_bytes())['actual_project_DSO_SHA256']
    shared.require(all(providers.get(p)==h for p,h in project.items()),'CPU308 reference project-provider record mismatch')
    for path,expected in project.items():
        shared.require(shared.H(path)==expected,'actual selected project provider bytes differ')
    budget = json.loads((shared.D/'EVENT_BUDGET.json').read_bytes())
    shared.require(budget['old_caller_upper_bound'] == 278 and budget['new_marks_upper_bound'] == 162
                   and budget['depth_markers_upper_bound'] == 14
                   and budget['prior_caller_upper_bound']+14 == budget['future_caller_upper_bound'] == 454
                   and 454 < budget['pool_events_unchanged'] == 512,
                   'attention event contract differs')
    result = dict(passed=True, node=node, actual_GPU=shared.step_id(), original_epoch=epoch,
                  Native_SHA256=digest, Native_path=str(library), Native_loaded=False,
                  actual_GPU_cards_used=0, provider_records=71, actual_project_provider_hashes=len(project), input_binding_SHA256=shared.H(shared.D/'INPUT_BINDING.json'),
                  source_manifest_SHA256=shared.H(shared.D/'SOURCE_MANIFEST.json'), events_upper_bound=454,
                  wall_seconds=time.monotonic()-started, weight_tensor_payload_bytes_read=0,
                  original_private_plan_retained_remote=True)
    shared.save(Path(out)/'pre'/(node+'-container.json'), result)


if __name__ == '__main__':
    main(sys.argv[1])
