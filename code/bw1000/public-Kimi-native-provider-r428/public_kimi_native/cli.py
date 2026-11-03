"""Explicit four-node qualified Kimi provider, public HTTP only on node0."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import sys
import threading
import time

QUALIFIED = '8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'
PUBLIC425 = '122b69d7c1571f1523e15c672839e2d4479e91863e80018b4a8bb60c2ce93724'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_tree(root, expected):
    root = Path(root).resolve()
    manifest = root / 'SOURCE_MANIFEST.json'
    assert sha(manifest) == expected
    for name, digest in json.loads(manifest.read_bytes())['files_SHA256'].items():
        path = root / name
        assert path.resolve().is_relative_to(root) and sha(path) == digest, name


def configure(config, environ=os.environ, hostname=None):
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', config['run'])
    assert type(config['nonce']) is str and len(config['nonce']) >= 32
    assert config['backend'] == 'qualified-native-kimi8df-tp16pp2-ar'
    assert config['Native_SHA256'] == QUALIFIED
    assert (config['TP'], config['PP'], config['capacity'], config['context']) == (16, 2, 1, 32768)
    assert config['prefill'] == 'W512' and config['draft_cards'] == 0
    assert type(config['native_port']) is int and 1 <= config['native_port'] <= 58535
    assert len(config['nodes']) == len(set(config['nodes'])) == 4
    node = config['node']
    assert 0 <= node < 4 and (hostname or socket.gethostname()).split('.')[0] == config['nodes'][node]
    assert int(environ['SLURM_PROCID']) + int(environ.get('GLOBAL_OFFSET_R2354', '0')) == node
    assert environ['SLURM_JOB_ID'] + '.' + environ['SLURM_STEP_ID'] == config['actual_step']
    for name in ['model_path', 'model_manifest', 'runtime_directory', 'public_provider_root', 'Common_python_root', 'Common_extension']:
        value = config[name]
        assert type(value) is str and '\\' not in value and PurePosixPath(value).is_absolute(), name
    manifest = json.loads(Path(config['model_manifest']).read_bytes())
    assert sha(config['model_manifest']) == config['model_manifest_SHA256']
    assert manifest['model_view']['path'] == config['model_path'] and manifest['nodes'] == config['nodes']
    assert manifest['native_DSO_SHA256'] == QUALIFIED and 'request_prefill_modes_r252' not in manifest
    fixed = dict(manifest['required_environment'], KIMI_ENTRY_R9143='1', KIMI_CAPACITY_ONLY_R9145='0',
        INFINILM_KIMI_BATCH_I192_R9125='1', INFINILM_KIMI_I192_GROUPED_R8669Y='0',
        INFINILM_KIMI_MOE_EXECUTION_R4665='0', KIMI_DRAFT_CARD_COUNT_R9142='0')
    for k, v in fixed.items():
        if k in environ and environ[k] != v: raise ValueError('Conflicting qualified startup policy: ' + k)
    environ.update(fixed)
    assert config['Native_seconds'] - config['service_seconds'] >= 600
    return dict(node=node, local_Native_internal_workers=8, TP_node_rank=node % 2,
        PP_stage=node // 2, sampling_node=2, public_queue_node=0, global_cards=32)


def prepare_public(config):
    # Preserve visible GPU import order; no hidden-GPU Common/SDK import test.
    import torch
    assert torch.cuda.device_count() == 8
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    public_root = Path(config['public_provider_root'])
    verify_tree(public_root, PUBLIC425)
    sys.path.insert(0, str(public_root))
    sys.path.insert(0, str(Path(config['Common_python_root'])))
    sys.path.insert(0, str(public_root / 'public-python'))
    import infinilm, _infinilm
    assert Path(infinilm.__file__).resolve() == (public_root / 'public-python/infinilm/__init__.py').resolve()
    assert Path(_infinilm.__file__).resolve() == Path(config['Common_extension']).resolve()
    assert sha(_infinilm.__file__) == config['Common_extension_SHA256']


def watch_shutdown(provider, http, done):
    c = provider.config
    path = Path(c['runtime_directory']) / ('STOP-' + c['run'] + '.json')
    identity = {k: c[k] for k in ['run', 'source_manifest_SHA256', 'config_SHA256', 'gpu_start_epoch']}
    while not done.wait(.2):
        try:
            stopped = False
            if path.exists():
                q = json.loads(path.read_bytes())
                assert q == dict(identity=identity, action='stop')
                stopped = True
            if stopped or provider.failure is not None or provider.closed or time.time() >= c['gpu_start_epoch'] + c['service_seconds']:
                provider.request_stop()
                http.should_exit = True
                return
        except BaseException as error:
            try: provider._trip(error)
            finally: http.should_exit = True
            return


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha', required=True)
    p.add_argument('--source-sha', required=True)
    p.add_argument('--node', type=int, choices=range(4), required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--nonce', required=True)
    p.add_argument('--actual-step', required=True)
    p.add_argument('--gpu-start-epoch', type=float, required=True)
    p.add_argument('--serve', action='store_true')
    a = p.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    verify_tree(root, a.source_sha)
    assert sha(a.config) == a.config_sha
    config = json.loads(a.config.read_bytes())
    config.update(node=a.node, run=a.run, nonce=a.nonce, actual_step=a.actual_step,
        source_manifest_SHA256=a.source_sha, config_SHA256=a.config_sha,
        gpu_start_epoch=a.gpu_start_epoch)
    topology = configure(config)
    if not a.serve:
        print(json.dumps(dict(source_and_config_valid=True, topology=topology,
            Native_dlopen=False, GPU=False, Model_READY=False)))
        return
    assert config['execute_authorized'] is True  # No device/Model authorization in this source card.
    assert time.time() < a.gpu_start_epoch + config['service_seconds']
    directory = Path(config['runtime_directory']);directory.mkdir(parents=True, exist_ok=True)
    # A duplicate normal-entry call refuses before Common/device preparation;
    # no deletion or overwrite of a previous leader's exclusive intent.
    with (directory / ('claim-' + a.run + '-node%d.json' % a.node)).open('x') as out:
        json.dump(dict(run=a.run, node=a.node, actual_step=a.actual_step,
            source_manifest_SHA256=a.source_sha, config_SHA256=a.config_sha,
            gpu_start_epoch=a.gpu_start_epoch, PID=os.getpid()), out)
    prepare_public(config)
    if a.node != 0:
        from .owner import run_owner
        run_owner(config, None, threading.Event(), lambda: None)
        return
    from infinilm.llm import FinishReason
    from infinilm.server.inference_server import InferenceServer
    from public_glm_native.prepared_queue import bind_public_sampling
    from .prepared import prepare
    from .provider import KimiProvider
    prepared = prepare(config['model_path'], FinishReason)
    provider = KimiProvider(config, prepared)
    server = InferenceServer(model_path=config['model_path'], device='cuda', dtype='bfloat16',
        tensor_parallel_size=16, pipeline_parallel_size=2, pipeline_parallel_stage=0,
        cache_type='static', max_batch_size=1, max_cache_len=32768, max_tokens=4096,
        temperature=0.0, top_p=1.0, top_k=1, enable_graph=False, skip_load=False,
        enable_prefix_caching=False, host=config['host'], port=config['HTTP_port'],
        tool_call_parser=prepared['parser'], engine_factory=provider,
        stream_timeout_s=3600.0, request_timeout_s=10800.0)
    bind_public_sampling(server, prepared['engine'])
    import uvicorn
    http = uvicorn.Server(uvicorn.Config(server._create_app(), host=config['host'], port=config['HTTP_port']))
    done = threading.Event()
    watcher = threading.Thread(target=watch_shutdown, args=(provider, http, done),
        name='Kimi-public-HTTP-lifecycle', daemon=False)
    watcher.start()
    primary = None
    try: http.run()
    except BaseException as error:
        primary = error
        raise
    finally:
        done.set();watcher.join(timeout=2)
        try: provider.stop()
        except BaseException as cleanup:
            if primary is None: raise
            if hasattr(primary, 'add_note'): primary.add_note('Native provider cleanup: ' + repr(cleanup))


if __name__ == '__main__': main()
