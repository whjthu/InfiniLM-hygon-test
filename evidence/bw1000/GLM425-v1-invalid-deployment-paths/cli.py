"""One ordinary public InferenceServer + explicit qualified GLM provider CLI."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time

from .actors import Actors
from .native_chain_r186 import verify as verify_native_chain
from .protocol import Run
from .worker import configure_process, torch_setup


def verify_package(expected):
    root = Path(__file__).resolve().parents[1]
    manifest = root / 'SOURCE_MANIFEST.json'
    raw = manifest.read_bytes()
    assert len(expected) == 64 and hashlib.sha256(raw).hexdigest() == expected
    q = json.loads(raw)
    for name, digest in q['files_SHA256'].items():
        path = root / name
        assert path.resolve().is_relative_to(root.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    return root


def prepare_public(config):
    # Preserve original visible-GPU0 common extension import before any Native
    # model or processor construction. This is not a hidden-GPU import probe.
    torch_setup(0)
    public = Path(config['public_python_root']).resolve()
    common = Path(config['Common_python_root']).resolve()
    sys.path.insert(0, str(common))
    sys.path.insert(0, str(public))
    import infinilm, _infinilm
    assert Path(infinilm.__file__).resolve() == public / 'infinilm/__init__.py'
    assert Path(_infinilm.__file__).resolve() == Path(config['Common_extension']).resolve()
    assert hashlib.sha256(Path(_infinilm.__file__).read_bytes()).hexdigest() == config['Common_extension_SHA256']


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha', required=True)
    p.add_argument('--source-sha', required=True)
    p.add_argument('--stage', type=int, choices=[0, 1], required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--actual-step', required=True)
    p.add_argument('--gpu-start-epoch', type=float, required=True)
    p.add_argument('--serve', action='store_true')
    a = p.parse_args(argv)
    root = verify_package(a.source_sha)
    raw_config = a.config.read_bytes()
    assert len(a.config_sha) == 64 and hashlib.sha256(raw_config).hexdigest() == a.config_sha
    config = json.loads(raw_config)
    assert config['backend'] == 'qualified-native-glm-nov16'
    assert config['Native_SHA256'] == '9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
    assert config['model_type'] == 'glm_moe_dsa'
    assert (config['TP'], config['PP'], config['rows'], config['context']) == (8, 2, 1, 32768)
    assert (config['Native_seconds'], config['service_seconds']) == (5100, 4500)
    config.update(stage=a.stage, run=a.run, actual_step=a.actual_step,
        gpu_start_epoch=a.gpu_start_epoch, source_manifest_SHA256=a.source_sha,
        provider_config_SHA256=a.config_sha,
        public_python_root=str(root / 'public-python'))
    if not a.serve:
        print(json.dumps(dict(source_and_schema_valid=True, Model_READY=False, GPU=False,
            Native_dlopen=False, ordinary_public_provider='source-only validation; no factory run')))
        return
    assert config['execute_authorized'] is True, 'Source template does not authorize Model execution'
    assert time.time() + 120 < a.gpu_start_epoch + config['service_seconds']
    assert socket.gethostname().split('.')[0] == config['nodes'][a.stage]
    configure_process(config)
    proof = verify_native_chain()
    assert proof['library_SHA256'] == config['Native_SHA256']
    run = Run(config)
    if a.stage == 0:
        group = Actors(config)
        try:
            prepare_public(config)
            group.start()
            group.wait_peer_node()
        except BaseException as error:
            if group._claimed:
                group._trip(error)
                try:group.stop()
                except BaseException:pass
            raise
        return
    # Claim before processor preparation. A duplicate frontend cannot poison a
    # previously running run if its own exclusive claim was refused.
    run.write('FRONTEND.json', dict(PID=os.getpid(), stage=1))
    provider = None
    try:
        prepare_public(config)
        from infinilm.processors import AutoInfinilmProcessor
        from infinilm.llm import FinishReason
        from infinilm.server.inference_server import InferenceServer
        from .prepared_queue import prepare_engine, bind_public_sampling
        from .service_policy import bind_prequeue, ProfileHeaderMiddleware
        from .provider import GLMProvider
        processor = AutoInfinilmProcessor.from_pretrained(config['processor_path'])
        prepared = prepare_engine('GLM', config['model_path'], processor=processor,
            finish_reasons=FinishReason, processor_path=config['processor_path'])
        bind_prequeue(prepared['engine'], config['profile'])
        run.write('PREPARED.json', dict(prepared_identity=prepared['identity']))
        provider = GLMProvider(config, prepared)
        server = InferenceServer(model_path=config['model_path'], device='cuda', dtype='bfloat16',
            tensor_parallel_size=8, pipeline_parallel_size=2, pipeline_parallel_stage=1,
            cache_type='static', max_batch_size=1, max_cache_len=32768, max_tokens=4096,
            temperature=0.0, top_p=1.0, top_k=1, enable_graph=False, skip_load=False,
            enable_prefix_caching=False, host=config['host'], port=config['port'],
            tool_call_parser=prepared['parser'], engine_factory=provider,
            stream_timeout_s=prepared['engine'].validation_timeouts_s['no_actual_progress'],
            request_timeout_s=prepared['engine'].validation_timeouts_s['absolute_request_including_queue'])
        bind_public_sampling(server, prepared['engine'])
        app = server._create_app()
        app.add_middleware(ProfileHeaderMiddleware, default=config['profile'])
        import uvicorn
        http = uvicorn.Server(uvicorn.Config(app, host=config['host'], port=config['port'], log_level='info'))
        done = threading.Event()
        def stop_http_on_failure_or_deadline():
            # Frontend lifecycle only; Native owners never hold/join HTTP.
            while not done.wait(.2):
                if provider.group.failed.is_set() or time.time() >= a.gpu_start_epoch + config['service_seconds']:
                    http.should_exit = True
        watcher = threading.Thread(target=stop_http_on_failure_or_deadline,
                                   name='GLM-public-HTTP-lifecycle', daemon=False)
        watcher.start()
        primary = None
        try:http.run()
        except BaseException as error:
            primary = error
            raise
        finally:
            done.set();watcher.join(timeout=2)
            try:provider.stop()
            except BaseException as cleanup:
                if primary is None:raise
                if hasattr(primary,'add_note'):primary.add_note('Public provider cleanup: '+repr(cleanup))
        assert not provider.group.failed.is_set()
    except BaseException as error:
        try:run.fail(error, stage=1)
        except BaseException as publication_error:
            if hasattr(error,'add_note'):error.add_note('Frontend failure publication: '+repr(publication_error))
        raise


if __name__ == '__main__':main()
