"""Actual rank owner: static create/load/service/cleanup; no HTTP object."""
import ctypes
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

from .checkpoint_loader import bind_library
from .glm_crossrequest_environment_r9110 import check_hardware
from .native_chain_r186 import verify as verify_native_chain
from .native_owner import run_resident_owner
from .owner_gates import NativeGates
from .protocol import Run
from .resident_context import ResidentContext
from .topology_r404 import read_layout, validated_nodes_r444


def configure_process(config):
    stage = config['stage']
    assert len(os.sched_getaffinity(0)) >= 32, 'original CPU binding before Torch/weights'
    fixed = dict(GLM_EXPECTED_NODES_R444=json.dumps(list(validated_nodes_r444(config['nodes'])),separators=(',',':')), GLM_PP_SIZE_R3603='2', GLM_CACHE_ROWS_R3627='1', GLM_CACHE_CAP_R3627='32768',
        GLM_STAGE_R5040=str(stage), GLM_RUN_R5040=config['run'], GLM_SOURCE_MANIFEST_SHA=config['source_manifest_SHA256'],
        GLM_V_ON_DEMAND_R403='1', GLM_BOUNDED_ROPE_R4279='1', GLM_SHARED_CONTROL_R4279='1',
        GLM_SHARED_PREFIX_VALIDATE_R5814='0', GLM_ACTUAL_CAPTURE_R5164='0', USE_WEIGHT_CACHE_R3604='0',
        GLM_GROUPED_ROUTED_R5123='0', GLM_ASYNC_PROJECTED_HOST_R5011='0', GLM_GROUP_PROJECTED_HOST_R5030='0',
        GLM_BATCH_ALLSELECTED_R5096='1', GLM_DECODE_ROW_DEDUP_R5386='0', GLM_PROJECTED_HOST_R4504='0')
    for key, value in fixed.items():
        if key in os.environ and os.environ[key] != value:
            raise ValueError('Conflicting Native startup policy: ' + key)
    os.environ.update(fixed)
    assert os.environ['INFINILM_OWNER_CPU_WIDTH_R2392'] == '4'
    assert os.environ['SLURM_PROCID'] == str(stage)
    assert os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'] == config['actual_step']
    assert not any(os.environ.get(k) for k in ['BASH_ENV', 'PYTHONSTARTUP', 'PYTHONINSPECT'])


def torch_setup(rank):
    import torch
    assert torch.cuda.device_count() == 8
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    for key in ['allow_tf32', 'allow_fp16_reduced_precision_reduction',
                'allow_bf16_reduced_precision_reduction', 'allow_fp16_accumulation']:
        setattr(torch.backends.cuda.matmul, key, False)
        assert getattr(torch.backends.cuda.matmul, key) is False
    assert torch.get_float32_matmul_precision() == 'highest'
    assert torch.cuda.mem_get_info(rank)[0] > 58 * 1024 ** 3
    return torch


def owner_main(rank, config, prepared=None, stop_intent=None):
    """Only rank0/emitter receives a same-process prepared queue object."""
    from .cli import verify_package
    verify_package(config['source_manifest_SHA256'])
    run = Run(config)
    stage = config['stage']
    emitter = stage == 1 and rank == 0
    assert emitter == (prepared is not None)
    configure_process(config)
    report_path = run.path / ('owner-%d-%d.json' % (stage, rank))
    try:
        proof = verify_native_chain()
        assert proof['library_SHA256'] == config['Native_SHA256']
        # Original prepared/common import occurs on visible GPU0 before the
        # owning rank is selected; preserve that order for extension static FA.
        torch = torch_setup(0)
        import infinilm, _infinilm
        assert Path(infinilm.__file__).resolve() == Path(config['public_python_root']).resolve() / 'infinilm/__init__.py'
        assert Path(_infinilm.__file__).resolve() == Path(config['Common_extension']).resolve()
        assert hashlib.sha256(Path(_infinilm.__file__).read_bytes()).hexdigest() == config['Common_extension_SHA256']
        torch.cuda.set_device(rank)
        assert torch.cuda.mem_get_info(rank)[0] > 58 * 1024 ** 3
        hardware = check_hardware(torch, rank)
        binding = bind_library(config['Native_library'], config['Native_SHA256'])
        binding.check(binding.lib.modern_glm_dense_mode_r48_set(2))
        stream = lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        payload = run.wait('COMMUNICATORS.json')['payload']
        expected_id_bytes = binding.lib.modern_glm_model_id_size()
        assert 0 < expected_id_bytes <= 65536 and len(payload['tp']) == 2
        for encoded in [payload['world'], *payload['tp'], payload['control']]:
            assert len(base64.b64decode(encoded, validate=True)) == expected_id_bytes
        report = dict(library_sha256=config['Native_SHA256'], native_chain=proof,
            hardware_gate_before_create=hardware, actual_owner_thread=threading.get_ident(),
            Torch_matmul_supported_modes={'float32_precision':torch.get_float32_matmul_precision(),
                **{key:getattr(torch.backends.cuda.matmul,key) for key in ['allow_tf32',
                    'allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction',
                    'allow_fp16_accumulation']}},
            Common_extension=dict(path=str(Path(_infinilm.__file__).resolve()),
                                  SHA256=config['Common_extension_SHA256']),
            original_gpu_start_epoch=config['gpu_start_epoch'], original_Native_seconds=config['Native_seconds'],
            TP=8, PP=2, all_serving_cards=16, cache_capacity=32768, cache_rows=1,
            Model_READY=False, completed=False)
        def save():
            temp = report_path.with_suffix('.tmp')
            temp.write_text(json.dumps(report, indent=2));temp.replace(report_path)
        def fail(error):
            run.fail(error, stage=stage, rank=rank)
        def phase(name, owner_rank):
            with (run.path / ('phase-%d-%d.jsonl' % (stage, rank))).open('a') as out:
                out.write(json.dumps(dict(phase=name, rank=owner_rank, stage=stage,
                    monotonic=time.monotonic(), realtime=time.time())) + '\n');out.flush()
        gates = NativeGates(torch=torch, rank=rank, stage=stage, stream=stream,
                            config=config, publish_failure=fail)
        from safetensors import safe_open
        layout = read_layout(os.environ, Path(config['model_path']) / 'config.json', config['nodes'][stage])
        assert layout['config_SHA256'] == config['config_SHA256']
        context = ResidentContext(binding, model_path=config['model_path'], layout=layout,
            rank=rank, stage=stage, comm_ids=payload, environ=os.environ,
            hostname=config['nodes'][stage], torch=torch, safe_open=safe_open,
            deadline_check=run.deadline_check, phase_marker=phase,
            lifecycle_path=run.path / ('lifecycle-%d-%d.json' % (stage, rank)),
            constructor_record_path=run.path / ('constructor-%d-%d.json' % (stage, rank)),
            report=report, owner_gates=gates)
        # Exact preparation identity is shared as JSON, never NativeRequest,
        # processor, locks or queue through multiprocessing.
        preparation = run.wait('PREPARED.json')
        identity = preparation['prepared_identity']
        from infinilm.llm import FinishReason
        def ready(current):
            assert current['all16_loaded_and_postload_4GiB_passed']
            current['service_agreement'] = True
            save()
            run.write('ready-%d-%d.json' % (stage, rank), dict(stage=stage, rank=rank,
                Native_SHA256=config['Native_SHA256'], config_SHA256=config['config_SHA256'],
                loaded=True, postload_4GiB=True, service_agreement=True,
                affinity=current['owner_affinity_after_load']))
        run_resident_owner(context, torch=torch, stream=stream, environ=os.environ,
            stage=stage, rank=rank, vocab=config['vocab_size'],
            report_path=report_path, queue_path=run.path / 'queue.json',
            engine=prepared['engine'] if emitter else None, FinishReason=FinishReason,
            prepared_identity=identity, eos_token_ids=identity['eos_token_ids'],
            run_identity=dict(run=config['run'], actual_step=config['actual_step'],
                              parent_source_manifest_SHA256=config['source_manifest_SHA256']),
            default_profile=config['profile'], profile_manifest_SHA256=config['source_manifest_SHA256'],
            deadline=time.monotonic() + max(0, config['gpu_start_epoch'] + config['service_seconds'] - time.time()),
            should_stop=lambda: bool(stop_intent is not None and stop_intent.is_set()) or run.stop_requested(),
            owner_ready=ready, owner_failure=fail)
        report.update(completed=True, worker_shutdown_completed=True, Model_READY=False)
        save()
        run.write('closed-%d-%d.json' % (stage, rank), dict(stage=stage, rank=rank,
            normal=report['normal_shutdown'], post_close_Torch_liveness=report['post_close_Torch_liveness']))
    except BaseException as error:
        # Publish before parent fail-stop/peer termination. Never retry a ctx.
        run.fail(error, stage=stage, rank=rank)
        error_path = run.path / ('error-%d-%d.json' % (stage, rank))
        try:
            error_path.write_text(json.dumps(dict(error=repr(error), traceback=traceback.format_exc(),
                stage=stage, rank=rank, PID=os.getpid()), indent=2))
        except BaseException: pass
        raise


def process_owner(rank, config):
    # Spawn arguments contain JSON/base64 configuration only. No queue object.
    try: owner_main(rank, config)
    except BaseException:
        traceback.print_exc()
        os._exit(2)
