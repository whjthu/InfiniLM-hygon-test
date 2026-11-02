"""Real prepared processor/engine ingress; optional Native root is forbidden."""
import ast
import asyncio
import copy
import hashlib
import importlib.machinery
import json
import os
import pathlib
import sys
import time
import types

D = pathlib.Path(__file__).resolve().parent
S = D / 'owner'
R = pathlib.Path('/refactor')
sys.path.insert(0, str(S))
from quality_plan import load_data, body, canonical, digest, NATIVE


def main(output):
    output.mkdir(exist_ok=False)
    q = dict(completed=False, passed=False, CPU_only=True, GPU=0, Model=0,
             weights_loaded=False, private_Native_dlopen=False, actual_prepared_ingress=False,
             CPU_request_retirement='finishLENGTH plus full real stream consumption only retires encoder objects; no Model generation or benchmark cap result',
             CASE_REFS_SHA256=digest((S / 'CASE_REFS.json').read_bytes()), cases={})
    start = time.monotonic()
    try:
        assert all(os.environ[k] == '-1' for k in ['CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'])
        from source_pin_r281 import verify
        verify()
        U = R / 'candidates/latest-family-service-r270/Deep/python/infinilm'
        class ForbiddenNative(types.ModuleType):
            def __getattr__(self, name):
                raise RuntimeError('Native API forbidden in CPU encoding: ' + name)
        root = types.ModuleType('infinilm'); root.__path__ = [str(U)]; root.__package__ = 'infinilm'
        sys.modules['infinilm'] = root
        core = ForbiddenNative('infinilm._core'); core.__file__ = 'FORBIDDEN_NATIVE'; core.__path__ = []
        core.__spec__ = importlib.machinery.ModuleSpec(core.__name__, None)
        root._core = core; sys.modules[core.__name__] = core
        llm = types.ModuleType('infinilm.llm'); llm.__path__ = [str(U / 'llm')]; llm.__package__ = 'infinilm.llm'
        sys.modules[llm.__name__] = llm
        for suffix, name in [('scheduler', 'SchedulerOutput'), ('static_scheduler', 'StaticSchedulerOutput')]:
            mod = types.ModuleType('infinilm.llm.' + suffix); setattr(mod, name, type(name, (), {})); sys.modules[mod.__name__] = mod
        class ForbiddenEngine:
            def __init__(self, *a, **k):
                raise RuntimeError('Model construction forbidden in CPU encoding')
        llm.AsyncLLMEngine = ForbiddenEngine
        def audit(event, args):
            if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)) and os.fsdecode(args[0]).lower().endswith(('.safetensors', '.bin', '.pt', '.pth')):
                raise RuntimeError('Weight/model-binary payload read forbidden')
        sys.addaudithook(audit)
        import torch
        torch.set_num_threads(1)
        def forbidden(*a, **k):
            raise RuntimeError('GPU API forbidden during CPU encoding')
        for name in ['_lazy_init', 'init', 'set_device', 'current_device']:
            setattr(torch.cuda, name, forbidden)
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
        from infinilm.llm.sampling_params import SamplingParams
        from infinilm.llm.request import FinishReason
        llm.SamplingParams = SamplingParams; llm.FinishReason = FinishReason
        from prepared_family_entry_r268 import prepare_app
        model = pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8')
        processor_path = R / 'results/latest-three-processors-r260/Deep-model-metadata'
        prepared = prepare_app('Deep', str(model), processor_path=processor_path)
        assert prepared['identity']['upstream_commit'] == '99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'
        assert prepared['identity']['Native_loaded_by_factory'] is prepared['identity']['weights_opened_by_factory'] is False
        from infinilm.server.agent_protocol import prepare_agent_request
        refs, cases = load_data()
        engine, server = prepared['engine'], prepared['server']
        for number, (case_id, case) in enumerate(cases.items()):
            original = body(case)
            data, schemas, choice = prepare_agent_request(copy.deepcopy(original), 'deepseek_v4')
            data.update(_agent_schemas=schemas, _agent_choice=choice)
            params = server._build_sampling_params(data)
            assert params.max_tokens == 4096
            req = engine.add_chat_request(messages=data['messages'], sampling_params=params,
                    request_id='CPU375-' + str(number), request_data=data,
                    add_generation_prompt=True, chat_template_kwargs=data.get('chat_template_kwargs') or {})
            assert req._admission_metadata_ready and req.max_new_tokens == 4096 and engine.take(0) is req
            ids = list(req.prompt_token_ids)
            h = req.metadata['history_render_audit']
            id_sha = digest(json.dumps(ids, separators=(',', ':')).encode())
            assert h['qualified'] and h['prompt_tokens'] == len(ids)
            assert req.metadata['Native_prompt_token_ids_SHA256'] == id_sha
            assert len(ids) + 4096 <= 32768
            private = output / ('private-IDs-%03d.json' % number)
            private.write_text(json.dumps(ids, separators=(',', ':')) + '\n')
            q['cases'][case_id] = dict(prompt_tokens=len(ids), prompt_IDs_SHA256=id_sha,
                    private_IDs_path=str(private), private_IDs_file_SHA256=digest(private.read_bytes()),
                    body_canonical_SHA256=digest(canonical(original)),
                    prompt_render_SHA256=digest(req.prompt.encode()), qualified_history=True)
            # Real Deep lifecycle: finish then consume the public stream's finally
            # to retire _requests. This is an encoder-only synthetic completion.
            req.finish(FinishReason.LENGTH)
            async def retire():
                async for event in engine.stream_request(req, timeout=1, request_timeout=2):
                    assert event.finished
            asyncio.run(retire())
            assert req.request_id not in engine._requests
        engine.stop()
        config_path = model / 'config.json'
        config = json.loads(config_path.read_bytes())
        indices = sorted(model.glob('*.index.json'))
        assert indices, 'Existing checkpoint index is required; no payload rehash claim'
        tokenizer_files = {p.name: digest(p.read_bytes()) for p in processor_path.iterdir()
                           if p.is_file() and (p.name.startswith('tokenizer') or p.name in ['special_tokens_map.json', 'chat_template.jinja'])}
        assert 'tokenizer.json' in tokenizer_files
        template = prepared['tokenizer'].get_chat_template()
        from native_binding_r206b import BASE, binding
        binding()  # Actual49-object source/dependency graph, no CDLL/weights.
        base_raw = BASE.read_bytes()
        assert digest(base_raw) == 'd206637dff3e79d17756962974ed6935e85299d137f602b988d8688d29d999d4'
        tree = ast.parse(base_raw)
        forward = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        selector_source = ast.get_source_segment(base_raw.decode(), forward)
        capi = R / 'candidates/Deep-native-context-guard-r180d/project-mirror/candidates/deep-c4-projection-model-r5920/incoming/modern_native_engine_probe_r22.cpp'
        capi_raw = capi.read_bytes()
        assert digest(capi_raw) == '8c82b8b6d4d4b4a37d795a7d45983331775cea2b8685961c7d16cc600cd5ff23'
        assert 'input.top_k=1;input.top_p=1;input.temperature=1;' in capi_raw.decode()
        worker = R / 'candidates/latest-family-native-r269/Deep/Native-source/engine/rank_worker.cpp'
        worker_raw = worker.read_bytes()
        assert digest(worker_raw) == '3d2f3e679a588d3dd2fee66355f766aeba3c9f4407a25057d1f325125e3e8f0c'
        assert 'top_p == 0.0f || top_k == 1 || temperature == 0.0f' in worker_raw.decode()
        sampling_evidence = {'CAPI_source_SHA256': digest(capi_raw), 'worker_source_SHA256': digest(worker_raw),
                             'Native_top_k': 1, 'Native_top_p': 1, 'Native_temperature': 1,
                             'greedy_basis': 'Actual CAPI top_k1 selects current worker parameter_greedy; HTTPtemperature0 is not the Native scalar'}
        q['selector_source'] = selector_source
        q['comparison_contract'] = dict(
            checkpoint_identity={'path': str(model), 'config_SHA256': digest(config_path.read_bytes()),
                                 'index_SHA256': {p.name: digest(p.read_bytes()) for p in indices},
                                 'hash_scope': 'config and checkpoint index only; not a new full payload hash', 'same_residency_pair': True},
            tokenizer_identity={'processor_path': str(processor_path), 'files_SHA256': tokenizer_files},
            template_identity={'actual_template_SHA256': digest(template.encode()), 'prepared_entry_identity': prepared['identity']},
            precision={'checkpoint_dtype': config.get('torch_dtype', config.get('dtype')),
                       'checkpoint_quantization': config.get('quantization_config'), 'Native_SHA256': NATIVE,
                       'forward_AST_SHA256': digest(ast.dump(forward).encode())},
            temperature=0,
            top_p={'request_field': 'omitted', 'Native_value': 1, 'sampling_evidence': sampling_evidence},
            sampling_seed={'applicable': False, 'basis': 'actual top_k1 parameter_greedy', 'sampling_evidence': sampling_evidence},
            output_budget=4096, context_capacity=32768,
            prompt_policy='Unchanged original custom0shot GSM/MMLU system and full user/choices; requested thinkingFalse; same official prepared ingress')
        q.update(passed=True, prepared_entry_identity=prepared['identity'], actual_prepared_ingress=True,
                 optional_GPU_import_isolated=True, actual_root_or_ABI_qualified=False)
    except BaseException as error:
        import traceback
        q.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        q.update(completed=True, seconds=time.monotonic() - start)
        assert not sys.modules.get('torch') or not sys.modules['torch'].cuda.is_initialized()
        (output / 'encoding.json').write_text(json.dumps(q, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: q.get(k) for k in ['completed', 'passed', 'error', 'seconds']}))
    return 0 if q['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main(pathlib.Path(sys.argv[1])))
