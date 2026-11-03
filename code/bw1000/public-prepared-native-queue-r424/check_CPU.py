"""Real static factory/queue/history/sampling consumers; processor is recorded.

No installed infinilm/FastAPI/Native imports, model payload, GPU or jobs. The
public sampling method is consumed verbatim from the pinned real public file.
"""
import argparse
import ast
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from prepared_queue import prepare_engine, bind_public_sampling


def run(template_path, public_server_path):
    template = Path(template_path).read_text(encoding='utf-8')
    assert hashlib.sha256(template.encode()).hexdigest() == '3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c'
    source = Path(public_server_path).read_bytes()
    assert hashlib.sha256(source).hexdigest() == 'ef831cc23a8a71ed8d89824187b2494e41c16ec23aee1bd2f8a7794d68fa96de'
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'InferenceServer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_build_sampling_params')
    ns = dict(SamplingParams=SimpleNamespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<actual-public-sampling-only>', 'exec'), ns)
    class Server:
        _build_sampling_params = ns['_build_sampling_params']
        max_tokens, temperature, top_p, top_k, ignore_eos = 96, 0.0, 1.0, 1, False

    class Processor:
        """Recorded tokenizer/render leaves, exact real template identity only."""
        def __init__(self, family): self.family = family
        def get_tokenizer(self): return self
        def get_chat_template(self): return template
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
            assert tokenize is False
            opening = '<|assistant|><think>' if self.family == 'GLM' else '<｜Assistant｜><think>'
            parts = []
            for m in messages:
                content = m.get('content') or ''
                if self.family == 'GLM' and m['role'] == 'assistant': content = content.strip()
                rc = m.get('reasoning_content') or m.get('reasoning')
                parts.append(opening + rc + '</think>' if rc else '<' + m['role'] + '>')
                parts.append(content)
                parts.append(json.dumps(m.get('tool_calls', []), ensure_ascii=False))
                parts.append('</' + m['role'] + '>')
            if add_generation_prompt: parts.append(opening)
            return ''.join(parts)
        def __call__(self, prompt, return_tensors):
            assert return_tensors is None
            return {'input_ids': [[ord(c) for c in prompt]]}
        def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids)

    reasons = SimpleNamespace(TIMEOUT='timeout', ERROR='error', CANCELED='canceled')
    cases = []
    async def retire(engine, request):
        rows = [q async for q in engine.stream_request(request, timeout=1, request_timeout=2)]
        assert len(rows) == 1 and rows[0].finished
        assert request.request_id not in engine._requests
    def rejected(fn):
        try: fn()
        except (ValueError, RuntimeError, TypeError): return
        raise AssertionError('Invalid input accepted')

    with tempfile.TemporaryDirectory(prefix='prepared-queue-') as tmp:
        model = Path(tmp)
        (model/'generation_config.json').write_text(json.dumps({'eos_token_id': [154820]}))
        for family in ['GLM', 'Deep']:
            prepared = prepare_engine(family, model, processor=Processor(family), finish_reasons=reasons)
            engine = prepared['engine']
            assert not {'server', 'app'}.intersection(prepared)
            assert prepared['identity']['Model_READY'] is False
            assert prepared['identity']['HTTP_created_by_factory'] is False
            server = Server()
            bind_public_sampling(server, engine)
            messages = [{'role': 'user', 'content': 'Use complete prior history.'},
                        {'role': 'assistant', 'content': ' done ', 'reasoning_content': 'Historical reasoning.'},
                        {'role': 'user', 'content': 'Continue with the same full input.'}]
            data = dict(messages=messages, _agent_schemas={}, _agent_choice='none',
                        max_completion_tokens=128, temperature=0.0)
            original = copy.deepcopy(data)
            params = server._build_sampling_params(data)
            assert params.max_tokens == 128 and data == original
            req = engine.add_chat_request(messages=messages, sampling_params=params,
                request_id=family+'-recorded', request_data=data)
            assert engine.take(0) is req and req._admission_metadata_ready
            assert req.metadata['history_render_audit']['qualified']
            assert req.metadata['history_render_audit']['input_RC_fields'] == 1
            assert req.max_new_tokens == 128 and req.prompt_token_ids == tuple(ord(c) for c in req.prompt)
            if family == 'GLM': engine.record_started(req)
            req.emit_token(ord('A')); req.emit_token(154820, visible=False)
            if family == 'GLM': engine.record_finished(req, 2, True)
            req.finish('eos_token')
            asyncio.run(retire(engine, req))
            assert data == original
            cases.append(dict(case=family+'_actual_factory_public_sampling_full_history_queue_retire', passed=True))

            for invalid in [dict(data, max_tokens=129), dict(data, max_completion_tokens=True),
                            dict(data, temperature=0.7), dict(data, max_completion_tokens=0)]:
                rejected(lambda: server._build_sampling_params(invalid))
                assert not engine._requests
            rejected(lambda: bind_public_sampling(server, engine))
            assert server._build_sampling_params({'messages': messages, '_agent_schemas': {}, '_agent_choice': 'none'}).max_tokens == 4096
            engine.stop(); assert engine.take(0) is None
            rejected(lambda: engine.start())
            cases.append(dict(case=family+'_prequeue_invalid_budget_sampling_and_no_private_HTTP_lifecycle', passed=True))

        class WrongTemplate(Processor):
            def get_chat_template(self): return 'wrong template'
        rejected(lambda: prepare_engine('GLM', model, processor=WrongTemplate('GLM'), finish_reasons=reasons))
        (model/'generation_config.json').write_text(json.dumps({'eos_token_id': True}))
        rejected(lambda: prepare_engine('Deep', model, processor=Processor('Deep'), finish_reasons=reasons))
        cases.append(dict(case='original_template_and_actual_generation_metadata_rejection', passed=True))

    return dict(passed=True, cases=cases, real_factory_sampling_history_and_queue_consumed=True,
        public_sampling_source_SHA256=hashlib.sha256(source).hexdigest(),
        exact_actual_template_SHA256=hashlib.sha256(template.encode()).hexdigest(),
        recording_leaves=['official processor construction/render/token IDs', 'SamplingParams storage'],
        public_HTTP_executed=False, Native_constructed=False, actual_checkpoint_encoding=False,
        Native_dlopen=False, GPU=False, Model_READY=False, jobs=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--public-server', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    q = run(args.template, args.public_server)
    out = Path(args.output)
    with out.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(q, f, indent=2); f.write('\n')
    print(json.dumps(q))
