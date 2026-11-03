"""Two actual checkpoint CPU encodings through old/current public preparation."""
from pathlib import Path
import ast
import asyncio
import copy
from enum import Enum
import hashlib
import importlib.abc
import importlib.util
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
OUTPUT = Path('/refactor/candidates/public-Kimi-native-provider-r428-CPU-v2')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ids_sha = lambda ids: hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()
text_sha = lambda s: hashlib.sha256(s.encode()).hexdigest()


class NoDevice(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'infinilm', '_infinilm'}:
            raise RuntimeError('Explicit CPU-codec scope forbids import: ' + fullname)


class Reason(str, Enum):
    CANCELED='canceled';ERROR='error';TIMEOUT='timeout';EOS_TOKEN='eos_token';LENGTH='length'


def protocol(name, filename):
    p = ROOT / 'upstream' / filename
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_body(fixture):
    path = Path(fixture['source_path'])
    assert sha(path) == fixture['source_SHA256']
    q = json.loads(path.read_bytes())
    body = q['body'] if fixture['source_kind'] == 'fixed-wrapper' else q
    target = copy.deepcopy(body)
    target['stream'] = fixture['stream']
    if fixture['stream']: target['stream_options'] = dict(include_usage=True)
    assert target['max_tokens'] == fixture['output_budget']
    assert target['messages'] == body['messages'] and target['tools'] == body['tools']
    return body, target


def independent_gold(body, expected):
    dataset = next(json.loads(m['content']) for m in body['messages']
        if m['role'] == 'tool' and isinstance(json.loads(m['content']), dict)
        and 'records' in json.loads(m['content']))
    values = {r['id']: r['value'] for r in dataset['records']}
    selected = [values[key] for key in dataset['query_ids']]
    assert all(type(v) is int for v in selected)
    if expected['function'] == 'calculate':
        assert expected['arguments'] == dict(a=selected[0], b=selected[1])
    else:
        assert expected['function'] == 'write_file' and expected['path'] == 'final.json'
        assert expected['content'] == dict(nonce=dataset['nonce'], total=sum(selected))
    return dict(dataset_nonce=dataset['nonce'], query_values=selected,
        query_ids=dataset['query_ids'], independently_derived=True)


async def retire_cpu_request(engine, req):
    # No Native state exists in this CPU task. This is queue lifetime only.
    req.mark_abort(Reason.CANCELED)
    req.finish(Reason.CANCELED)
    async for _ in engine.stream_request(req, timeout=2, request_timeout=3): pass
    assert req.request_id not in engine._requests


def main():
    assert not any(n in sys.modules for n in ['torch', 'transformers', '_infinilm', 'infinilm'])
    sys.meta_path.insert(0, NoDevice())
    OUTPUT.mkdir(exist_ok=False)
    result = OUTPUT / 'CPU-input-encoding.json'
    q = dict(passed=False, completed=False, CPU_only=True, GPU=False, Slurm_GPU=0,
        Native_dlopen=False, Torch_imported=False, Transformers_imported=False,
        weight_payload_read=False, checkpoint_CPU_tokenizer_actual=True,
        schema_validation_actual=True, original9cases_repeated=False, cases=[])
    def save():
        tmp = result.with_suffix('.tmp')
        tmp.write_text(json.dumps(q, indent=2)+'\n');tmp.replace(result)
    start = time.monotonic();save()
    try:
        manifest = ROOT / 'SOURCE_MANIFEST.json'
        assert sha(manifest) == os.environ['KIMI428_CPU_SOURCE_SHA256']
        m = json.loads(manifest.read_bytes())
        for n, h in m['files_SHA256'].items(): assert sha(ROOT/n) == h, n
        q['source_manifest_SHA256'] = sha(manifest)
        assert sha(ROOT/'upstream/agent_protocol_252.py') == 'c91ab84a5ddc800602dcae0319ad5c02ad3b01c1936a2bd57bce90c2ad15abc3'
        assert sha(ROOT/'upstream/agent_protocol_current.py') == 'c2b71479ec598e1df0d5b152a798c28be5f7905550e10385755f4766479ac326'
        old = protocol('qualified252_protocol_CPU428', 'agent_protocol_252.py')
        current = protocol('current_protocol_CPU428', 'agent_protocol_current.py')
        from public_kimi_native.prepared import prepare
        from public_kimi_native.runtime.long_codec_r179 import KimiLongCodec
        from public_kimi_native.runtime.chat_prepare_r223 import encode as old_encode
        codec = KimiLongCodec('/shared/Kimi-K3')
        q['codec_identity'] = codec.identity
        q['protocol_source_SHA256'] = dict(old=sha(ROOT/'upstream/agent_protocol_252.py'), current=sha(ROOT/'upstream/agent_protocol_current.py'))
        card = json.loads((ROOT/'FIXTURE_CARD.json').read_bytes())
        assert len(card['plan']) == 2
        for fixture in card['plan']:
            assert time.monotonic() - start < 240
            original, body = source_body(fixture)
            before = copy.deepcopy(body)
            row = dict(index=fixture['index'], source_path=fixture['source_path'],
                source_SHA256=fixture['source_SHA256'], stream=fixture['stream'],
                output_budget=fixture['output_budget'], gold=independent_gold(body, fixture['gold']),
                original_messages_SHA256=text_sha(json.dumps(body['messages'], ensure_ascii=False, separators=(',', ':'))))
            q['cases'].append(row);save()
            a, _, _ = old.prepare_agent_request(copy.deepcopy(body), 'kimi_k3')
            old_prompt, old_ids = old_encode(codec, a['messages'], a, True, a['chat_template_kwargs'])
            b, schemas, choice = current.prepare_agent_request(copy.deepcopy(body), 'kimi_k3')
            b.update(_agent_schemas=schemas, _agent_choice=choice)
            engine = prepare('/shared/Kimi-K3', Reason, codec)['engine']
            prompt, ids = engine.encode_chat(b['messages'], b, True, b['chat_template_kwargs'])
            row.update(actual_prompt_tokens=len(ids), actual_prompt_SHA256=text_sha(prompt),
                actual_prompt_IDs_SHA256=ids_sha(ids), old_prompt_tokens=len(old_ids),
                old_prompt_SHA256=text_sha(old_prompt), old_prompt_IDs_SHA256=ids_sha(old_ids),
                old_new_prompt_bytes_equal=old_prompt == prompt, old_new_IDs_equal=old_ids == ids,
                constraint_system_messages=sum(1 for msg in b['messages'] if msg.get('role') == 'system'
                    and (msg.get('content') or '').startswith('[Request tool-call constraints]\n')),
                input_plus_budget=len(ids)+fixture['output_budget'],
                original_body_not_mutated=body == before)
            raw_body = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
            private = OUTPUT / ('case-%d-private.json' % fixture['index'])
            private.write_text(json.dumps(dict(body=body, prompt=prompt, prompt_token_ids=ids), ensure_ascii=False, indent=2)+'\n')
            row.update(actual_request_SHA256=hashlib.sha256(raw_body).hexdigest(), private_path=str(private),
                private_SHA256=sha(private), private_bytes=private.stat().st_size)
            save()  # Save real derived fields before any equivalence/admission assertion.
            assert prompt == old_prompt and ids == old_ids, 'Input-interface contract differs; not a Model quality judgment'
            assert len(ids) == fixture['qualified_prompt_tokens'] and ids_sha(ids) == fixture['qualified_prompt_IDs_SHA256']
            assert row['constraint_system_messages'] == 1 and row['input_plus_budget'] <= 32768
            params = SimpleNamespace(temperature=0.0, max_tokens=fixture['output_budget'], stop=None)
            engine.validate_data(b, params)
            req = engine.add_chat_request(messages=b['messages'], sampling_params=params,
                request_id='CPU-encoding-%d' % fixture['index'], request_data=b,
                chat_template_kwargs=b['chat_template_kwargs'])
            assert engine.take(.1) is req and list(req.prompt_token_ids) == ids
            h = req.metadata['history_render_audit']
            assert h['qualified'] and h['Native_prompt_token_ids_SHA256'] == ids_sha(ids)
            assert h['RC_one_contribution_per_source_field'] and h['RC_order_preserved'] and not h['markers_sent_to_Native']
            row.update(history_render_audit=h, real_queue_same_IDs=True)
            asyncio.run(retire_cpu_request(engine, req));engine.stop()
            row.update(real_queue_retired=True, passed=True);save()
        assert len(q['cases']) == 2 and all(r['passed'] for r in q['cases'])
        assert not any(n in sys.modules for n in ['torch', 'transformers', '_infinilm', 'infinilm'])
        q.update(passed=True, consumer_wall_s=time.monotonic()-start)
    except BaseException as error:
        q.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        q.update(completed=True, wall_s=time.monotonic()-start)
        save()


if __name__ == '__main__': main()
