"""Actual K3/public prepared FIFO encoding of fixed444 questions; no Native.

Only transport-independent Reasons are a recording leaf, as in CPU876. The
sampling dataclass, public protocol, codec, history and FIFO are real sources.
"""
from pathlib import Path
from enum import Enum
import asyncio,copy,hashlib,importlib,importlib.util,json,os,sys,time,traceback,types,unicodedata

D=Path(__file__).resolve().parent
OUT=Path('/refactor/candidates/Kimi-wide-fewshot-quality-r448-CPU-v1')
sha=lambda b:hashlib.sha256(b).hexdigest()
compact=lambda q:json.dumps(q,ensure_ascii=False,separators=(',',':')).encode()

class Reasons(str,Enum):
    EOS_TOKEN='eos_token';LENGTH='length';TIMEOUT='timeout';CANCELED='canceled';ERROR='error';STOP_STRING='stop_string'

def verify_tree(root,expected):
    root=Path(root);raw=(root/'SOURCE_MANIFEST.json').read_bytes();assert sha(raw)==expected
    for n,h in json.loads(raw)['files_SHA256'].items():
        p=root/n;assert p.resolve().is_relative_to(root.resolve()) and sha(p.read_bytes())==h,n

def question_key(v):
    return sha(' '.join(unicodedata.normalize('NFC',v).split()).encode())

def questions(q):
    out=[]
    if isinstance(q,dict):
        if isinstance(q.get('question'),str):out.append(q['question'])
        for k,v in q.items():
            if k not in ['source','metadata','sources']:out.extend(questions(v))
    elif isinstance(q,list):
        for v in q:out.extend(questions(v))
    return out

def file_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def observation():
    maps=Path('/proc/self/maps').read_text()
    found=[line for line in maps.splitlines()if any(v in line for v in ['_infinilm.cpython','libKimi_','libGLM_','libDeep_','libinfinilm_runtime','libinfiniops','libinfinirt','libinfiniccl'])]
    torch=sys.modules.get('torch');cuda=getattr(torch,'cuda',None)if torch else None
    return dict(project_or_SDK_maps=found,Torch_imported=torch is not None,Torch_CUDA_initialized=getattr(cuda,'_initialized',False)if cuda else False)

def main():
    assert not OUT.exists(),'Immutable CPU result exists; do not replay'
    OUT.mkdir();report=OUT/'encoding.json';start=time.monotonic()
    q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=0,Model_READY=False,Native_dlopen=False,weights_loaded=False,HTTP=0,cases=[],planned_questions=256,planned_HTTP=256,single_baseline=True,no_reselection=True,actual_Kimi_codec=False,actual_public_protocol=False,actual_prepared_FIFO=False,recording_leaves=['FinishReason enum; no owned Model emit/reset/finish'],actual_service_or_Native_executed=False)
    def save():
        tmp=report.with_suffix('.tmp');tmp.write_bytes((json.dumps(q,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode());tmp.replace(report)
    save()
    try:
        assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
        raw=(D/'SOURCE_MANIFEST.json').read_bytes();assert sha(raw)==os.environ['KIMI448_SOURCE_SHA256']
        m=json.loads(raw)
        for n,h in m['files_SHA256'].items():assert sha((D/n).read_bytes())==h,n
        pins=json.loads((D/'PINS.json').read_bytes());plan=json.loads((D/'PLAN.json').read_bytes())
        q.update(source_manifest_SHA256=sha(raw),Native_SHA256=pins['Native_SHA256'],plan_SHA256=sha((D/'PLAN.json').read_bytes()))
        assert plan['questions']==plan['HTTP']==256 and plan['arms']==1 and not plan['GPU_GO']
        assert (plan['future_budget']['Native_s'],plan['future_budget']['service_s'],plan['future_budget']['GPU_s'],plan['future_budget']['keeper_s'])==(28800,28200,29400,32400)
        for key in ['provider','runtime','public']:
            verify_tree(pins[key]['path'],pins[key]['SHA256'])
        q['initial_modules']=observation();assert not q['initial_modules']['project_or_SDK_maps']
        def no_weights(event,args):
            if event=='open'and isinstance(args[0],(str,bytes,os.PathLike))and os.fsdecode(args[0]).lower().endswith(('.safetensors','.bin','.pth','.pt')):raise RuntimeError('CPU encoder forbids weight payload')
        sys.addaudithook(no_weights)
        src={}
        for name,pin in pins['inputs'].items():
            b=Path(pin['path']).read_bytes();assert sha(b)==pin['SHA256'],name;src[name]=json.loads(b)
        cases=src['cases']['cases'];old_requests=src['requests']['requests']
        assert len(cases)==len(old_requests)==256 and [c['id']for c in cases]==[r['case_id']for r in old_requests]
        old_enc={r['case_id']:r for r in src['GLM_encoding']['cases']}
        assert {c['suite']for c in cases}=={'gsm8k','mmlu'} and sum(c['suite']=='gsm8k'for c in cases)==128
        overlaps=[]
        for pin in pins['known_prior_question_files']:
            b=Path(pin['path']).read_bytes();assert sha(b)==pin['SHA256'];known={question_key(s)for s in questions(json.loads(b))}
            matched=[c['id']for c in cases if question_key(c['question'])in known]
            overlaps.append(dict(path=pin['path'],SHA256=pin['SHA256'],known_unique_questions=len(known),matching_case_ids=matched,overlap=len(matched),scope=pin['scope']))
        q.update(known_prior_overlap=overlaps,overlap_scope='Exact normalized known source inventory only, not universal unseen status; fixedcases never replaced based on overlap or answers')
        prefix='kimi448_actual_public';package=types.ModuleType(prefix);package.__path__=[str(Path(pins['public']['path'])/'public-python/infinilm/server')];sys.modules[prefix]=package
        protocol=importlib.import_module(prefix+'.agent_protocol')
        assert sha(Path(protocol.__file__).read_bytes())==pins['actual_protocol_SHA256']
        sampling=file_module('kimi448_actual_sampling',Path(pins['public']['path'])/'public-python/infinilm/llm/sampling_params.py')
        assert sha(Path(sampling.__file__).read_bytes())==pins['actual_sampling_SHA256']
        from public_kimi_native.runtime.long_codec_r179 import KimiLongCodec
        from public_kimi_text.prepared import prepare
        codec=KimiLongCodec('/shared/Kimi-K3');prepared=prepare('/shared/Kimi-K3',Reasons,codec=codec);engine=prepared['engine'];engine.start()
        assert Path(sys.modules['public_kimi_text.prepared'].__file__).resolve()==(Path(pins['provider']['path'])/'public_kimi_text/prepared.py').resolve()
        assert Path(sys.modules['public_kimi_native.public_encode'].__file__).resolve()==(Path(pins['runtime']['path'])/'public_kimi_native/public_encode.py').resolve()
        q.update(actual_Kimi_codec=True,actual_public_protocol=True,actual_prepared_FIFO=True,codec_identity=codec.identity,EOS_ids=sorted(codec.eos_token_ids),actual_loaded_source_modules={n:dict(path=str(v.__file__),SHA256=sha(Path(v.__file__).read_bytes()))for n,v in list(sys.modules.items())if n.startswith(('public_kimi_text','public_kimi_native'))and getattr(v,'__file__',None)})
        bodies=[]
        for ordinal,(case,old)in enumerate(zip(cases,old_requests)):
            assert old['case_id']==case['id'] and sha(compact(old['body']))==old['compact_body_SHA256']==old_enc[case['id']]['request_body_SHA256']
            body=copy.deepcopy(old['body']);body.update(model='Kimi-K3',chat_template_kwargs=dict(enable_thinking=False))
            assert body['messages']==old['body']['messages'] and case['question']in body['messages'][1]['content']
            assert body['max_tokens']==4096 and body['temperature']==0 and body['stream']is False and body['tool_choice']=='none'
            original=copy.deepcopy(body);data,schemas,choice=protocol.prepare_agent_request(copy.deepcopy(body),'kimi_k3');data.update(_agent_schemas=schemas,_agent_choice=choice)
            assert choice=='none' and data['chat_template_kwargs'].get('enable_thinking',data['chat_template_kwargs'].get('thinking'))is False
            params=sampling.SamplingParams(temperature=0.,top_p=1.,top_k=1,max_tokens=4096)
            engine.validate_data(data,params)
            req=engine.add_chat_request(messages=data['messages'],sampling_params=params,request_id='Kimi448-CPU-'+str(ordinal),request_data=data,add_generation_prompt=True,chat_template_kwargs=data['chat_template_kwargs'])
            assert engine.take(timeout=1)is req
            ids=list(req.prompt_token_ids);h=req.metadata['history_render_audit'];id_sha=sha(compact(ids))
            assert h['qualified'] and h['prequeue_bound'] and not h['markers_sent_to_Native'] and h['generation_thinking']is False
            assert h['Native_prompt_token_ids_SHA256']==id_sha and req.prompt.endswith('<|open|>response<|sep|>')
            assert body['messages'][1]['content']in req.prompt and req.max_new_tokens==4096 and len(ids)+4096<=32768
            reverse,reverse_ids=engine.encode_chat(data['messages'],data,True,data['chat_template_kwargs'])
            assert reverse==req.prompt and list(reverse_ids)==ids and body==original
            dest=OUT/('case-%03d-private.json'%ordinal)
            dest.write_bytes((json.dumps(dict(case_id=case['id'],body=body,prompt=req.prompt,prompt_token_ids=ids),ensure_ascii=False)+'\n').encode())
            # CPU administrative retirement only. No token/Native/reset is simulated.
            req.finish(Reasons.CANCELED)
            async def retire():
                async for token in engine.stream_request(req,timeout=1,request_timeout=2):pass
            asyncio.run(retire());assert req.request_id not in engine._requests and not req.generated_token_ids
            row=dict(case_id=case['id'],suite=case['suite'],source_index=case['source_index'],shot_count=case['shot_count'],selected_ordinal=ordinal,source_case_compact_SHA256=sha(compact(case)),original_GLM_body_SHA256=old['compact_body_SHA256'],request_body_SHA256=sha(compact(body)),prompt_SHA256=sha(req.prompt.encode()),prompt_IDs_SHA256=id_sha,prompt_tokens=len(ids),output_budget=4096,input_plus_output=len(ids)+4096,prefill_chunk_tokens=[min(512,len(ids)-i)for i in range(0,len(ids),512)],actual_thinking=False,history_render_audit=h,CPU_queue_retired=True,Native_reset_or_generation_executed=False,private_path=str(dest),private_SHA256=sha(dest.read_bytes()))
            q['cases'].append(row);bodies.append(dict(case_id=case['id'],body=body,request_body_SHA256=row['request_body_SHA256']))
            save()
        assert len(q['cases'])==256 and not engine._requests;engine.stop()
        requests=OUT/'requests-private.json';requests.write_bytes((json.dumps(dict(requests=bodies,HTTP=256,arms=1),ensure_ascii=False,indent=2)+'\n').encode())
        q.update(passed=True,matching_full_fewshot_bodies=256,actual_encodings=256,all_Kimi_input_plus4096_within32768=True,prompt_tokens_min=min(c['prompt_tokens']for c in q['cases']),prompt_tokens_max=max(c['prompt_tokens']for c in q['cases']),requests_file=str(requests),requests_file_SHA256=sha(requests.read_bytes()),quality_result_created=False)
    except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        q.update(completed=True,elapsed_s=time.monotonic()-start,final_modules=observation())
        if q['final_modules']['project_or_SDK_maps']or q['final_modules']['Torch_CUDA_initialized']:q.update(passed=False,unexpected_device_library=True)
        save();print(json.dumps(dict(path=str(report),SHA256=sha(report.read_bytes()),passed=q['passed'],actual_CPU=q['actual_CPU'],actual_encodings=len(q['cases']),elapsed_s=q['elapsed_s'],error=q.get('error'))),flush=True)
    return 0 if q['passed']else 1

if __name__=='__main__':raise SystemExit(main())
