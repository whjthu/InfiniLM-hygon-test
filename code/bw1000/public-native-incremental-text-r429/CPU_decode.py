"""Actual pinned GLM tokenizer on128 original output-ID sequences, CPU only."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,time,traceback

D=Path(__file__).resolve().parent
R=Path('/refactor')
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    out=R/'candidates/public-native-incremental-text-r429-CPU-v1'
    out.mkdir(exist_ok=False);report=out/'result.json';started=time.monotonic()
    q=dict(completed=False,passed=False,GPU=False,Native_dlopen=False,Model_READY=False,
        actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        weight_payload_read=False,actual_HTTP_SSE=False,cases=[])
    def save():
        temp=report.with_suffix('.tmp');temp.write_text(json.dumps(q,indent=2)+'\n');temp.replace(report)
    save()
    try:
        assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
        raw=(D/'SOURCE_MANIFEST.json').read_bytes();assert sha(raw)==os.environ['R429_CPU_SOURCE_SHA256']
        manifest=json.loads(raw)
        for name,h in manifest['files_SHA256'].items():assert sha((D/name).read_bytes())==h,name
        plan=json.loads((D/'PLAN.json').read_bytes())
        private=R/'results/GLM429-original128-visible-IDs-private.json'
        b=private.read_bytes();assert sha(b)==plan['private_fixture_SHA256']
        original=json.loads(b);assert len(original['sequences'])==128
        model=Path('/old/models/ZhipuAI-GLM-5.3')
        for name,h in plan['tokenizer_assets_SHA256'].items():assert sha((model/name).read_bytes())==h,name
        prior=R/'candidates/GLM-NoV-W4-quality-encoding-r418-CPU-v2/encoding.json'
        assert sha(prior.read_bytes())=='2051a3a74e5ce471b53c38ddffd0f744e1a0669bdc2312590f6bd6d2a5013c12'
        def audit(event,args):
            if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
                path=os.fsdecode(args[0]).lower()
                if path.endswith(('.safetensors','.bin','.pt','.pth')):
                    raise RuntimeError('Model weight/binary payload forbidden')
        sys.addaudithook(audit)
        from transformers import AutoTokenizer
        tick=time.monotonic()
        tokenizer=AutoTokenizer.from_pretrained(str(model),trust_remote_code=True,local_files_only=True)
        q['tokenizer_load_s']=time.monotonic()-tick
        spec=importlib.util.spec_from_file_location('r429_actual_text_prefix',D/'incremental_text.py')
        candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
        def decode(ids):return tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        q['tokenizer_class']=type(tokenizer).__module__+'.'+type(tokenizer).__qualname__
        q['tokenizer_assets_SHA256']=plan['tokenizer_assets_SHA256']
        sequences=list(original['sequences'])
        for i,text in enumerate(['中文空格 and_under_score 🙂🚀', 'a  b\n\t中\ufffd文', '<think>思考</think>\n#### 42']):
            sequences.append(dict(id='CPU_unicode_'+str(i),visible_ids=tokenizer.encode(text,add_special_tokens=False),synthetic_text=True))
        prefixes=0;delayed=0
        for seq in sequences:
            ids=seq['visible_ids'];assert ids and all(type(v)is int and v>=0 for v in ids)
            full=decode(ids);prefix=candidate.TextPrefix(decode);chunks=[];first=None
            for count in range(1,len(ids)+1):
                delta=prefix.advance(ids[:count],final=False);prefixes+=1
                if delta:
                    chunks.append(delta)
                    if first is None:first=count
                else:delayed+=1
            chunks.append(prefix.advance(ids,final=True))
            merged=''.join(chunks);assert merged==full,seq['id']
            q['cases'].append(dict(id=seq['id'],visible_tokens=len(ids),first_text_at_visible_count=first,
                final_text_SHA256=sha(full.encode()),joined_deltas_SHA256=sha(merged.encode()),
                deltas=sum(bool(v)for v in chunks),original_model_sequence=not seq.get('synthetic_text',False)))
        maps=Path('/proc/self/maps').read_text()
        assert not any(x in maps for x in ['libGLM_','libDeep_','libKimi_','_infinilm.cpython','libinfinirt.so'])
        q.update(passed=True,source_manifest_SHA256=sha(raw),private_fixture_SHA256=sha(b),
            original_model_sequences=128,synthetic_unicode_sequences=3,prefix_decodes=prefixes,
            no_text_delta_prefixes=delayed,actual_checkpoint_tokenizer=True,final_concatenation_equals_P0=True,
            Native_request_loop_executed=False,Native_reset_or_devices_qualified=False,
            framework_Torch_imported='torch'in sys.modules,
            scope='Actual checkpoint tokenizer prefix/Unicode preservation only. Original8local realNativeRequest/bridge tests use a byte tokenizer; this job does not turn their Native recording into a real Model/HTTP consumer.',
            timing='CPU tokenizer work only; no GPU/Model latency or clientSSE speedup claim.')
    except BaseException as error:
        q.update(error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        q.update(completed=True,wall_s=time.monotonic()-started);save()

if __name__=='__main__':main()
