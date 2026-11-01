"""Text-only preparation. Preserve the actual r252 request and encode full inputs."""
from pathlib import Path
import copy, hashlib, importlib.util, json, sys


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ids_sha(ids): return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()


def freeze(root, model):
    root, model = Path(root), Path(model)
    source=root/'incoming/Kimi-append512-service-r252'
    expected=json.loads((source/'service-source-manifest-r252.json').read_bytes())
    assert sha(source/'service-source-manifest-r252.json')=='a88218f1954395ec03a15ffa9351da02bf60439253ba42e01bde91180504556a'
    names=['runtime/long_codec_r179.py','runtime/xtml_codec_r143.py','runtime/frozen_encoding_k3_r143.py',
           'runtime/chat_prepare_r223.py','overlay/python/infinilm/server/agent_protocol.py']
    for n in names: assert sha(source/n)==expected['source_SHA256'][n]
    sys.path.insert(0,str(source/'runtime'))
    spec=importlib.util.spec_from_file_location('pinned_agent_protocol',source/names[-1])
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    from long_codec_r179 import KimiLongCodec
    from chat_prepare_r223 import encode
    codec=KimiLongCodec(model)
    p=root/'results/Kimi-append512-fixed-real-request-r252.json'
    assert sha(p)=='e7e95a9cc6e537e582bd575c75d61930f4c9ff31992f3d4dd77ecace8ecda843'
    original=json.loads(p.read_bytes())
    def render(body):
        data,_,_=api.prepare_agent_request(body,'kimi_k3')
        return encode(codec,data['messages'],data,True,data['chat_template_kwargs'])
    text,ids=render(original['body'])
    assert len(ids)==8635 and ids_sha(ids)=='66090df28f4a0b8024a03f769dd9d1f76c91a0348ea7141a31c48ae96f373934'
    def archive_body(count,notes=0):
        body=copy.deepcopy(original['body'])
        index=max(i for i,m in enumerate(body['messages']) if m['role']=='user')
        archive='\n'.join('Archive record %04d: warehouse %d checked valve %d; measured pressure %d kPa; maintenance completed.'%(i,i%17,i%31,90+i%23) for i in range(count))
        body['messages'][index]['content']='Unrelated maintenance archive follows as data. Use the original task after the archive.\n<archive>\n'+archive+'\n'+('Archive note: inspected.\n'*notes)+'</archive>\n\n'+body['messages'][index]['content']
        return body
    lo,hi=0,2048
    while lo<hi:
        mid=(lo+hi+1)//2
        if len(render(archive_body(mid))[1])<=32640: lo=mid
        else: hi=mid-1
    records=lo;lo,hi=0,128
    while lo<hi:
        mid=(lo+hi+1)//2
        if len(render(archive_body(records,mid))[1])<=32640: lo=mid
        else: hi=mid-1
    long_body=archive_body(records,lo);long_text,long_ids=render(long_body)
    assert 32635<=len(long_ids)<=32640 and len(long_ids)+128<=32768
    fixtures=[]
    for name,body,prompt,values,constructed in [('retained8635',original['body'],text,ids,False),('near32k',long_body,long_text,long_ids,True)]:
        fixtures.append(dict(name=name,body=body,prompt_tokens=values,input_tokens=len(values),
            prompt_ID_SHA256=ids_sha(values),rendered_prompt_SHA256=hashlib.sha256(prompt.encode()).hexdigest(),
            expected_calculate_arguments=original['expected_calculate_arguments'],
            constructed_archive=constructed,archive_records=records if constructed else 0,archive_notes=lo if constructed else 0,
            output_budget=128,eos=[163585,163586]))
    assert 'torch' not in sys.modules and 'infinilm' not in sys.modules
    return dict(schema=1,source_request_SHA256=sha(p),tokenizer=codec.identity,fixtures=fixtures,
        order=[['retained8635','AR'],['near32k','AR'],['near32k','spec'],['retained8635','spec']],
        scope='Two fixed engine-level diagnostic inputs, one retained real request and one constructed archive extension; not traffic or benchmark qualification',
        full_prompt_preserved=True,answer_dependent_selection=False,GPU=False,weights_loaded=False,
        source_files_SHA256={n:sha(source/n)for n in names})


if __name__=='__main__':
    print(json.dumps(freeze('/refactor','/shared/Kimi-K3'),ensure_ascii=False))
