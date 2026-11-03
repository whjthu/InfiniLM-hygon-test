"""Actual public prepared encoder, no server, queue, prompt crop or new tokenizer."""
import copy
import hashlib
import json
from pathlib import Path

BODY_SHA='f902fed8c1d4db6cd6e8afa1b91b729f5d9182e93f781b4f64952bc56d0fa077'
IDS_SHA='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'
TEACHER_FILE_SHA='ce899e73d47ec43740de22e50edcff19d4e88b0678f94af45fd16c505bf5f4c0'
TEACHER_IDS_SHA='596df3e1131e55184edc034f57f8447e025b91a38c7d6a20a69eaa8b61f39513'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
ids_sha=lambda ids:sha(json.dumps(list(ids),separators=(',',':')).encode())


def read_assets(body_path,teacher_path,reference):
    body_raw=Path(body_path).read_bytes();teacher_raw=Path(teacher_path).read_bytes()
    assert sha(body_raw)==BODY_SHA and sha(teacher_raw)==TEACHER_FILE_SHA
    body=json.loads(body_raw);record=json.loads(teacher_raw);teacher=record['teacher_token_ids']
    assert record['prompt_IDs_SHA256']==IDS_SHA
    assert len(teacher)==83 and all(type(v)is int and 0<=v<129280 and v!=1 for v in teacher)
    assert ids_sha(teacher)==TEACHER_IDS_SHA
    assert body['max_tokens']==128 and body['stream']is True and body['tool_choice']=='none'
    assert body['chat_template_kwargs']=={'enable_thinking':False}and body['text_delivery']=='P1'
    assert reference['prompt_count']==8489 and reference['prompt_IDs_SHA256']==IDS_SHA
    assert reference['teacher_count']==83 and reference['teacher_IDs_SHA256']==TEACHER_IDS_SHA
    assert reference['modes']==[0,1,0,1]and reference['max_forwards']==368 and reference['raw_total_limit']==128*1024**2
    return body,teacher,record


def encode(body,processor):
    from infinilm.server.agent_protocol import prepare_agent_request
    from public_deep_model.prepared_chat import make_prepared_encode_chat
    before=copy.deepcopy(body)
    prepared,schemas,choice=prepare_agent_request(body,'deepseek_v4')
    prepared['_agent_schemas']=schemas;prepared['_agent_choice']=choice
    prompt,ids=make_prepared_encode_chat(processor)(prepared['messages'],prepared,
        bool(prepared.get('add_generation_prompt',True)),prepared.get('chat_template_kwargs')or{})
    assert body==before and len(ids)==8489 and ids_sha(ids)==IDS_SHA
    return ids,dict(body_SHA256=BODY_SHA,prompt_count=len(ids),prompt_IDs_SHA256=ids_sha(ids),
        rendered_prompt_SHA256=sha(prompt.encode()),source='Original1374 body -> same public prepare_agent_request -> same prepared_chat -> actual processor',
        no_original_raw_prompt_IDs_in_1374_archive=True,HTTP=False,queue_created=False,prompt_cropped=False)
