"""Decode only a completed private source bundle into a new ignored local directory."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

def consume(bundle,output):
    if bundle.stat().st_size>768*1024:raise ValueError('private bundle bound')
    raw=bundle.read_bytes();value=json.loads(raw)
    if value['endpoint'] not in ('hygon_declared_base','a100_original') or value['planned_files']!=2:raise ValueError('exact endpoint/source denominator')
    rows=value['files']
    if [r['name'] for r in rows]!=['modeling_internvl.py','modeling_qwen3.py']:raise ValueError('exact two source names')
    output.mkdir(mode=0o700);files=[];total=0;actual_read=0;attempted=0
    for row in rows:
        if type(row['attempted']) is not bool or type(row['actual_read_bytes']) is not int or not 0<=row['actual_read_bytes']<=262145:
            raise ValueError('bounded actual file-read attempt counter')
        if row['attempted'] is False and row['actual_read_bytes']!=0:raise ValueError('unattempted file has read bytes')
        actual_read+=row['actual_read_bytes'];attempted+=row['attempted']
        public={k:row.get(k) for k in ('name','state','bytes','SHA256','error_type','errno','reason_SHA256','source_kind','attempted','actual_read_bytes')}
        if row['state']=='COMPLETE_STABLE_SOURCE':
            body=base64.b64decode(row['body_base64'],validate=True);total+=len(body)
            if len(body)!=row['bytes'] or not 0<len(body)<=262144 or hashlib.sha256(body).hexdigest()!=row['SHA256'] or total>524288:
                raise ValueError('transported private source identity differs')
            if row['stable_path_fd_identity'] is not True:raise ValueError('source stability absent')
            with (output/row['name']).open('xb') as stream:stream.write(body)
        elif row.get('body_base64') is not None:raise ValueError('failed source unexpectedly exported a body')
        files.append(public)
    count=sum(r['state']=='COMPLETE_STABLE_SOURCE' for r in rows)
    if count!=value['completed_files'] or value['missing_or_failed_files']!=2-count or total!=value['source_body_bytes']:raise ValueError('complete/failed/source-byte denominator differs')
    if actual_read!=value['actual_source_bytes_read'] or actual_read>524290 or attempted!=value['actual_attempted_files']:
        raise ValueError('actual partial bytecounts/attempt denominator differs')
    memory=value['initial_memory'];vm=memory['VmSize_bytes'];rss=memory['VmRSS_bytes'];limit=memory['AS_limit_bytes']
    if type(vm) is not int or type(rss) is not int or min(vm,rss)<0 or limit!=max(268435456,vm+67108864):raise ValueError('initial memory/AS formula differs')
    admitted=vm<=1073741824 and limit<=2147483648
    if memory['admitted'] is not admitted or memory['AS_is_RSS'] is not False:raise ValueError('AS admission/measurement distinction differs')
    if not admitted and attempted:raise ValueError('source body attempted after memory admission failed')
    result=dict(state='LOCAL_SOURCE_BUNDLE_IDENTITY_VERIFIED',endpoint=value['endpoint'],original_request=value['request_id'],original_epoch=value['epoch'],
        bundle_bytes=len(raw),bundle_SHA256=hashlib.sha256(raw).hexdigest(),planned_files=2,completed_files=count,failed_files=2-count,
        source_body_bytes=total,actual_source_bytes_read=actual_read,actual_attempted_files=attempted,initial_memory=memory,files=files,
        original_CPU357_actual_loaded_provider_identity='NOT_PROVEN',original_CPU357_failure_preserved=True,source_bodies_private_not_Git=True,new_model_imports_or_training=0)
    with (output/'summary.json').open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(consume(a.bundle,a.output),sort_keys=True))
