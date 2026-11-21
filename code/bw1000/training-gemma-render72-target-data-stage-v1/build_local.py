"""Build one private train-only packet and test its actual local installation."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import time

D=Path(__file__).resolve().parent
G=D.parents[2]
OUT=G/'local-only/training-gemma-render72-target-data-stage-v1-original'


def pin(raw):return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())


def main():
    origin=time.time();OUT.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('new_gemma_data_receiver',D/'receive_once.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    inputs=json.loads((D/'INPUTS.json').read_bytes())
    h={'request_id':r.REQUEST,'target':r.TARGET,'files':[
        {'name':x['target_relative'],'bytes':x['bytes'],'SHA256':x['SHA256']} for x in inputs['files']]}
    encoded=json.dumps(h,sort_keys=True,separators=(',',':')).encode()
    prefix=r.MAGIC+struct.pack('<I',len(encoded))+encoded
    payload=OUT/'payload.private.bin';logical=0
    with payload.open('xb') as f:
        f.write(prefix)
        for item in inputs['files']:
            r.need(time.time()<origin+60,'local preparation epoch exhausted')
            p=G/item['source_relative'];r.need(not p.is_symlink(),'source alias')
            raw=p.read_bytes();r.need(pin(raw)=={k:item[k] for k in ('bytes','SHA256')},'actual source differs')
            f.write(raw);logical+=len(raw)
    body=payload.read_bytes();r.need(len(body)<r.MAX_BODY+16384,'packet exceeds bound')
    with payload.open('rb') as stream:
        actual=r.install(stream,OUT/'roundtrip',inputs,origin,local_test=True)
    r.need(actual['passed'] and actual['completed_files']==16,'actual data installation rehearsal failed')
    refused=[]
    try:r.install(io.BytesIO(body),OUT/'roundtrip',inputs,origin,local_test=True)
    except ValueError:refused.append('existing_directory_no_overwrite')
    else:raise AssertionError('existing directory accepted')
    bad=json.loads(encoded);bad['files'][0]['name']='../escape'
    encoded_bad=json.dumps(bad,sort_keys=True,separators=(',',':')).encode()
    try:r.install(io.BytesIO(r.MAGIC+struct.pack('<I',len(encoded_bad))+encoded_bad),OUT/'bad-path',inputs,origin,local_test=True)
    except ValueError:refused.append('changed_or_escaping_header_before_directory_creation')
    else:raise AssertionError('modified header accepted')
    r.need(not (OUT/'bad-path').exists(),'unsafe header created target directory')
    short=r.install(io.BytesIO(body[:-1]),OUT/'truncated',inputs,origin,local_test=True)
    r.need(not short['passed'] and short['completed_files']==15,'truncation failure denominator lost')
    refused.append('truncated_body_retains_15_of_16_partial')
    result={'schema':'GEMMA72_REAL_TRAIN_DATA_PACKET_AND_LOCAL_RECEIVER_V1','passed':True,'request_id':r.REQUEST,
        'planned_files':16,'actual_exact_source_files':16,'actual_payload':pin(body),'logical_source_bytes':logical,
        'actual_local_installation_files':16,'train_rows':72,'train_media':12,'view_rows':800,
        'new_independent_examples':0,'actual_new_native_encodes':0,'actual_optimizer_updates':0,
        'dev_or_blind_data_bodies':0,'SSH':0,'Slurm':0,'GPU':0,'actual_target_installation':False,
        'local_normal_cases':1,'local_rejected_cases':refused,'READY':False,'training_GO':False,
        'wall_seconds':time.time()-origin,'source_inputs':pin((D/'INPUTS.json').read_bytes()),
        'receiver_source':pin((D/'receive_once.py').read_bytes()),'original_CPU_request':'8d6d573cc64b4ab3b082b17682043a11',
        'CPU_original_intent_missing_does_not_prove_not_executed':True}
    r.json_file(OUT/'result.json',result)
    print(json.dumps({k:result[k] for k in ('passed','actual_exact_source_files','actual_local_installation_files','train_rows','view_rows','actual_target_installation','READY','wall_seconds')},sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
