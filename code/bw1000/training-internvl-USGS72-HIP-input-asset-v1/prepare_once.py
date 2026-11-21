"""Create T03's missing HIP-path train72 input asset from frozen public charts."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import struct
import time
import zlib

G=Path(__file__).resolve().parents[3]
OUT=G/'local-only/training-internvl-USGS72-HIP-input-asset-v1-original'
SOURCE=G/'local-only/training-USGS-multievent-render144-v2-original/dataset'
TARGET='/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-internvl-USGS72-native-CPU-v1/data'
REQUEST='c3dfa76b1764440e8cc8b9dd806cb46a'


def pin(raw):return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())


def need(ok,why):
    if not ok:raise ValueError(why)


def write(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return pin(raw)


def save(path,obj):return write(path,(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())


def read_pinned(path,wanted):
    need(not path.is_symlink() and 0<wanted['bytes']<=2*1024*1024,'bounded source member required')
    before=path.stat()
    with path.open('rb') as f:raw=f.read(wanted['bytes']+1)
    after=path.stat()
    need(pin(raw)==wanted and all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'actual source changed or differs')
    return raw


def main():
    origin=time.time();OUT.mkdir(exist_ok=False)
    save(OUT/'intent.json',dict(request=REQUEST,epoch=origin,deadline=origin+60,planned_rows=72,planned_media=12,
        target_data=TARGET,target_invoked=False,HTTP=0,GPU=0,optimizer_updates=0))
    result=dict(schema='INTERNVL_HIP_USGS72_TRAIN_INPUT_ASSET_LOCAL_V1',request=REQUEST,passed=False,
        planned_rows=72,completed_rows=0,planned_media=12,completed_media=0,actual_target_copy_calls=0,
        native_encodes=0,optimizer_updates=0,HTTP=0,GPU=0,SSH=0,Slurm=0,
        dev_input_gold_or_media_body_read=False,blind_input_gold_or_media_body_read=False,
        READY=False,training_GO=False,GPU_GO=False,Model_GO=False)
    try:
        helper=G/'code/bw1000/training-domain-data-registry-v1/prepare_vlm.py'
        raw=read_pinned(helper,dict(bytes=6137,SHA256='d19a019322730ac740f2e51310fc3621e3f4f8ffd67e1a39b5b6e6511233e57d'))
        ast.parse(raw)
        spec=importlib.util.spec_from_file_location('frozen_five_vlm_data_adapter',helper)
        common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
        manifest=read_pinned(SOURCE/'manifest-private.json',dict(bytes=14864,SHA256='4fbf6f2f8c03731eb84882fd56f63f2c4586f92d5c5d149b9dfd667973f961dc'))
        m=json.loads(manifest)
        need(m['actual_images']==24 and m['actual_split_QA']=={'train':72,'dev':36,'blind':36},'original complete split manifest differs')
        groups={s:set() for s in ('train','dev','blind')};events={s:set() for s in groups}
        for chart in m['charts']:
            groups[chart['split']].add(chart['group']);events[chart['split']].update(chart['event_ids'])
        for a,b in (('train','dev'),('train','blind'),('dev','blind')):
            need(not groups[a]&groups[b] and not events[a]&events[b],'source event/closure split overlap')
        source_names=('train/inputs.jsonl','train/labels-private.jsonl')
        source_raw={n:read_pinned(SOURCE/n,m['outputs'][n]) for n in source_names}
        inputs=[json.loads(x) for x in source_raw[source_names[0]].splitlines()]
        labels=[json.loads(x) for x in source_raw[source_names[1]].splitlines()]
        train,empty=common.join(inputs,labels,[])
        need(len(train)==72 and empty==[] and common.CASES['T03.sft']==4096,'actual common T03 ordered join differs')
        prepared=OUT/'data';prepared.mkdir()
        media={};body_hashes=set()
        for row in train:
            need(time.time()<origin+57,'local data preparation epoch exhausted')
            need(len(row['images'])==1,'single complete native HF image required')
            name=row['images'][0];relative=common.relative(name,'train')
            if name not in media:
                image=read_pinned(SOURCE/name,m['outputs'][name])
                need(image[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',image[16:24])==(512,512),'actual original PNG header differs')
                pos=8;chunks=0
                while pos<len(image):
                    size=int.from_bytes(image[pos:pos+4],'big');end=pos+12+size
                    need(end<=len(image),'truncated PNG chunk')
                    body=image[pos+4:pos+8+size]
                    need(zlib.crc32(body)==int.from_bytes(image[pos+8+size:end],'big'),'PNG chunk CRC differs')
                    chunks+=1;pos=end
                need(pos==len(image) and pin(image)['SHA256'] not in body_hashes,'PNG identity repeats')
                body_hashes.add(pin(image)['SHA256'])
                write(prepared/name,image)
                media[name]=dict(m['outputs'][name],target_path=str(PurePosixPath(TARGET)/relative),
                    source_path=str(SOURCE/name),target_presence_verified=False,source_dimensions=[512,512],PNG_chunks=chunks)
                result['completed_media']+=1
            row['images']=[media[name]['target_path']]
            need([v['role'] for v in row['messages']]==['user','assistant']
                 and row['messages'][1]['content'].startswith('FINAL: '),'supervised FINAL format differs')
        need(len(media)==12,'full source media denominator differs')
        data=b''.join(common.canonical(row) for row in train)
        input_pin=write(prepared/'train-swift.jsonl',data)
        roundtrip=[json.loads(line) for line in data.splitlines()]
        need(len(roundtrip)==72 and roundtrip==train and len({r['id'] for r in roundtrip})==72,'output complete identity/order differs')
        binding=dict(schema='T03_HIP_TRAIN_ONLY_DOMAIN_INPUT_ASSET_V1',case='T03.sft',max_sequence_length=4096,
            source_manifest=pin(manifest),source_train_files={n:m['outputs'][n] for n in source_names},
            files={'train-swift.jsonl':dict(input_pin,rows=72)},media=media,target_data=TARGET,
            training_messages='original full user/image + assistant FINAL from same ordered common adapter join',
            source_order_unchanged=True,literal_special_tokens_added=False,truncation_performed=False,
            expected_native_model_type='internvl',expected_processor_template='internvl_hf',expected_template_type_argument=None,
            original_attention='sdpa',original_language_GC=True,freeze_visual_backbone=True,
            native_EOS_mask_and_selected_logits_qualification='PENDING_ACTUAL_TARGET',
            native_encodes=0,target_presence_verified=False,dev_body_read=False,blind_body_read=False,
            training_GO=False,GPU_GO=False,Model_GO=False)
        binding_pin=save(prepared/'data-binding-private.json',binding)
        result.update(passed=True,state='T03_HIP_TRAIN72_REBOUND_ASSET_LOCAL_PASS',completed_rows=72,
            unique_input_IDs=72,unique_media_SHA256=12,source_manifest_SHA256=pin(manifest)['SHA256'],
            source_group_counts={s:len(x) for s,x in groups.items()},source_event_counts={s:len(x) for s,x in events.items()},
            image_body_bytes=sum(x['bytes'] for x in media.values()),original_image_bytes_unchanged=True,
            exact_common_ordered_join=True,common_adapter=pin(raw),input_file=input_pin,binding_file=binding_pin,
            source_input_identity_sequence_SHA256=pin(common.canonical([pin(common.canonical(r))['SHA256'] for r in inputs]))['SHA256'],
            assistant_FINAL_rows=72,literal_EOS_added=False,target_data=TARGET,actual_native72_HIP_qualified=False)
    except Exception as e:
        result.update(state='FAILED_PARTIAL_PRESERVED',error_type=type(e).__name__,reason=str(e)[:256])
    result['wall_seconds']=time.time()-origin
    save(OUT/'result.json',result)
    print(json.dumps({k:result[k] for k in ('passed','planned_rows','completed_rows','planned_media','completed_media','READY','wall_seconds')},sort_keys=True))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
